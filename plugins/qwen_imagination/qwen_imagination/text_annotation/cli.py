from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import urllib.request
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[2]
SALT_ROOT = PLUGIN_ROOT.parents[2]
SRC_ROOT = SALT_ROOT / "src"
for candidate in (SALT_ROOT, SRC_ROOT, PLUGIN_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from ..regional.roi import (  # noqa: E402
    HumanROIGenerator,
    SCHPLIPBackend,
    SegmentAnythingBackend,
    UltralyticsPoseBackend,
)
from ..regional.runtime import OfficialSwinIRBackend  # noqa: E402
from ..regional.schema import SourceItem  # noqa: E402
from ..regional.sysu_sources import (  # noqa: E402
    collect_test_source_records,
    load_train_source_records,
)
from .config import TextAnnotationConfig, load_text_annotation_config  # noqa: E402
from .manifest import (  # noqa: E402
    atomic_json,
    consolidate_shard,
    load_source_record,
    save_source_record,
    valid_cached_record,
)
from .pipeline import TextAnnotationPipeline  # noqa: E402
from .reasoner import TextAnnotationReasoner  # noqa: E402
from .track_anchor import (  # noqa: E402
    PrecomputedSwinIRStore,
    TrackAnchorTextAnnotationPipeline,
)


def _load_category_stats(config: TextAnnotationConfig) -> dict:
    value = config.roi.get("category_stats_path")
    if not value:
        return {}
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_roi(config: TextAnnotationConfig) -> HumanROIGenerator:
    device = str(config.roi.get("device", "cuda:0"))
    return HumanROIGenerator(
        pose=UltralyticsPoseBackend(config.assets["yolo_pose"], device=device),
        parsing=SCHPLIPBackend(
            config.roi["schp_root"], config.assets["schp_lip"], device=device
        ),
        sam=SegmentAnythingBackend(
            config.assets["sam_vit_b"],
            device=device,
            repository=config.roi["sam_root"],
        ),
        strict=True,
    )


def _build_reasoner(config: TextAnnotationConfig) -> TextAnnotationReasoner:
    return TextAnnotationReasoner(
        endpoint=str(config.qwen["endpoint"]),
        model_id=str(config.qwen["model_id"]),
        timeout_seconds=float(config.qwen.get("timeout_seconds", 360)),
        enable_thinking=bool(config.qwen.get("thinking_mode", False)),
        reasoning_effort=str(config.qwen.get("reasoning_effort", "none")),
        temperature=float(config.qwen.get("temperature", 0.35)),
        max_tokens=int(config.qwen.get("max_tokens", 2048)),
        response_profile=str(config.qwen.get("response_profile", "detailed_v1")),
        roi_board_size_px=int(config.roi_board_size_px),
    )


def build_pipeline(
    config: TextAnnotationConfig,
) -> TextAnnotationPipeline | TrackAnchorTextAnnotationPipeline:
    roi = _build_roi(config)
    reasoner = _build_reasoner(config)
    if config.strategy == "track_anchor":
        return TrackAnchorTextAnnotationPipeline(
            config,
            roi=roi,
            reasoner=reasoner,
            store=PrecomputedSwinIRStore(
                config.dataset_root, config.precomputed_swinir_root
            ),
            category_stats=_load_category_stats(config),
        )
    device = str(config.roi.get("device", "cuda:0"))
    swin = OfficialSwinIRBackend(
        config.swinir["root"], config.assets["swinir_model"], device
    )
    return TextAnnotationPipeline(
        config,
        swin=swin,
        roi=roi,
        reasoner=reasoner,
        category_stats=_load_category_stats(config),
        reference_store=(
            PrecomputedSwinIRStore(config.dataset_root, config.precomputed_swinir_root)
            if config.precomputed_swinir_root is not None
            else None
        ),
    )


def collect_sources(config: TextAnnotationConfig, split: str) -> list[SourceItem]:
    by_modality = {}
    for modality in config.modalities:
        rows = []
        if split in {"train", "all"}:
            rows.extend(
                (record, "train")
                for record in load_train_source_records(config.dataset_root, modality)
            )
        if split in {"evaluation", "all"}:
            rows.extend(
                (record, "evaluation")
                for record in collect_test_source_records(config.dataset_root, modality)
            )
        by_modality[modality] = rows
    if not by_modality:
        return []
    sources = []
    for index in range(max(len(rows) for rows in by_modality.values())):
        for modality in config.modalities:
            rows = by_modality[modality]
            if index >= len(rows):
                continue
            record, source_split = rows[index]
            sources.append(
                SourceItem(
                    source_key=record.source_key,
                    image=config.dataset_root / record.source_key,
                    identity=record.identity,
                    camera=record.camera,
                    modality=modality,
                    split=source_split,
                )
            )
    return sources


def select_shard(
    sources: list[SourceItem], *, shard_index: int, num_shards: int
) -> list[SourceItem]:
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    return [
        source
        for index, source in enumerate(sources)
        if index % int(num_shards) == int(shard_index)
    ]


def group_tracks(sources: list[SourceItem]) -> list[list[SourceItem]]:
    groups: dict[tuple[str, str, str], list[SourceItem]] = {}
    for source in sources:
        key = (source.split, source.camera, source.identity)
        groups.setdefault(key, []).append(source)
    return list(groups.values())


def _failure_record(
    config: TextAnnotationConfig, source: SourceItem, error: Exception
) -> dict:
    return {
        "schema_version": int(config.schema_version),
        "annotation_version": config.annotation_version,
        "run_signature": config.run_signature(),
        "source_key": source.source_key,
        "image": str(source.image),
        "identity": source.identity,
        "camera": source.camera,
        "modality": source.modality,
        "split": source.split,
        "status": "failed",
        "failure": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )[-6000:],
        },
    }


def run(
    config: TextAnnotationConfig,
    *,
    split: str,
    shard_index: int,
    num_shards: int,
    limit: int | None,
    fail_fast: bool,
    overwrite: bool,
    pipeline: TextAnnotationPipeline | TrackAnchorTextAnnotationPipeline | None = None,
) -> dict:
    all_sources = collect_sources(config, split)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    if config.strategy == "track_anchor":
        all_tracks = group_tracks(all_sources)
        tracks = [
            track
            for index, track in enumerate(all_tracks)
            if index % int(num_shards) == int(shard_index)
        ]
        if limit is not None:
            tracks = tracks[:limit]
        sources = [source for track in tracks for source in track]
    else:
        sources = select_shard(
            all_sources,
            shard_index=shard_index,
            num_shards=num_shards,
        )
        if limit is not None:
            sources = sources[:limit]
        tracks = [[source] for source in sources]
    records = []
    cached_count = 0
    cached_track_count = 0
    direct_vlm_record_count = 0
    vlm_request_count = 0
    active_pipeline = pipeline
    source_index = 0
    for track_index, track in enumerate(tracks, start=1):
        cached = [
            None
            if overwrite
            else load_source_record(config.output_root, source.source_key)
            for source in track
        ]
        track_cached = all(
            valid_cached_record(row, config.run_signature()) for row in cached
        )
        if track_cached:
            track_records = cached
            cached_count += len(track)
            cached_track_count += 1
        else:
            try:
                active_pipeline = active_pipeline or build_pipeline(config)
                if config.strategy == "track_anchor":
                    track_records = active_pipeline.process_track(track)
                    vlm_request_count += 1
                else:
                    track_records = [active_pipeline.process(track[0])]
                    vlm_request_count += 1
            except Exception as error:
                track_records = [
                    _failure_record(config, source, error) for source in track
                ]
                for record in track_records:
                    save_source_record(config.output_root, record)
                if fail_fast:
                    raise
            else:
                if len(track_records) != len(track):
                    raise ValueError(
                        f"track pipeline returned {len(track_records)} records for "
                        f"{len(track)} sources"
                    )
                for record in track_records:
                    save_source_record(config.output_root, record)
        for source, record, cached_record in zip(track, track_records, cached):
            source_index += 1
            records.append(record)
            direct_vlm_record_count += int(
                bool(record.get("annotation_provenance", {}).get("direct_vlm"))
            )
            print(
                json.dumps(
                    {
                        "source": source.source_key,
                        "index": source_index,
                        "total": len(sources),
                        "track_index": track_index,
                        "track_total": len(tracks),
                        "status": record["status"],
                        "cached": track_cached and record is cached_record,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
    summary = consolidate_shard(
        config.output_root,
        records,
        split=split,
        shard_index=shard_index,
        num_shards=num_shards,
        expected_source_count=len(sources),
    )
    summary["cached_source_count"] = cached_count
    summary["track_count"] = len(tracks)
    summary["cached_track_count"] = cached_track_count
    summary["direct_vlm_record_count"] = direct_vlm_record_count
    summary["vlm_request_count"] = vlm_request_count
    summary["strategy"] = config.strategy
    summary["run_signature"] = config.run_signature()
    stem = f"{split}.shard-{shard_index:05d}-of-{num_shards:05d}.summary.json"
    atomic_json(config.output_root / "manifests" / stem, summary)
    return summary


def preflight(config: TextAnnotationConfig, *, check_server: bool = True) -> dict:
    if not config.dataset_root.is_dir():
        raise FileNotFoundError(config.dataset_root)
    assets = {}
    for name, path in sorted(config.assets.items()):
        if not path.is_file():
            raise FileNotFoundError(f"missing text annotation asset {name}: {path}")
        assets[name] = {"path": str(path), "bytes": path.stat().st_size}
    for name in ("schp_root", "sam_root"):
        path = Path(config.roi[name]).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
    swin_root = Path(config.swinir["root"]).expanduser().resolve()
    if not swin_root.is_dir():
        raise FileNotFoundError(swin_root)
    if config.precomputed_swinir_root is not None:
        derived = Path(config.precomputed_swinir_root).expanduser().resolve()
        required = [
            derived / "manifest.json",
            derived / "train_rgb_swinir_x2_img.npy",
            derived / "train_ir_swinir_x2_img.npy",
            derived / "eval",
        ]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(path)
    result = {
        "valid": True,
        "plugin": "qwen-text-annotation-v2",
        "dataset_root": str(config.dataset_root),
        "output_root": str(config.output_root),
        "strategy": config.strategy,
        "precomputed_swinir_root": (
            str(config.precomputed_swinir_root)
            if config.precomputed_swinir_root is not None
            else None
        ),
        "assets": assets,
        "run_signature": config.run_signature(),
    }
    if check_server:
        endpoint = str(config.qwen.get("health_endpoint", ""))
        with urllib.request.urlopen(endpoint, timeout=10) as response:
            result["qwen_server"] = {
                "endpoint": endpoint,
                "status": int(response.status),
                "body": response.read().decode("utf-8", errors="replace")[:1000],
            }
    return result


def qwen_server_command(config: TextAnnotationConfig) -> list[str]:
    return [
        str(config.qwen["server_binary"]),
        "--model",
        str(config.assets["qwen_model"]),
        "--mmproj",
        str(config.assets["qwen_mmproj"]),
        "--alias",
        str(config.qwen["model_id"]),
        "--host",
        str(config.qwen.get("host", "127.0.0.1")),
        "--port",
        str(int(config.qwen.get("port", 18080))),
        "--ctx-size",
        str(int(config.qwen.get("context_size", 8192))),
        "--parallel",
        "1",
        "--n-gpu-layers",
        "999",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--reasoning-budget",
        str(int(config.qwen.get("reasoning_budget", 1024))),
        "--flash-attn",
        "on",
        "--offline",
        "--no-webui",
        "--jinja",
    ]


def serve(config: TextAnnotationConfig, *, execute: bool) -> dict:
    command = qwen_server_command(config)
    binary = Path(command[0])
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"llama-server is not executable: {binary}")
    result = {
        "execute": bool(execute),
        "command": command,
        "cuda_visible_devices": str(config.qwen.get("gpu", 0)),
    }
    if execute:
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = result["cuda_visible_devices"]
        os.execvpe(command[0], command, environment)
        raise AssertionError("os.execvpe returned unexpectedly")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dataset-scale SYSU-MM01 global plus regional text annotation"
    )
    parser.add_argument("--config", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="action", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--skip-server", action="store_true")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--split", choices=("train", "evaluation", "all"), default="train"
    )
    run_parser.add_argument("--shard-index", type=int, default=0)
    run_parser.add_argument("--num-shards", type=int, default=1)
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--fail-fast", action="store_true")
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.add_argument("--device")
    run_parser.add_argument("--strategy", choices=("exact", "track_anchor"))
    run_parser.add_argument("--qwen-endpoint")
    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--execute", action="store_true")
    serve_parser.add_argument("--gpu", type=int)
    serve_parser.add_argument("--port", type=int)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_text_annotation_config(args.config)
    if args.action == "preflight":
        result = preflight(config, check_server=not args.skip_server)
    elif args.action == "run":
        if args.strategy is not None:
            config.strategy = str(args.strategy)
            config.validate()
        if args.device is not None:
            config.roi["device"] = str(args.device)
        if args.qwen_endpoint is not None:
            config.qwen["endpoint"] = str(args.qwen_endpoint)
        result = run(
            config,
            split=args.split,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            limit=args.limit,
            fail_fast=args.fail_fast,
            overwrite=args.overwrite,
        )
    else:
        if args.gpu is not None:
            config.qwen["gpu"] = int(args.gpu)
        if args.port is not None:
            config.qwen["port"] = int(args.port)
        result = serve(config, execute=args.execute)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

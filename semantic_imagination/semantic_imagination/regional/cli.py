from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from salt_vi.data.sysu_sources import (  # noqa: E402
    collect_test_source_records,
    load_train_source_records,
)

from .config import RegionalConfig, load_regional_config  # noqa: E402
from .manifest import (  # noqa: E402
    atomic_json,
    consolidate_manifests,
    load_source_record,
    sha256_file,
)
from .pipeline import RegionalImaginationPipeline, category_statistics  # noqa: E402
from .qwen import LlamaServerQwenReasoner  # noqa: E402
from .roi import (  # noqa: E402
    HumanROIGenerator,
    SCHPLIPBackend,
    SegmentAnythingBackend,
    UltralyticsPoseBackend,
)
from .runtime import (  # noqa: E402
    ExistingPASDBackend,
    OfficialSwinIRBackend,
    SALTIdentityBackend,
    validate_assets,
)
from .schema import SourceItem  # noqa: E402


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_stats(config: RegionalConfig) -> dict:
    value = config.roi.get("category_stats_path")
    if not value:
        return {}
    path = _repo_path(value)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_pipeline(config: RegionalConfig) -> RegionalImaginationPipeline:
    device = str(config.roi.get("device", "cuda:3"))
    swin = OfficialSwinIRBackend(
        _repo_path(config.swinir["root"]),
        config.assets["swinir_model"].path,
        device,
    )
    roi = HumanROIGenerator(
        pose=UltralyticsPoseBackend(config.assets["yolo_pose"].path, device=device),
        parsing=SCHPLIPBackend(
            _repo_path(config.roi["schp_root"]),
            config.assets["schp_lip"].path,
            device=device,
        ),
        sam=SegmentAnythingBackend(
            config.assets["sam_vit_b"].path,
            device=device,
            repository=_repo_path(config.roi["sam_root"]),
        ),
        strict=True,
    )
    reasoner = LlamaServerQwenReasoner(
        endpoint=str(config.qwen.get("endpoint", "http://127.0.0.1:8080/v1/chat/completions")),
        model_id=str(config.qwen.get("model_id", "third-party-qwen3.8-27b-ud-q4-k-xl")),
        timeout_seconds=float(config.qwen.get("timeout_seconds", 180)),
        enable_thinking=bool(config.qwen.get("thinking_mode", True)),
        reasoning_effort=str(config.qwen.get("reasoning_effort", "high")),
    )
    pasd = ExistingPASDBackend(
        _repo_path(config.pasd.get("config_path", "pasd_plugin/configs/sysu.yaml")),
        device=str(config.pasd.get("device", device)),
    )
    identity = SALTIdentityBackend(
        _repo_path(config.identity["config_path"]),
        config.assets["identity_checkpoint"].path,
        str(config.identity.get("device", device)),
    )
    return RegionalImaginationPipeline(
        config,
        swin=swin,
        roi=roi,
        reasoner=reasoner,
        pasd=pasd,
        identity=identity,
        category_stats=_load_stats(config),
    )


def _qwen_health(config: RegionalConfig) -> dict:
    endpoint = str(config.qwen.get("health_endpoint", "http://127.0.0.1:8080/health"))
    with urllib.request.urlopen(endpoint, timeout=10) as response:
        body = response.read().decode("utf-8", errors="replace")
    return {"endpoint": endpoint, "status": int(response.status), "body": body[:1000]}


def _git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={path}", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _validate_roi_sources(config: RegionalConfig) -> dict:
    sources = {}
    for name in ("schp", "sam"):
        root = _repo_path(config.roi[f"{name}_root"])
        expected = str(config.roi[f"{name}_revision"])
        actual = _git_revision(root)
        if actual != expected:
            raise ValueError(
                f"QRI {name} source revision mismatch: expected {expected}, got {actual}"
            )
        if subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={root}",
                "-C",
                str(root),
                "status",
                "--porcelain",
            ],
            text=True,
        ).strip():
            raise ValueError(f"QRI {name} source worktree must be clean: {root}")
        sources[name] = {"root": str(root), "revision": actual}
    return sources


def _validate_swinir_source(config: RegionalConfig) -> dict:
    root = _repo_path(config.swinir["root"])
    revision = _git_revision(root)
    expected = str(config.swinir["revision"])
    if revision != expected:
        raise ValueError(
            f"QRI SwinIR source revision mismatch: expected {expected}, got {revision}"
        )
    network = root / "models" / "network_swinir.py"
    digest = sha256_file(network)
    if digest != str(config.swinir["network_sha256"]):
        raise ValueError("QRI SwinIR network implementation checksum mismatch")
    return {"root": str(root), "revision": revision, "network_sha256": digest}


def _validate_pasd_assets(config: RegionalConfig) -> dict:
    from pasd_plugin.config import PluginConfig

    path = _repo_path(config.pasd.get("config_path", "pasd_plugin/configs/sysu.yaml"))
    pasd_config = PluginConfig.from_yaml(path)
    pasd_config.validate_assets()
    return {"config": str(path), "output_contract": pasd_config.output_contract()}


def _qwen_server_command(config: RegionalConfig) -> list[str]:
    binary = _repo_path(config.qwen["server_binary"])
    return [
        str(binary),
        "--model",
        str(config.assets["qwen_model"].path),
        "--mmproj",
        str(config.assets["qwen_mmproj"].path),
        "--alias",
        str(config.qwen["model_id"]),
        "--host",
        str(config.qwen.get("host", "127.0.0.1")),
        "--port",
        str(int(config.qwen.get("port", 8080))),
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
        "--jinja",
    ]


def _validate_qwen_runtime(config: RegionalConfig) -> dict:
    root = _repo_path(config.qwen["llama_root"])
    expected = str(config.qwen["llama_revision"])
    actual = _git_revision(root)
    if actual != expected:
        raise ValueError(
            f"QRI llama.cpp revision mismatch: expected {expected}, got {actual}"
        )
    binary = Path(_qwen_server_command(config)[0])
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"QRI llama-server binary is not executable: {binary}")
    digest = sha256_file(binary)
    expected_digest = str(config.qwen["server_sha256"])
    if digest != expected_digest:
        raise ValueError(
            f"QRI llama-server checksum mismatch: expected {expected_digest}, got {digest}"
        )
    return {
        "root": str(root),
        "revision": actual,
        "binary": str(binary),
        "binary_sha256": digest,
        "command": _qwen_server_command(config),
        "cuda_visible_devices": str(config.qwen.get("gpu", 3)),
    }


def preflight(config: RegionalConfig, *, check_server: bool = True) -> dict:
    assets = validate_assets(config)
    required = {
        "qwen_model",
        "qwen_mmproj",
        "swinir_model",
        "identity_checkpoint",
        "yolo_pose",
        "schp_lip",
        "sam_vit_b",
    }
    missing = sorted(required.difference(config.assets))
    if missing:
        raise ValueError(f"QRI config omits required formal assets: {missing}")
    if not config.dataset_root.is_dir():
        raise FileNotFoundError(config.dataset_root)
    result = {
        "valid": True,
        "plugin": "qwen-regional-imagination-v1",
        "model_status": "third-party checkpoint; not an official Qwen model claim",
        "assets": assets,
        "roi_sources": _validate_roi_sources(config),
        "swinir_source": _validate_swinir_source(config),
        "pasd": _validate_pasd_assets(config),
        "qwen_runtime": _validate_qwen_runtime(config),
        "dataset_root": str(config.dataset_root),
        "output_root": str(config.output_root),
    }
    if check_server:
        result["qwen_server"] = _qwen_health(config)
    return result


def serve(config: RegionalConfig, *, execute: bool) -> dict:
    runtime = _validate_qwen_runtime(config)
    validate_assets(config)
    if not execute:
        return {"execute": False, **runtime}
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = runtime["cuda_visible_devices"]
    os.execvpe(runtime["command"][0], runtime["command"], environment)
    raise AssertionError("os.execvpe returned unexpectedly")


def _sources(
    config: RegionalConfig, limit: int | None, split: str
) -> list[SourceItem]:
    by_modality = {}
    for modality in config.modalities:
        records = []
        if split in {"train", "all"}:
            records.extend(
                (record, "train")
                for record in load_train_source_records(config.dataset_root, modality)
            )
        if split in {"evaluation", "all"}:
            records.extend(
                (record, "evaluation")
                for record in collect_test_source_records(config.dataset_root, modality)
            )
        by_modality[modality] = records
    interleaved = []
    length = max(len(records) for records in by_modality.values())
    for index in range(length):
        for modality in config.modalities:
            records = by_modality[modality]
            if index >= len(records):
                continue
            record, source_split = records[index]
            interleaved.append(
                SourceItem(
                    source_key=record.source_key,
                    image=config.dataset_root / record.source_key,
                    identity=record.identity,
                    camera=record.camera,
                    modality=modality,
                    split=source_split,
                )
            )
            if limit is not None and len(interleaved) >= limit:
                return interleaved
    return interleaved


def _valid_cached(config: RegionalConfig, pipeline, source: SourceItem) -> dict | None:
    record = load_source_record(config.output_root, source.source_key)
    if not record or record.get("build_sha256") != pipeline.build_sha256:
        return None
    for world in record.get("worlds", []):
        path = config.output_root / world["output"]
        if not path.is_file() or sha256_file(path) != world["output_sha256"]:
            return None
    return record


def run(
    config: RegionalConfig, *, limit: int | None, fail_fast: bool, split: str
) -> dict:
    pipeline = build_pipeline(config)
    sources = _sources(config, limit, split)
    records = []
    for index, source in enumerate(sources, start=1):
        cached = _valid_cached(config, pipeline, source)
        record = cached or pipeline.process(source, allow_fallback=not fail_fast)
        records.append(record)
        print(
            json.dumps(
                {
                    "source": source.source_key,
                    "index": index,
                    "total": len(sources),
                    "fallback": record["fallback"],
                    "worlds": len(record["worlds"]),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
    stats = category_statistics(records)
    stats_path = config.output_root / "manifests" / "category_u_swin_stats.json"
    atomic_json(stats_path, stats)
    summaries = consolidate_manifests(
        config.output_root,
        records,
        expected_source_count=len(sources),
        build_sha256=pipeline.build_sha256,
    )
    return {
        "complete": True,
        "source_count": len(records),
        "split": split,
        "fallback_source_count": sum(bool(record["fallback"]) for record in records),
        "category_stats": str(stats_path),
        "manifests": summaries,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen Regional Imagination v1 offline pipeline")
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("preflight", "run", "serve"):
        command = subparsers.add_parser(action)
        command.add_argument("--config", required=True, type=Path)
    subparsers.choices["preflight"].add_argument("--skip-server", action="store_true")
    subparsers.choices["run"].add_argument("--limit", type=int)
    subparsers.choices["run"].add_argument("--fail-fast", action="store_true")
    subparsers.choices["run"].add_argument("--category-stats", type=Path)
    subparsers.choices["run"].add_argument(
        "--split", choices=("train", "evaluation", "all"), default="train"
    )
    subparsers.choices["run"].add_argument(
        "--device", choices=("cuda:1", "cuda:2", "cuda:3")
    )
    subparsers.choices["serve"].add_argument("--execute", action="store_true")
    subparsers.choices["serve"].add_argument("--gpu", type=int, choices=(1, 2, 3))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_regional_config(args.config)
    if args.action == "preflight":
        result = preflight(config, check_server=not args.skip_server)
    elif args.action == "run":
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be positive")
        if args.device is not None:
            config.roi["device"] = args.device
            config.pasd["device"] = args.device
            config.identity["device"] = args.device
        if args.category_stats is not None:
            stats_path = args.category_stats.expanduser().resolve()
            if not stats_path.is_file():
                raise FileNotFoundError(stats_path)
            config.roi["category_stats_path"] = str(stats_path)
        result = run(
            config,
            limit=args.limit,
            fail_fast=args.fail_fast,
            split=args.split,
        )
    else:
        if args.gpu is not None:
            config.qwen["gpu"] = args.gpu
        result = serve(config, execute=args.execute)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

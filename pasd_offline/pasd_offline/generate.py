from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterator

from PIL import Image, ImageStat

from .config import GenerationConfig
from .runtime import PASDGenerator
from .tasks import GenerationTask, group_tasks_by_source


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_png(path: Path, image: Image.Image, compress_level: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.png")
    try:
        image.save(temporary, format="PNG", compress_level=compress_level)
        with Image.open(temporary) as check:
            check.verify()
        with Image.open(temporary) as check:
            if check.size != image.size or check.mode != "RGB":
                raise ValueError(f"invalid temporary PNG contract: {temporary}")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolved_output(task: GenerationTask, output_root: Path) -> Path:
    return task.output if task.output.is_absolute() else output_root / task.output


def _source_metadata_path(output_root: Path, source_key: str) -> Path:
    source = Path(source_key)
    return output_root / "metadata" / source.parent / f"{source.stem}.json"


def _source_lock_path(output_root: Path, source_key: str) -> Path:
    source = Path(source_key)
    return output_root / ".locks" / source.parent / f"{source.stem}.lock"


@contextlib.contextmanager
def _source_lock(path: Path) -> Iterator[bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+", encoding="utf-8")
    locked = False
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
            stream.seek(0)
            stream.truncate()
            stream.write(json.dumps({"pid": os.getpid(), "time": time.time()}))
            stream.flush()
        except BlockingIOError:
            pass
        yield locked
    finally:
        if locked:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _task_contract(tasks: list[GenerationTask]) -> str:
    payload = [
        {
            "source_key": task.source_key,
            "view_index": task.view_index,
            "caption": task.caption,
            "seed": task.seed,
            "output": str(task.output),
        }
        for task in tasks
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_is_complete(tasks: list[GenerationTask], output_root: Path) -> bool:
    source_key = tasks[0].source_key or str(tasks[0].image)
    marker_path = _source_metadata_path(output_root, source_key)
    if not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("task_contract_sha256") != _task_contract(tasks):
            return False
        if len(marker.get("views", [])) != len(tasks):
            return False
        return all(_resolved_output(task, output_root).is_file() for task in tasks)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def generate_task(generator: PASDGenerator, task: GenerationTask, output_root: Path) -> dict:
    output_path = _resolved_output(task, output_root)
    image = generator.generate(task.image, task.caption, task.seed)
    compress_level = int(getattr(getattr(generator, "config", None), "png_compress_level", 4))
    _atomic_png(output_path, image.convert("RGB"), compress_level)
    return {
        **{key: str(value) if isinstance(value, Path) else value for key, value in asdict(task).items()},
        "output": str(output_path),
        "input_sha256": sha256(task.image),
        "output_sha256": sha256(output_path),
        "output_size": list(image.size),
    }


def generate_source_group(
    generator: PASDGenerator,
    tasks: list[GenerationTask],
    output_root: Path,
    batch_size: int,
    physical_gpu: int | None = None,
) -> dict:
    if not tasks:
        raise ValueError("source task group is empty")
    source_key = tasks[0].source_key or str(tasks[0].image)
    if len(tasks) != 5 or [task.view_index for task in tasks] != list(range(5)):
        raise ValueError(f"five-view task contract is invalid for {source_key}")
    images, geometry = generator.generate_views(
        tasks[0].image,
        [task.caption for task in tasks],
        [task.seed for task in tasks],
        modality=tasks[0].modality,
        batch_size=batch_size,
    )
    if len(images) != len(tasks):
        raise RuntimeError(f"PASD returned {len(images)} images for {len(tasks)} tasks")
    input_digest = sha256(tasks[0].image)
    views = []
    for task, image in zip(tasks, images):
        output_path = _resolved_output(task, output_root)
        image = image.convert("RGB")
        if image.size != (generator.config.target_width, generator.config.target_height):
            raise ValueError(f"PASD output has wrong size {image.size}: {source_key}")
        if max(ImageStat.Stat(image).var) <= 0:
            raise ValueError(f"PASD output is constant: {source_key} view={task.view_index}")
        _atomic_png(output_path, image, generator.config.png_compress_level)
        views.append(
            {
                "view_index": task.view_index,
                "caption": task.caption,
                "seed": task.seed,
                "output": str(output_path.relative_to(output_root)),
                "output_sha256": sha256(output_path),
                "output_size": list(image.size),
            }
        )
    marker = {
        "schema_version": 2,
        "source_key": source_key,
        "image": str(tasks[0].image),
        "input_sha256": input_digest,
        "identity": tasks[0].identity,
        "camera": tasks[0].camera,
        "modality": tasks[0].modality,
        "split": tasks[0].split,
        "physical_gpu": physical_gpu,
        "batch_size": int(batch_size),
        "num_inference_steps": generator.config.num_inference_steps,
        "geometry": geometry,
        "task_contract_sha256": _task_contract(tasks),
        "views": views,
        "completed_at_unix": time.time(),
    }
    _atomic_json(_source_metadata_path(output_root, source_key), marker)
    return marker


def generate_worker(
    config: GenerationConfig,
    tasks: list[GenerationTask],
    batch_size: int,
    physical_gpu: int,
    max_sources: int | None = None,
    contention_check: Callable[[], bool] | None = None,
) -> dict:
    if physical_gpu not in config.gpu_allowlist or physical_gpu == 0:
        raise ValueError(f"physical GPU {physical_gpu} is outside the allowed set {config.gpu_allowlist}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible.split(",")[0].strip() != str(physical_gpu):
        raise RuntimeError(
            f"worker GPU mapping mismatch: physical={physical_gpu} CUDA_VISIBLE_DEVICES={visible!r}"
        )
    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    generator = PASDGenerator(config)
    if batch_size <= 0:
        first = tasks[0]
        benchmark = generator.benchmark_batches(
            first.image,
            first.caption,
            first.seed,
            memory_limit_gib=config.min_free_memory_gib,
        )
        batch_size = int(benchmark["selected_batch_size"])
        _atomic_json(output_root / "benchmarks" / f"gpu{physical_gpu}.json", benchmark)
    completed = skipped = locked = 0
    journal_path = output_root / "journals" / f"worker-gpu{physical_gpu}-{os.getpid()}.jsonl"
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("a", encoding="utf-8") as journal:
        for group in group_tasks_by_source(tasks):
            if max_sources is not None and completed >= max_sources:
                break
            if source_is_complete(group, output_root):
                skipped += 1
                continue
            source_key = group[0].source_key or str(group[0].image)
            with _source_lock(_source_lock_path(output_root, source_key)) as acquired:
                if not acquired:
                    locked += 1
                    continue
                if source_is_complete(group, output_root):
                    skipped += 1
                    continue
                marker = generate_source_group(
                    generator, group, output_root, batch_size, physical_gpu=physical_gpu
                )
                journal.write(json.dumps(marker, ensure_ascii=False) + "\n")
                journal.flush()
                completed += 1
            if (
                contention_check is not None
                and completed % config.worker_chunk_size == 0
                and contention_check()
            ):
                return {
                    "status": "resource_contention",
                    "completed": completed,
                    "skipped": skipped,
                    "locked": locked,
                }
    return {"status": "complete_scan", "completed": completed, "skipped": skipped, "locked": locked}


def consolidate_manifest(output_root: str | Path, expected_sources: int | None = None) -> dict:
    output_root = Path(output_root).expanduser().resolve()
    markers = []
    manifest_path = output_root / "manifest.jsonl"
    temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as stream:
        for path in sorted((output_root / "metadata").rglob("*.json")):
            marker = json.loads(path.read_text(encoding="utf-8"))
            markers.append(marker)
            for view in marker["views"]:
                row = {
                    key: marker[key]
                    for key in ("source_key", "identity", "camera", "modality", "split")
                }
                row.update(view)
                encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                    "utf-8"
                )
                stream.write(encoded)
                digest.update(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, manifest_path)
    summary = {
        "schema_version": 2,
        "source_count": len(markers),
        "view_count": sum(len(marker["views"]) for marker in markers),
        "expected_source_count": expected_sources,
        "complete": expected_sources is None or len(markers) == expected_sources,
        "manifest_jsonl": str(manifest_path),
        "manifest_jsonl_sha256": digest.hexdigest(),
    }
    _atomic_json(output_root / "manifest.json", summary)
    return summary


def generate_batch(config: GenerationConfig, tasks: list[GenerationTask]) -> list[dict]:
    """Compatibility sequential entry point used by the small CLI."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    generator = PASDGenerator(config)
    entries = []
    groups = group_tasks_by_source(tasks)
    if all(len(group) == 5 for group in groups):
        for group in groups:
            entries.append(generate_source_group(generator, group, config.output_root, 1))
        consolidate_manifest(config.output_root, expected_sources=len(groups))
        return entries
    for task in tasks:
        entries.append(generate_task(generator, task, config.output_root))
    return entries

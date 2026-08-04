#!/usr/bin/env python3
"""Build verified SYSU-MM01 SwinIR x2 assets without touching source data.

SwinIR inference is deliberately FP32.  This is a data-production pipeline,
not a throughput benchmark: a non-finite model result must stop the build
before a single byte can be written as an apparently valid uint8 image.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SOURCE_ARRAYS = {
    "rgb": "train_rgb_resized_img.npy",
    "ir": "train_ir_resized_img.npy",
}
SOURCE_LABELS = {
    "rgb": "train_rgb_resized_label.npy",
    "ir": "train_ir_resized_label.npy",
}
OUTPUT_ARRAYS = {
    "rgb": "train_rgb_swinir_x2_img.npy",
    "ir": "train_ir_swinir_x2_img.npy",
}
MODALITY_CAMERAS = {
    "rgb": ("cam1", "cam2", "cam4", "cam5"),
    "ir": ("cam3", "cam6"),
}
BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
SOURCE_RESAMPLING = {
    "bicubic": BICUBIC,
    "bilinear": BILINEAR,
}
GIB = 1024 ** 3
SCHEMA_VERSION = 4
NUMERIC_POLICY = "fp32-no-autocast-finite-before-uint8-v1"
BUILD_CONTRACT_FILENAME = ".build-contract.json"
OFFICIAL_SWINIR_REVISION = "6545850fbf8df298df73d81f3e8cba638787c8bd"
OFFICIAL_SWINIR_MODEL_SHA256 = "2032ebf8f401dd3ce2fae5f3852117cb72101ec6ed8358faa64c2a3fa09ed4ac"
OFFICIAL_SWINIR_NETWORK_SHA256 = "9e143898679ebeebc5d2fc94ad1b89c38aa4a4d43da4e0fcba0f93e476994913"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--swinir-root", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--modalities", nargs="+", choices=("rgb", "ir"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--source-size", nargs=2, type=int, default=(288, 144), metavar=("H", "W"))
    parser.add_argument(
        "--source-resampling",
        choices=tuple(SOURCE_RESAMPLING),
        default="bicubic",
        help="Interpolation used to define the common LR source before SwinIR.",
    )
    parser.add_argument("--min-free-before-gb", type=float, default=40.0)
    parser.add_argument("--min-free-after-gb", type=float, default=20.0)
    parser.add_argument("--smoke-only", action="store_true",
                        help="Run the real-checkpoint FP32 smoke gate without creating assets.")
    return parser.parse_args()


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(paths, root):
    digest = hashlib.sha256()
    count = 0
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
        count += 1
    return {"sha256": digest.hexdigest(), "file_count": count}


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def canonical_sha256(payload):
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def git_revision(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def assert_clean_swinir_source(swinir_root):
    """Bind generation to a clean Git revision of the imported SwinIR source."""
    root = Path(swinir_root).resolve()
    revision = git_revision(root)
    if revision is None:
        raise RuntimeError(f"SwinIR source must be a Git worktree: {root}")
    checks = (
        ("tracked worktree", ("diff", "--name-only")),
        ("staged", ("diff", "--cached", "--name-only")),
        ("untracked Python", ("ls-files", "--others", "--exclude-standard", "--", "*.py")),
    )
    dirty = {}
    for label, args in checks:
        paths = subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip().splitlines()
        if paths:
            dirty[label] = paths
    if dirty:
        details = "; ".join(f"{label}: {paths}" for label, paths in dirty.items())
        raise RuntimeError(f"SwinIR source must be clean: {details}")
    return revision


def official_swinir_identity_errors(revision, model_sha256, implementation):
    expected = {
        "revision": OFFICIAL_SWINIR_REVISION,
        "model_sha256": OFFICIAL_SWINIR_MODEL_SHA256,
        "network_sha256": OFFICIAL_SWINIR_NETWORK_SHA256,
    }
    observed = {
        "revision": revision,
        "model_sha256": model_sha256,
        "network_sha256": (implementation or {}).get("network_sha256"),
    }
    return [
        f"official SwinIR {key} mismatch: expected {expected[key]}, observed {observed[key]}"
        for key in expected
        if observed[key] != expected[key]
    ]


def assert_official_swinir_identity(revision, model_sha256, implementation):
    errors = official_swinir_identity_errors(revision, model_sha256, implementation)
    if errors:
        raise RuntimeError("; ".join(errors))


def load_swinir(swinir_root, model_path, device):
    swinir_root = Path(swinir_root).resolve()
    sys.path.insert(0, str(swinir_root))
    module = importlib.import_module("models.network_swinir")
    network_file = Path(module.__file__).resolve()
    try:
        network_file.relative_to(swinir_root)
    except ValueError as error:
        raise RuntimeError(
            f"Imported SwinIR module is outside requested root: {network_file}"
        ) from error
    model = module.SwinIR(
        upscale=2,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )
    payload = torch.load(str(model_path), map_location="cpu")
    state = payload.get("params_ema", payload.get("params", payload)) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=True)
    model.float().eval().to(device)
    implementation = {
        "network_file": str(network_file),
        "network_sha256": sha256_file(network_file),
        "torch_version": torch.__version__,
        "timm_version": importlib.metadata.version("timm"),
    }
    return model, implementation


def normalize_ir(images):
    luminance = np.rint(
        images[..., 0] * 0.299 + images[..., 1] * 0.587 + images[..., 2] * 0.114
    ).astype(np.uint8)
    return np.repeat(luminance[..., None], 3, axis=-1)


def prepare_images(images, modality):
    images = np.asarray(images)
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected uint8 NHWC images, got {images.dtype} {images.shape}")
    if modality == "ir":
        images = normalize_ir(images)
    return torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float().div_(255.0)


def assert_finite_output(tensor, expected_hw=None):
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError(f"Expected NCHW SwinIR tensor, got {type(tensor)} {getattr(tensor, 'shape', None)}")
    if expected_hw is not None and tuple(tensor.shape[-2:]) != tuple(expected_hw):
        raise ValueError(f"SwinIR output size {tuple(tensor.shape[-2:])} != expected {tuple(expected_hw)}")
    finite = torch.isfinite(tensor)
    if not bool(finite.all().item()):
        invalid = int((~finite).sum().item())
        raise FloatingPointError(f"SwinIR produced {invalid}/{tensor.numel()} non-finite values")


def finalize_images(tensor, modality, expected_hw=None):
    assert_finite_output(tensor, expected_hw=expected_hw)
    images = (
        tensor.detach().float().clamp_(0.0, 1.0).mul_(255.0).round_()
        .byte().permute(0, 2, 3, 1).cpu().numpy()
    )
    if modality == "ir":
        gray = np.rint(images.astype(np.float32).mean(axis=-1)).astype(np.uint8)
        images = np.repeat(gray[..., None], 3, axis=-1)
    return images


def infer(model, images, modality, device):
    inputs = prepare_images(images, modality).to(device, non_blocking=True)
    if inputs.dtype != torch.float32:
        raise TypeError(f"SwinIR inputs must be float32, got {inputs.dtype}")
    # Explicitly disable autocast so an ambient caller context cannot change
    # the numeric policy of the derived dataset.
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        outputs = model(inputs.float())
    expected_hw = (images.shape[1] * 2, images.shape[2] * 2)
    return finalize_images(outputs, modality, expected_hw=expected_hw)


def resize_source_images(images, source_size, source_resampling="bicubic"):
    """Standardize source arrays to the same LR size used by every SR group."""
    images = np.asarray(images)
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected uint8 NHWC images, got {images.dtype} {images.shape}")
    source_h, source_w = source_size
    interpolation = SOURCE_RESAMPLING[source_resampling]
    if images.shape[1:3] == (source_h, source_w):
        return np.ascontiguousarray(images)
    return np.stack([
        np.asarray(Image.fromarray(image).resize((source_w, source_h), interpolation), dtype=np.uint8)
        for image in images
    ])


def empty_content_stats():
    return {
        "image_count": 0,
        "value_count": 0,
        "nonzero_value_count": 0,
        "all_zero_image_count": 0,
        "constant_image_count": 0,
        "minimum": 255,
        "maximum": 0,
        "sum": 0,
        "sum_squares": 0,
    }


def update_content_stats(stats, images):
    images = np.asarray(images)
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected uint8 NHWC content, got {images.dtype} {images.shape}")
    flat = images.reshape(images.shape[0], -1)
    stats["image_count"] += int(images.shape[0])
    stats["value_count"] += int(images.size)
    stats["nonzero_value_count"] += int(np.count_nonzero(images))
    stats["all_zero_image_count"] += int(np.count_nonzero(np.max(flat, axis=1) == 0))
    stats["constant_image_count"] += int(np.count_nonzero(np.ptp(flat, axis=1) == 0))
    stats["minimum"] = min(stats["minimum"], int(images.min()))
    stats["maximum"] = max(stats["maximum"], int(images.max()))
    values = images.astype(np.uint64, copy=False)
    stats["sum"] += int(values.sum(dtype=np.uint64))
    stats["sum_squares"] += int(np.square(values, dtype=np.uint64).sum(dtype=np.uint64))


def finalize_content_stats(stats):
    stats = dict(stats)
    count = stats["value_count"]
    if count == 0:
        raise ValueError("Cannot finalize empty content statistics")
    mean = stats["sum"] / count
    variance = max(stats["sum_squares"] / count - mean * mean, 0.0)
    stats["mean"] = mean
    stats["std"] = variance ** 0.5
    stats["nonzero_fraction"] = stats["nonzero_value_count"] / count
    return stats


def assert_non_degenerate(stats, label):
    if stats["all_zero_image_count"]:
        raise ValueError(f"{label} contains {stats['all_zero_image_count']} all-zero images")
    if stats["constant_image_count"]:
        raise ValueError(f"{label} contains {stats['constant_image_count']} constant images")
    if stats["maximum"] - stats["minimum"] < 16 or stats["std"] < 1.0:
        raise ValueError(f"{label} has degenerate dynamic range: {stats}")
    if stats["nonzero_fraction"] < 0.01:
        raise ValueError(f"{label} has implausibly sparse content: {stats}")


def array_content_stats(array, stop=None, chunk_size=64):
    stats = empty_content_stats()
    stop = array.shape[0] if stop is None else int(stop)
    for start in range(0, stop, chunk_size):
        update_content_stats(stats, np.asarray(array[start:min(start + chunk_size, stop)]))
    return finalize_content_stats(stats)


def progress_contract(source_path, model_sha256, modality, expected_shape, build_contract_sha256):
    return {
        "schema_version": SCHEMA_VERSION,
        "numeric_policy": NUMERIC_POLICY,
        "source_path": str(source_path.resolve()),
        "source_sha256": sha256_file(source_path),
        "model_sha256": model_sha256,
        "build_contract_sha256": build_contract_sha256,
        "modality": modality,
        "shape": list(expected_shape),
    }


def progress_path(output_path):
    return output_path.with_name(output_path.name + ".progress.json")


def build_array(
    model, source_path, output_path, modality, source_size, batch_size, device,
    model_sha256, build_contract_sha256, source_resampling="bicubic",
):
    source = np.load(source_path, mmap_mode="r")
    source_h, source_w = source_size
    if source.ndim != 4 or source.shape[-1] != 3 or source.dtype != np.uint8:
        raise ValueError(f"Unexpected source array {source_path}: {source.shape} {source.dtype}")
    expected_shape = (source.shape[0], source_h * 2, source_w * 2, 3)
    partial = output_path.with_name(output_path.name + ".partial.npy")
    progress = progress_path(output_path)
    start = 0
    if partial.exists() != progress.exists():
        raise ValueError(f"Orphaned partial build state: {partial} / {progress}")
    contract = progress_contract(
        source_path, model_sha256, modality, expected_shape, build_contract_sha256
    )
    if partial.exists() and progress.exists():
        state = json.loads(progress.read_text(encoding="utf-8"))
        recorded_contract = {key: state.get(key) for key in contract}
        if recorded_contract != contract:
            raise ValueError(f"Incompatible partial build state: {progress}")
        start = int(state["next_index"])
        target = np.lib.format.open_memmap(partial, mode="r+")
        if target.shape != expected_shape or target.dtype != np.uint8:
            raise ValueError(f"Partial array does not match expected output: {partial}")
    else:
        target = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint8, shape=expected_shape)
        atomic_json(progress, {**contract, "next_index": 0})

    if start:
        prefix_stats = array_content_stats(target, stop=start)
        assert_non_degenerate(prefix_stats, f"{modality} resumed train prefix")

    started = time.time()
    for index in range(start, source.shape[0], batch_size):
        stop = min(index + batch_size, source.shape[0])
        images = resize_source_images(source[index:stop], source_size, source_resampling)
        target[index:stop] = infer(model, images, modality, device)
        target.flush()
        atomic_json(progress, {**contract, "next_index": stop})
        if stop == source.shape[0] or stop % max(batch_size * 25, 1) == 0:
            elapsed = max(time.time() - started, 1e-6)
            print(f"{modality} train: {stop}/{source.shape[0]} ({(stop-start)/elapsed:.2f} images/s)", flush=True)
    del target
    os.replace(partial, output_path)
    progress.unlink(missing_ok=True)
    output = np.load(output_path, mmap_mode="r")
    content = array_content_stats(output)
    assert_non_degenerate(content, f"{modality} train output")
    return {
        "shape": list(expected_shape),
        "dtype": "uint8",
        "sha256": sha256_file(output_path),
        "content": content,
    }


def read_test_ids(source_root):
    path = source_root / "exp" / "test_id.txt"
    values = path.read_text(encoding="utf-8").replace("\n", ",").split(",")
    return path, [f"{int(value):04d}" for value in values if value.strip()]


def evaluation_paths(source_root, modality):
    _, test_ids = read_test_ids(source_root)
    paths = []
    for camera in MODALITY_CAMERAS[modality]:
        for identity in test_ids:
            directory = source_root / camera / identity
            if directory.is_dir():
                paths.extend(path for path in directory.iterdir() if path.is_file())
    return sorted(paths)


def build_eval_images(
    model, source_root, output_root, modality, source_size, batch_size, device,
    source_resampling="bicubic",
):
    paths = evaluation_paths(source_root, modality)
    pending = [path for path in paths if not (output_root / "eval" / path.relative_to(source_root)).is_file()]
    source_h, source_w = source_size
    interpolation = SOURCE_RESAMPLING[source_resampling]
    for index in range(0, len(pending), batch_size):
        batch_paths = pending[index:index + batch_size]
        images = []
        for path in batch_paths:
            with Image.open(path) as image:
                image = image.convert("RGB").resize((source_w, source_h), interpolation)
                images.append(np.asarray(image, dtype=np.uint8))
        outputs = infer(model, np.stack(images), modality, device)
        for path, output in zip(batch_paths, outputs):
            destination = output_root / "eval" / path.relative_to(source_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".tmp")
            Image.fromarray(output).save(temporary, format="PNG", compress_level=3)
            os.replace(temporary, destination)
        if index % max(batch_size * 25, 1) == 0:
            print(f"{modality} eval: {min(index + batch_size, len(pending))}/{len(pending)}", flush=True)
    output_paths = [output_root / "eval" / path.relative_to(source_root) for path in paths]
    stats = empty_content_stats()
    for start in range(0, len(output_paths), 64):
        batch = []
        for path in output_paths[start:start + 64]:
            with Image.open(path) as image:
                batch.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
        update_content_stats(stats, np.stack(batch))
    content = finalize_content_stats(stats)
    assert_non_degenerate(content, f"{modality} evaluation output")
    return {**hash_tree(output_paths, output_root), "content": content}


def smoke_test(model, source_root, modality, source_size, device, source_resampling="bicubic"):
    source = np.load(source_root / SOURCE_ARRAYS[modality], mmap_mode="r")
    images = resize_source_images(source[:1], source_size, source_resampling)
    output = infer(model, images, modality, device)
    stats = empty_content_stats()
    update_content_stats(stats, output)
    stats = finalize_content_stats(stats)
    assert_non_degenerate(stats, f"{modality} FP32 smoke output")
    if modality == "ir" and not (
        np.array_equal(output[..., 0], output[..., 1])
        and np.array_equal(output[..., 1], output[..., 2])
    ):
        raise ValueError("IR smoke output contains unequal channels")
    reference = normalize_ir(images) if modality == "ir" else images
    reconstructed = np.stack([
        np.asarray(Image.fromarray(image).resize((source_size[1], source_size[0]), BICUBIC), dtype=np.uint8)
        for image in output
    ])
    mse = float(np.mean((reference.astype(np.float32) - reconstructed.astype(np.float32)) ** 2))
    consistency_psnr = float("inf") if mse == 0 else float(10.0 * np.log10((255.0 ** 2) / mse))
    if consistency_psnr < 20.0:
        raise ValueError(f"{modality} FP32 smoke consistency is only {consistency_psnr:.2f} dB")
    stats["downsampled_source_psnr"] = consistency_psnr
    return stats


def build_identity(args, modalities, source_size, model_sha256, swinir_implementation):
    sources = {}
    for modality in modalities:
        source_path = (args.source_root / SOURCE_ARRAYS[modality]).resolve()
        label_path = (args.source_root / SOURCE_LABELS[modality]).resolve()
        sources[modality] = {
            "train_array": str(source_path),
            "train_array_sha256": sha256_file(source_path),
            "train_label": str(label_path),
            "train_label_sha256": sha256_file(label_path),
            "eval_tree": hash_tree(evaluation_paths(args.source_root, modality), args.source_root),
        }
    test_id_path, _ = read_test_ids(args.source_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "identity_type": "SYSU-MM01-SwinIR-x2-build",
        "numeric_policy": NUMERIC_POLICY,
        "source_root": str(args.source_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "source_size_hw": list(source_size),
        "output_size_hw": [source_size[0] * 2, source_size[1] * 2],
        "source_resampling": args.source_resampling,
        "modalities": list(modalities),
        "batch_size": int(args.batch_size),
        "device": str(args.device),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "repository_revision": git_revision(Path(__file__).resolve().parents[4]),
        "swinir_revision": git_revision(args.swinir_root),
        "swinir_implementation": swinir_implementation,
        "model_path": str(args.model_path.resolve()),
        "model_sha256": model_sha256,
        "test_id_sha256": sha256_file(test_id_path),
        "sources": sources,
    }


def establish_build_identity(output_root, identity):
    """Create or verify the immutable identity governing every resumable asset."""
    output_root = Path(output_root)
    contract_path = output_root / BUILD_CONTRACT_FILENAME
    if output_root.exists():
        entries = list(output_root.iterdir())
        if entries and not contract_path.is_file():
            raise RuntimeError(
                f"Refusing provenance-free existing assets in {output_root}; "
                "use a clean output root"
            )
    else:
        output_root.mkdir(parents=True)
    if contract_path.is_file():
        recorded = json.loads(contract_path.read_text(encoding="utf-8"))
        if recorded != identity:
            raise RuntimeError(f"Existing build identity does not match this invocation: {contract_path}")
    else:
        atomic_json(contract_path, identity)
    return contract_path, canonical_sha256(identity)


def main():
    args = parse_args()
    from salt_vi.utils.super_resolution.provenance import assert_clean_algorithm_source

    assert_clean_algorithm_source(REPO_ROOT)
    swinir_revision = assert_clean_swinir_source(args.swinir_root)
    modalities = tuple(dict.fromkeys(args.modalities))
    if set(modalities) != {"rgb", "ir"}:
        raise ValueError("The formal v2 dataset must build rgb and ir together")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    source_size = tuple(args.source_size)
    if min(source_size) < 1:
        raise ValueError("--source-size values must be positive")
    for path in (args.source_root, args.swinir_root, args.model_path):
        if not path.exists():
            raise FileNotFoundError(path)

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Immutable manifest already exists: {manifest_path}")

    disk = shutil.disk_usage(args.output_root.parent)
    if disk.free < args.min_free_before_gb * GIB:
        raise OSError(f"Need at least {args.min_free_before_gb:g} GiB free before generation; found {disk.free/GIB:.2f}")
    expected_bytes = 0
    for modality in modalities:
        source = np.load(args.source_root / SOURCE_ARRAYS[modality], mmap_mode="r")
        expected_bytes += int(source.size * source.dtype.itemsize * 4)
        expected_bytes += len(evaluation_paths(args.source_root, modality)) * source_size[0] * 2 * source_size[1] * 2 * 3
    if disk.free - expected_bytes < args.min_free_after_gb * GIB:
        raise OSError("Projected SR arrays would violate the required free-space reserve")

    model_sha256 = sha256_file(args.model_path)
    expected_network_file = args.swinir_root.resolve() / "models/network_swinir.py"
    if not expected_network_file.is_file():
        raise FileNotFoundError(expected_network_file)
    assert_official_swinir_identity(
        swinir_revision,
        model_sha256,
        {"network_sha256": sha256_file(expected_network_file)},
    )
    model, swinir_implementation = load_swinir(
        args.swinir_root, args.model_path, args.device
    )
    assert_official_swinir_identity(
        swinir_revision, model_sha256, swinir_implementation
    )
    smoke = {
        modality: smoke_test(
            model, args.source_root, modality, source_size, args.device,
            args.source_resampling,
        )
        for modality in modalities
    }
    if args.smoke_only:
        print(json.dumps({
            "valid": True,
            "numeric_policy": NUMERIC_POLICY,
            "swinir_implementation": swinir_implementation,
            "smoke_test": smoke,
        }, indent=2))
        return
    identity = build_identity(
        args, modalities, source_size, model_sha256, swinir_implementation
    )
    contract_path, build_contract_sha256 = establish_build_identity(args.output_root, identity)
    sources = {}
    outputs = {}
    for modality in modalities:
        source_path = args.source_root / SOURCE_ARRAYS[modality]
        label_path = args.source_root / SOURCE_LABELS[modality]
        output_path = args.output_root / OUTPUT_ARRAYS[modality]
        labels = np.load(label_path, mmap_mode="r")
        source_array = np.load(source_path, mmap_mode="r")
        if labels.shape[0] != source_array.shape[0]:
            raise ValueError(f"{modality} source image/label count mismatch")
        sources[modality] = {
            "train_array": str(source_path),
            "train_array_sha256": sha256_file(source_path),
            "train_label": str(label_path),
            "train_label_count": int(labels.shape[0]),
            "train_label_sha256": sha256_file(label_path),
            "eval_tree": hash_tree(evaluation_paths(args.source_root, modality), args.source_root),
        }
        if output_path.exists():
            array = np.load(output_path, mmap_mode="r")
            expected_shape = (source_array.shape[0], source_size[0] * 2, source_size[1] * 2, 3)
            if array.shape != expected_shape or array.dtype != np.uint8:
                raise ValueError(f"Existing output violates contract: {output_path} {array.shape} {array.dtype}")
            content = array_content_stats(array)
            assert_non_degenerate(content, f"{modality} existing train output")
            outputs[modality] = {
                "train_array": {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "sha256": sha256_file(output_path),
                    "content": content,
                }
            }
        else:
            outputs[modality] = {
                "train_array": build_array(
                    model, source_path, output_path, modality, source_size, args.batch_size, args.device,
                    model_sha256, build_contract_sha256, args.source_resampling,
                )
            }
        outputs[modality]["eval_tree"] = build_eval_images(
            model, args.source_root, args.output_root, modality, source_size, args.batch_size,
            args.device, args.source_resampling,
        )

    test_id_path, test_ids = read_test_ids(args.source_root)
    # Close the long-running build's time-of-check/time-of-use window before
    # publishing an immutable manifest.
    assert_clean_algorithm_source(REPO_ROOT)
    final_swinir_revision = assert_clean_swinir_source(args.swinir_root)
    final_implementation = dict(swinir_implementation)
    final_implementation["network_sha256"] = sha256_file(
        final_implementation["network_file"]
    )
    assert_official_swinir_identity(
        final_swinir_revision, sha256_file(args.model_path), final_implementation
    )
    final_identity = build_identity(
        args, modalities, source_size, sha256_file(args.model_path), final_implementation
    )
    if final_identity != identity:
        raise RuntimeError("Build inputs or implementation changed during SR generation")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": "SwinIR-M-classical-SR-DF2K-x2",
        "created_at_unix": time.time(),
        "command": shlex.join(sys.argv),
        "modalities": list(modalities),
        "source_root": str(args.source_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "source_size_hw": list(source_size),
        "output_size_hw": [source_size[0] * 2, source_size[1] * 2],
        "source_resampling": args.source_resampling,
        "ir_policy": "BT.601 luminance before SR; channel mean and replication after SR",
        "numeric_policy": NUMERIC_POLICY,
        "build_identity": identity,
        "build_identity_sha256": build_contract_sha256,
        "build_contract_path": str(contract_path),
        "build_contract_file_sha256": sha256_file(contract_path),
        "smoke_test": smoke,
        "eval_encoding": "lossless PNG bytes under original relative filenames",
        "swinir": {
            "root": str(args.swinir_root.resolve()),
            "revision": git_revision(args.swinir_root),
            "model_path": str(args.model_path.resolve()),
            "model_sha256": model_sha256,
            "implementation": swinir_implementation,
        },
        "dataset": {
            "test_id_path": str(test_id_path),
            "test_id_sha256": sha256_file(test_id_path),
            "test_identity_count": len(test_ids),
            "sources": sources,
        },
        "outputs": outputs,
    }
    atomic_json(manifest_path, manifest)
    os.chmod(contract_path, 0o444)
    os.chmod(manifest_path, 0o444)
    print(manifest_path)


if __name__ == "__main__":
    main()

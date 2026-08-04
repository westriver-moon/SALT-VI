#!/usr/bin/env python3
"""Validate integrity and semantic correctness of SYSU-MM01 SwinIR x2 assets."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .build_sysu_swinir_x2 import (
        NUMERIC_POLICY,
        BUILD_CONTRACT_FILENAME,
        OUTPUT_ARRAYS,
        SCHEMA_VERSION,
        SOURCE_ARRAYS,
        SOURCE_LABELS,
        array_content_stats,
        assert_non_degenerate,
        evaluation_paths,
        finalize_content_stats,
        hash_tree,
        normalize_ir,
        official_swinir_identity_errors,
        resize_source_images,
        sha256_file,
        empty_content_stats,
        update_content_stats,
        canonical_sha256,
    )
except ImportError:
    from build_sysu_swinir_x2 import (
        NUMERIC_POLICY,
        BUILD_CONTRACT_FILENAME,
        OUTPUT_ARRAYS,
        SCHEMA_VERSION,
        SOURCE_ARRAYS,
        SOURCE_LABELS,
        array_content_stats,
        assert_non_degenerate,
        evaluation_paths,
        finalize_content_stats,
        hash_tree,
        normalize_ir,
        official_swinir_identity_errors,
        resize_source_images,
        sha256_file,
        empty_content_stats,
        update_content_stats,
        canonical_sha256,
    )


SOURCE_SIZE = (288, 144)
OUTPUT_SIZE = (576, 288)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--quick", action="store_true", help="Skip SHA-256 checks; semantic scans still run")
    parser.add_argument("--consistency-samples", type=int, default=64)
    parser.add_argument("--min-consistency-psnr", type=float, default=20.0)
    parser.add_argument("--min-sample-psnr", type=float, default=15.0)
    parser.add_argument("--min-p05-psnr", type=float, default=20.0)
    return parser.parse_args()


def content_errors(stats, label):
    try:
        assert_non_degenerate(stats, label)
    except ValueError as error:
        return [str(error)]
    return []


def eval_content_stats(paths):
    stats = empty_content_stats()
    for start in range(0, len(paths), 64):
        images = []
        for path in paths[start:start + 64]:
            with Image.open(path) as image:
                image = image.convert("RGB")
                if image.size != (OUTPUT_SIZE[1], OUTPUT_SIZE[0]):
                    raise ValueError(f"Evaluation image has size {image.size}, expected {(OUTPUT_SIZE[1], OUTPUT_SIZE[0])}: {path}")
                images.append(np.asarray(image, dtype=np.uint8))
        update_content_stats(stats, np.stack(images))
    return finalize_content_stats(stats)


def sampled_indices(count, requested):
    sample_count = min(int(requested), int(count))
    if sample_count < 1:
        return []
    anchors = {0, count // 2, count - 1}
    anchors.update(int(value) for value in np.linspace(0, count - 1, max(1, sample_count // 2)))
    rng = np.random.RandomState(0)
    remaining = max(0, sample_count - len(anchors))
    if remaining:
        anchors.update(int(value) for value in rng.choice(count, size=remaining, replace=False))
    return sorted(anchors)[:sample_count]


def psnr(reference, prediction):
    error = np.mean((reference.astype(np.float32) - prediction.astype(np.float32)) ** 2)
    return float("inf") if error == 0 else float(10.0 * math.log10((255.0 ** 2) / error))


def psnr_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "sample_count": int(values.size),
        "minimum_psnr": float(np.min(values)),
        "p05_psnr": float(np.percentile(values, 5)),
        "p10_psnr": float(np.percentile(values, 10)),
        "mean_psnr": float(np.mean(values)),
    }


def downsampled_consistency(source_images, output_images, modality):
    source_images = resize_source_images(source_images, SOURCE_SIZE)
    if modality == "ir":
        source_images = normalize_ir(source_images)
    values = []
    for source, output in zip(source_images, output_images):
        reconstructed = np.asarray(
            Image.fromarray(np.asarray(output)).resize((SOURCE_SIZE[1], SOURCE_SIZE[0]), Image.BICUBIC),
            dtype=np.uint8,
        )
        values.append(psnr(source, reconstructed))
    return values


def train_consistency(source, output, modality, requested):
    indices = sampled_indices(source.shape[0], requested)
    source_images = np.stack([source[index] for index in indices])
    output_images = np.stack([output[index] for index in indices])
    values = downsampled_consistency(source_images, output_images, modality)
    return psnr_summary(values)


def eval_consistency(source_paths, output_paths, source_root, modality, requested):
    camera_groups = {}
    for index, path in enumerate(source_paths):
        camera = path.relative_to(source_root).parts[0]
        camera_groups.setdefault(camera, []).append(index)
    per_camera_requested = max(1, int(math.ceil(requested / max(len(camera_groups), 1))))
    indices = []
    sampled_cameras = []
    for camera, group in sorted(camera_groups.items()):
        selected = [group[index] for index in sampled_indices(len(group), per_camera_requested)]
        indices.extend(selected)
        sampled_cameras.extend([camera] * len(selected))
    source_images = []
    output_images = []
    for index in indices:
        with Image.open(source_paths[index]) as image:
            image = image.convert("RGB").resize((SOURCE_SIZE[1], SOURCE_SIZE[0]), Image.BICUBIC)
            source_images.append(np.asarray(image, dtype=np.uint8))
        with Image.open(output_paths[index]) as image:
            output_images.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    values = downsampled_consistency(np.stack(source_images), np.stack(output_images), modality)
    result = psnr_summary(values)
    result["per_camera"] = {
        camera: psnr_summary([value for value, item_camera in zip(values, sampled_cameras) if item_camera == camera])
        for camera in sorted(set(sampled_cameras))
    }
    return result


def consistency_errors(values, args, label):
    errors = []
    scopes = [(label, values)]
    scopes.extend(
        (f"{label} camera {camera}", camera_values)
        for camera, camera_values in values.get("per_camera", {}).items()
    )
    for scope, summary in scopes:
        if summary["minimum_psnr"] < args.min_sample_psnr:
            errors.append(
                f"{scope} minimum PSNR {summary['minimum_psnr']:.2f} < {args.min_sample_psnr:.2f} dB"
            )
        if summary["p05_psnr"] < args.min_p05_psnr:
            errors.append(
                f"{scope} P05 PSNR {summary['p05_psnr']:.2f} < {args.min_p05_psnr:.2f} dB"
            )
        if summary["mean_psnr"] < args.min_consistency_psnr:
            errors.append(
                f"{scope} mean PSNR {summary['mean_psnr']:.2f} < "
                f"{args.min_consistency_psnr:.2f} dB"
            )
    return errors


def validate(args):
    errors = []
    checks = {}
    manifest_path = args.output_root / "manifest.json"
    if not manifest_path.is_file():
        return {"valid": False, "errors": [f"Missing manifest: {manifest_path}"], "checks": checks, "quick": args.quick}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    contract_path = args.output_root / BUILD_CONTRACT_FILENAME
    if not contract_path.is_file():
        errors.append(f"missing immutable build identity: {contract_path}")
    else:
        build_identity = json.loads(contract_path.read_text(encoding="utf-8"))
        if build_identity != manifest.get("build_identity"):
            errors.append("build identity does not match manifest")
        if canonical_sha256(build_identity) != manifest.get("build_identity_sha256"):
            errors.append("build identity canonical hash mismatch")
        if sha256_file(contract_path) != manifest.get("build_contract_file_sha256"):
            errors.append("build identity file hash mismatch")
        errors.extend(official_swinir_identity_errors(
            build_identity.get("swinir_revision"),
            build_identity.get("model_sha256"),
            build_identity.get("swinir_implementation"),
        ))
        current_builder = Path(__file__).with_name("build_sysu_swinir_x2.py")
        if build_identity.get("builder_sha256") != sha256_file(current_builder):
            errors.append("dataset was not generated by the current pinned builder")

        manifest_swinir = manifest.get("swinir", {})
        if manifest_swinir.get("revision") != build_identity.get("swinir_revision"):
            errors.append("manifest SwinIR revision differs from build identity")
        if manifest_swinir.get("model_sha256") != build_identity.get("model_sha256"):
            errors.append("manifest SwinIR model hash differs from build identity")
        if manifest_swinir.get("implementation") != build_identity.get("swinir_implementation"):
            errors.append("manifest SwinIR implementation differs from build identity")

    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"manifest schema_version must be {SCHEMA_VERSION}")
    if manifest.get("numeric_policy") != NUMERIC_POLICY:
        errors.append(f"manifest numeric_policy must be {NUMERIC_POLICY}")
    if manifest.get("source_size_hw") != list(SOURCE_SIZE) or manifest.get("output_size_hw") != list(OUTPUT_SIZE):
        errors.append("manifest size contract is not 288x144 -> 576x288")
    if set(manifest.get("modalities", [])) != {"rgb", "ir"}:
        errors.append("formal derived dataset must contain both rgb and ir modalities")
    test_id_path = args.source_root / "exp" / "test_id.txt"
    if not test_id_path.is_file():
        errors.append(f"missing SYSU test identity file: {test_id_path}")
    elif not args.quick and sha256_file(test_id_path) != manifest.get("dataset", {}).get("test_id_sha256"):
        errors.append("SYSU test identity file changed after generation")

    for modality in manifest.get("modalities", []):
        if modality not in SOURCE_ARRAYS:
            errors.append(f"unsupported manifest modality: {modality}")
            continue
        source_path = args.source_root / SOURCE_ARRAYS[modality]
        label_path = args.source_root / SOURCE_LABELS[modality]
        output_path = args.output_root / OUTPUT_ARRAYS[modality]
        if not source_path.is_file() or not label_path.is_file() or not output_path.is_file():
            errors.append(f"missing {modality} source, label, or output training array")
            continue
        source = np.load(source_path, mmap_mode="r")
        labels = np.load(label_path, mmap_mode="r")
        output = np.load(output_path, mmap_mode="r")
        if labels.shape[0] != source.shape[0]:
            errors.append(f"{modality} source image/label count mismatch")
        expected_shape = (source.shape[0], OUTPUT_SIZE[0], OUTPUT_SIZE[1], 3)
        if output.shape != expected_shape:
            errors.append(f"{modality} array shape {output.shape} != {expected_shape}")
            continue
        if output.dtype != np.uint8:
            errors.append(f"{modality} array dtype is {output.dtype}, expected uint8")
            continue

        train_stats = array_content_stats(output)
        errors.extend(content_errors(train_stats, f"{modality} train output"))
        if modality == "ir":
            for start in range(0, output.shape[0], 64):
                chunk = output[start:start + 64]
                if not (np.array_equal(chunk[..., 0], chunk[..., 1]) and np.array_equal(chunk[..., 1], chunk[..., 2])):
                    errors.append("IR train output contains unequal color channels")
                    break

        expected_eval = evaluation_paths(args.source_root, modality)
        output_eval = [args.output_root / "eval" / path.relative_to(args.source_root) for path in expected_eval]
        missing = [str(path.relative_to(args.output_root)) for path in output_eval if not path.is_file()]
        eval_stats = None
        if missing:
            errors.append(f"{modality} evaluation cache is missing {len(missing)} images")
        else:
            try:
                eval_stats = eval_content_stats(output_eval)
                errors.extend(content_errors(eval_stats, f"{modality} evaluation output"))
            except (OSError, ValueError) as error:
                errors.append(str(error))
            if modality == "ir":
                for path in output_eval:
                    with Image.open(path) as image:
                        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
                    if not (np.array_equal(pixels[..., 0], pixels[..., 1]) and np.array_equal(pixels[..., 1], pixels[..., 2])):
                        errors.append(f"IR evaluation output contains unequal channels: {path}")
                        break

        consistency = {}
        try:
            consistency["train"] = train_consistency(source, output, modality, args.consistency_samples)
            if not missing:
                consistency["eval"] = eval_consistency(
                    expected_eval, output_eval, args.source_root, modality,
                    max(1, args.consistency_samples // 2)
                )
            for split, values in consistency.items():
                errors.extend(consistency_errors(values, args, f"{modality} {split}"))
        except (OSError, ValueError) as error:
            errors.append(f"{modality} consistency check failed: {error}")

        recorded_source = manifest.get("dataset", {}).get("sources", {}).get(modality, {})
        recorded_output = manifest.get("outputs", {}).get(modality, {})
        if not args.quick:
            if sha256_file(source_path) != recorded_source.get("train_array_sha256"):
                errors.append(f"{modality} source array changed after generation")
            if sha256_file(label_path) != recorded_source.get("train_label_sha256"):
                errors.append(f"{modality} source labels changed after generation")
            if hash_tree(expected_eval, args.source_root) != recorded_source.get("eval_tree"):
                errors.append(f"{modality} source evaluation tree changed after generation")
            if sha256_file(output_path) != recorded_output.get("train_array", {}).get("sha256"):
                errors.append(f"{modality} output array hash mismatch")
            if not missing and hash_tree(output_eval, args.output_root) != {
                key: recorded_output.get("eval_tree", {}).get(key) for key in ("sha256", "file_count")
            }:
                errors.append(f"{modality} evaluation tree hash mismatch")

        checks[modality] = {
            "source_count": int(source.shape[0]),
            "output_count": int(output.shape[0]),
            "eval_count": len(expected_eval),
            "missing_eval_count": len(missing),
            "train_content": train_stats,
            "eval_content": eval_stats,
            "consistency": consistency,
        }

    return {"valid": not errors, "errors": errors, "checks": checks, "quick": args.quick}


def main():
    args = parse_args()
    result = validate(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

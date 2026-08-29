#!/usr/bin/env python3
"""Original-resolution SwinIR x4 with the legacy IR channel policy.

No source/destination resize. Window padding and tiled model inference only.
Python 3.8 / torch 1.8 compatible. CPU inventory does not import torch.
"""
import argparse
import collections
import filecmp
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(str(temp), str(path))


def load_config(path):
    config = json.loads(Path(path).read_text())
    if (config["scale"] != 4 or config["dtype"] != "float32"
            or config["ir_policy"] != "legacy_weighted_input_mean_quantized_output"
            or config["pre_resize"] or config["post_resize"]
            or config["window"] != 8
            or config["padding"] != "symmetric_right_bottom_to_window_multiple"):
        raise ValueError("This pipeline only supports native-size x4 FP32 with legacy IR channels")
    tile, overlap = config["tile"], config["overlap"]
    if tile != 0 and (tile < 8 or tile % 8 or overlap < 0 or overlap % 8 or overlap >= tile):
        raise ValueError("tile=0 means whole image; otherwise tile/overlap must be window multiples with 0 <= overlap < tile")
    for key in ("output_dir", "manifest_dir"):
        config[key] = str((Path(path).resolve().parent / config[key]).resolve())
    return config


def read_ids(path):
    return {int(x) for x in re.split(r"[\s,]+", Path(path).read_text().strip()) if x}


def index_rows(path):
    for line in Path(path).read_text().splitlines():
        if line.strip():
            relative, label = line.strip().rsplit(None, 1)
            relative = relative.replace("\\", "/")
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("Index path must stay inside the dataset: " + relative)
            yield relative, int(label)


def image_files(directory):
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)


def tile_starts(length, tile, overlap):
    if length <= tile:
        return [0]
    return list(range(0, length - tile, tile - overlap)) + [length - tile]


def work_pixels(width, height, config):
    """Total LR pixels evaluated, including window padding and tile overlaps."""
    w, h = math.ceil(width / 8) * 8, math.ceil(height / 8) * 8
    tile, overlap = config["tile"], config["overlap"]
    if tile == 0:
        return w * h
    return (min(w, tile) * min(h, tile)
            * len(tile_starts(w, tile, overlap)) * len(tile_starts(h, tile, overlap)))


def build_inventory(config):
    all_rows, summaries = [], {}
    for dataset, base in config["roots"].items():
        root = Path(base)
        entries = {}

        def add(relative, modality, reference, label=None):
            item = entries.setdefault(relative, {
                "dataset": dataset, "source": relative, "modality": modality,
                "references": [], "aliases": []})
            if item["modality"] != modality:
                raise ValueError("Conflicting modalities: " + relative)
            record = {"index": reference}
            if label is not None:
                record["label"] = label
            if record not in item["references"]:
                item["references"].append(record)
            return item

        if dataset == "sysu":
            for split in ("train", "val", "test"):
                for pid in sorted(read_ids(root / "exp" / (split + "_id.txt"))):
                    for camera in range(1, 7):
                        directory = root / ("cam%d" % camera) / ("%04d" % pid)
                        if directory.is_dir():
                            for path in image_files(directory):
                                add(path.relative_to(root).as_posix(),
                                    "ir" if camera in (3, 6) else "rgb", split, pid)
        elif dataset == "regdb":
            for trial in range(1, 11):
                for split in ("train", "test"):
                    for modal, modality in (("visible", "rgb"), ("thermal", "ir")):
                        index = "idx/%s_%s_%d.txt" % (split, modal, trial)
                        for relative, label in index_rows(root / index):
                            add(relative, modality, index, label)
        elif dataset == "llcm":
            for split in ("train", "test"):
                for modal, modality in (("vis", "rgb"), ("nir", "ir")):
                    index = "idx/%s_%s.txt" % (split, modal)
                    for relative, label in index_rows(root / index):
                        add(relative, modality, index, label)
            # SALT-VI test loaders use camera-organized copies, not the idx paths.
            # Verify bytes before treating a camera path as an alias; no hashes.
            test_ids = read_ids(root / "idx/test_id.txt")
            for modal, modality in (("vis", "rgb"), ("nir", "ir")):
                cameras = range(1, 10) if modal == "vis" else (1, 2, 4, 5, 6, 7, 8, 9)
                for camera in cameras:
                    for pid in sorted(test_ids):
                        directory = root / ("test_" + modal) / ("cam%d" % camera) / ("%04d" % pid)
                        if not directory.is_dir():
                            continue
                        for path in image_files(directory):
                            relative = path.relative_to(root).as_posix()
                            canonical = (Path(modal) / path.parent.name / path.name).as_posix()
                            if canonical in entries and filecmp.cmp(str(path), str(root / canonical), shallow=False):
                                entries[canonical]["aliases"].append(relative)
                            else:
                                add(relative, modality, "test_camera_candidates", pid)
        else:
            raise ValueError("Unknown dataset: " + dataset)

        claimed_outputs = set()
        rows = []
        for relative, item in sorted(entries.items()):
            source = root / relative
            with Image.open(source) as im:
                width, height = im.size
                item["source_mode"] = im.mode
            item.update(width=width, height=height,
                        source_bytes=source.stat().st_size,
                        source_mtime_ns=source.stat().st_mtime_ns,
                        output=Path(relative).with_suffix(".png").as_posix(),
                        work_pixels=work_pixels(width, height, config))
            for source_name in [relative] + item["aliases"]:
                output_name = Path(source_name).with_suffix(".png").as_posix()
                if output_name in claimed_outputs:
                    raise ValueError("Output name collision: " + output_name)
                claimed_outputs.add(output_name)
            rows.append(item)
        summaries[dataset] = {
            "unique_images": len(rows),
            "modalities": dict(collections.Counter(r["modality"] for r in rows)),
            "alias_paths": sum(len(r["aliases"]) for r in rows),
            "source_bytes": sum(r["source_bytes"] for r in rows),
            "output_raw_rgb_bytes": sum(r["width"] * r["height"] * 16 * 3 for r in rows),
            "total_work_pixels": sum(r["work_pixels"] for r in rows),
            "width_range": [min(r["width"] for r in rows), max(r["width"] for r in rows)],
            "height_range": [min(r["height"] for r in rows), max(r["height"] for r in rows)],
            "largest_image": max(rows, key=lambda r: r["width"] * r["height"])["source"],
            "source_modes": dict(collections.Counter(r["source_mode"] for r in rows)),
        }
        all_rows.extend(rows)
        print(json.dumps({dataset: summaries[dataset]}, ensure_ascii=False), flush=True)
    dest = Path(config["manifest_dir"])
    dest.mkdir(parents=True, exist_ok=True)
    temp = dest / "images.jsonl.tmp"
    with temp.open("w") as out:
        for row in all_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(str(temp), str(dest / "images.jsonl"))
    write_json(dest / "summary.json", summaries)
    write_json(dest / "inventory_config.json", config)
    return all_rows


def load_rows(config, datasets=None):
    directory = Path(config["manifest_dir"])
    inventory_config = json.loads((directory / "inventory_config.json").read_text())
    if config != inventory_config:
        raise ValueError("Config changed since inventory: use a new manifest/output directory and rebuild inventory")
    with (directory / "images.jsonl").open() as stream:
        return [r for r in map(json.loads, stream) if not datasets or r["dataset"] in datasets]


def normalize_ir(images):
    # Exact old builder's input formula, including float64 arithmetic / np.rint.
    luminance = np.rint(images[..., 0] * 0.299 + images[..., 1] * 0.587
                        + images[..., 2] * 0.114).astype(np.uint8)
    return np.repeat(luminance[..., None], 3, axis=-1)


def prepare_images(images, modality):
    import torch
    images = np.asarray(images)
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("Expected uint8 NHWC RGB images")
    if modality == "ir":
        images = normalize_ir(images)
    elif modality != "rgb":
        raise ValueError(modality)
    return torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float().div_(255.)


def finalize_images(tensor, modality, expected_hw):
    import torch
    if tensor.ndim != 4 or tensor.shape[1] != 3 or tuple(tensor.shape[-2:]) != tuple(expected_hw):
        raise ValueError("Wrong SwinIR output shape: " + str(tensor.shape))
    if not bool(torch.isfinite(tensor).all().item()):
        raise FloatingPointError("SwinIR produced non-finite output")
    images = (tensor.detach().float().clamp_(0., 1.).mul_(255.).round_()
              .byte().permute(0, 2, 3, 1).cpu().numpy())
    if modality == "ir":
        # Preserve old ordering: first RGB uint8 quantization, then channel mean.
        gray = np.rint(images.astype(np.float32).mean(axis=-1)).astype(np.uint8)
        images = np.repeat(gray[..., None], 3, axis=-1)
    elif modality != "rgb":
        raise ValueError(modality)
    return images


def padded_input(images):
    h, w = images.shape[1:3]
    return np.pad(images, ((0, 0), (0, (-h) % 8), (0, (-w) % 8), (0, 0)), mode="symmetric")


def infer(model, images, modality, device, config):
    import torch
    h, w = images.shape[1:3]
    # Channel conversion before padding (padding only repeats existing pixels).
    tensor = prepare_images(padded_input(images), modality)
    _, _, ph, pw = tensor.shape
    tile, overlap, scale = config["tile"], config["overlap"], 4
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        if tile == 0 or (ph <= tile and pw <= tile):
            output = model(tensor.to(device)).cpu()
        else:
            # CPU assembly bounds GPU memory by the fixed tile size.
            output = torch.zeros((1, 3, ph * scale, pw * scale), dtype=torch.float32)
            counts = torch.zeros((1, 1, ph * scale, pw * scale), dtype=torch.float32)
            for top in tile_starts(ph, tile, overlap):
                for left in tile_starts(pw, tile, overlap):
                    bottom, right = min(top + tile, ph), min(left + tile, pw)
                    patch = model(tensor[:, :, top:bottom, left:right].to(device)).cpu()
                    output[:, :, top*scale:bottom*scale, left*scale:right*scale] += patch
                    counts[:, :, top*scale:bottom*scale, left*scale:right*scale] += 1
            if not bool((counts > 0).all().item()):
                raise RuntimeError("Uncovered tile pixels")
            output.div_(counts)
    return finalize_images(output[:, :, :h*4, :w*4], modality, (h*4, w*4))[0]


def assert_idle_gpu(gpu):
    # Physical index. Never kill, preempt, or use an occupied GPU.
    for observation in range(2):
        raw = subprocess.check_output([
            "nvidia-smi", "-i", str(gpu),
            "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        uuid, memory, utilization = [s.strip() for s in raw.strip().split(",")]
        processes = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
        if uuid in processes or int(memory) > 512 or int(utilization) > 10:
            raise RuntimeError("Physical GPU %s is occupied; nothing was launched: %s" % (gpu, raw.strip()))
        if observation == 0:
            time.sleep(1)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    return uuid


def load_model(config, device="cuda:0"):
    import torch
    sys.path.insert(0, str(HERE / "vendor"))
    from network_swinir import SwinIR
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                   img_range=1., depths=[6]*6, embed_dim=180, num_heads=[6]*6,
                   mlp_ratio=2, upsampler="nearest+conv", resi_connection="1conv")
    checkpoint = torch.load(config["checkpoint"], map_location="cpu")
    key = next((k for k in ("params_ema", "params", "state_dict") if k in checkpoint), None)
    model.load_state_dict(checkpoint[key] if key else checkpoint, strict=True)
    model.eval().float().to(device)
    metadata = {"checkpoint_key": key or "bare_state_dict",
                "parameters": sum(p.numel() for p in model.parameters()),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0) if str(device).startswith("cuda") else "cpu"}
    print(json.dumps(metadata), flush=True)
    return model, metadata


def source_path(row, config):
    path = Path(config["roots"][row["dataset"]]) / row["source"]
    stat = path.stat()
    if stat.st_size != row["source_bytes"] or stat.st_mtime_ns != row["source_mtime_ns"]:
        raise RuntimeError("Source changed since inventory: " + str(path))
    return path


def output_valid(path, row):
    if not path.exists():
        return False
    try:
        with Image.open(path) as im:
            im.load()  # Decode entire PNG, not only its header.
            if im.format != "PNG" or im.mode != "RGB" or im.size != (row["width"]*4, row["height"]*4):
                return False
            if row["modality"] == "ir":
                a = np.asarray(im)
                return bool(np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 0], a[..., 2]))
            return True
    except (OSError, ValueError):
        return False


def save_output(path, pixels, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    Image.fromarray(pixels).save(temp, format="PNG", compress_level=config["png_compress_level"])
    os.replace(str(temp), str(path))


def ensure_aliases(root, row):
    canonical = root / row["dataset"] / row["output"]
    for alias in row["aliases"]:
        path = root / row["dataset"] / Path(alias).with_suffix(".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink() and path.resolve() == canonical.resolve():
            continue
        if os.path.lexists(str(path)):
            raise FileExistsError("Refusing to overwrite alias path: " + str(path))
        path.symlink_to(os.path.relpath(str(canonical), str(path.parent)))


def output_contract(root, config, metadata):
    stat = Path(config["checkpoint"]).stat()
    contract = {"config": config, "model": metadata,
                "checkpoint_bytes": stat.st_size, "checkpoint_mtime_ns": stat.st_mtime_ns}
    path = root / "contract.json"
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise RuntimeError("Output contract differs; choose a separate output directory")
    else:
        if root.exists() and any(root.iterdir()):
            raise RuntimeError("Nonempty output directory has no contract: " + str(root))
        write_json(path, contract)


def process_row(model, row, root, config):
    start = time.perf_counter()
    with Image.open(source_path(row, config)) as im:
        if im.size != (row["width"], row["height"]):
            raise ValueError("Source dimensions changed")
        pixels = np.asarray(im.convert("RGB"), dtype=np.uint8)
    result = infer(model, pixels[None], row["modality"], "cuda:0", config)
    path = root / row["dataset"] / row["output"]
    save_output(path, result, config)
    ensure_aliases(root, row)
    elapsed = time.perf_counter() - start
    if not output_valid(path, row):
        raise RuntimeError("Written image failed validation: " + str(path))
    return {"dataset": row["dataset"], "modality": row["modality"],
            "source": row["source"], "width": row["width"], "height": row["height"],
            "work_pixels": row["work_pixels"], "seconds": elapsed,
            "png_bytes": path.stat().st_size}


def benchmark(model, rows, config, metadata, per_stratum, gpu_uuid):
    import torch
    if per_stratum < 1:
        raise ValueError("per-stratum must be positive")
    root = HERE / "benchmarks" / time.strftime("%Y%m%d-%H%M%S")
    output_contract(root, config, metadata)
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["modality"])].append(row)
    # Warm up kernels; not included in estimates.
    warm = min(rows, key=lambda r: r["work_pixels"])
    with Image.open(source_path(warm, config)) as im:
        pixels = np.asarray(im.convert("RGB"))[None]
    for _ in range(2):
        infer(model, pixels, warm["modality"], "cuda:0", config)
    torch.cuda.reset_peak_memory_stats()
    rng, strata, samples = random.Random(20260826), [], []
    for (dataset, modality), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda r: (r["work_pixels"], r["source"]))
        for quartile in range(4):
            population = ordered[len(ordered)*quartile//4:len(ordered)*(quartile+1)//4]
            if not population:
                continue
            chosen = rng.sample(population, min(per_stratum, len(population)))
            results = [process_row(model, row, root, config) for row in chosen]
            samples.extend(results)
            mean_seconds = sum(r["seconds"] for r in results) / len(results)
            mean_bytes = sum(r["png_bytes"] for r in results) / len(results)
            stratum = {"dataset": dataset, "modality": modality, "quartile": quartile,
                       "population": len(population), "sample_count": len(results),
                       "mean_seconds": mean_seconds,
                       "estimated_seconds": len(population)*mean_seconds,
                       "estimated_png_bytes": len(population)*mean_bytes}
            strata.append(stratum)
            print(json.dumps(stratum), flush=True)
    # Explicit largest-size tests are separate, not over-weighted in the estimate.
    extremes = [process_row(model, max(group, key=lambda r: r["width"]*r["height"]), root, config)
                for group in groups.values()]
    estimates = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        selected = [s for s in strata if s["dataset"] == dataset]
        seconds = sum(s["estimated_seconds"] for s in selected)
        estimates[dataset] = {
            "images": sum(s["population"] for s in selected),
            "hours_point_estimate": seconds/3600,
            "hours_planning_range": [seconds/3600*0.8, seconds/3600*1.5],
            "png_gib_estimate": sum(s["estimated_png_bytes"] for s in selected)/1024**3}
    report = {"gpu_uuid": gpu_uuid, "model": metadata, "config": config,
              "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
              "peak_reserved_mib": torch.cuda.max_memory_reserved()/1024**2,
              "estimates": estimates, "strata": strata, "samples": samples, "extremes": extremes,
              "range_note": "Heuristic 0.8-1.5x range, not a statistical confidence interval; assumes exclusive GPU availability. Includes input decode, channels, inference, PNG write and aliases; excludes GPU queue time. Validation overhead is not included in sample seconds."}
    write_json(root / "report.json", report)
    print(json.dumps({"report": str(root/"report.json"), "estimates": estimates,
                      "peak_allocated_mib": report["peak_allocated_mib"]}), flush=True)


def run(model, rows, config, metadata, limit):
    root = Path(config["output_dir"])
    output_contract(root, config, metadata)
    completed, skipped = 0, 0
    start = time.monotonic()
    with (root/"progress.jsonl").open("a", buffering=1) as log:
        for row in rows:
            source_path(row, config)
            path = root / row["dataset"] / row["output"]
            if output_valid(path, row):
                ensure_aliases(root, row)
                skipped += 1
                continue
            result = process_row(model, row, root, config)
            log.write(json.dumps(result) + "\n")
            completed += 1
            if completed % 50 == 0:
                print(json.dumps({"generated": completed, "skipped": skipped,
                                  "remaining": len(rows)-completed-skipped,
                                  "elapsed_seconds": time.monotonic()-start}), flush=True)
            if limit and completed >= limit:
                break
    print(json.dumps({"generated": completed, "skipped": skipped, "selected": len(rows)}), flush=True)


def verify(rows, config, output_root=None):
    root = Path(output_root or config["output_dir"])
    stats = collections.Counter()
    for row in rows:
        path = root / row["dataset"] / row["output"]
        if not path.exists():
            stats["missing"] += 1
        elif not output_valid(path, row):
            stats["invalid"] += 1
        else:
            stats["valid"] += 1
        for alias in row["aliases"]:
            target = root / row["dataset"] / Path(alias).with_suffix(".png")
            if not target.is_symlink() or target.resolve() != path.resolve() or not path.exists():
                stats["missing_or_invalid_alias"] += 1
    print(json.dumps(stats), flush=True)
    return stats


def geometry_enabled(args):
    if args.geometry == "off":
        return False
    settings = json.loads(Path(args.geometry_config).read_text())
    enabled = True if args.geometry == "on" else settings["enabled"]
    if type(enabled) is not bool:
        raise ValueError("Geometry enabled must be a boolean")
    if enabled:
        from geometry_stage import load_settings
        load_settings(args.geometry_config)  # Validate before starting expensive SR.
    return enabled


def launch_geometry(args, config):
    """Run the CPU stage in its existing YOLO environment, not the SR environment."""
    settings = json.loads(Path(args.geometry_config).read_text())
    # Do not resolve the venv interpreter symlink: that would select its base
    # environment and lose the venv's installed YOLO dependencies.
    interpreter = os.path.abspath(str(Path(args.geometry_config).resolve().parent/settings["python"]))
    command = [interpreter, "-u", str(HERE/"geometry_stage.py"),
               "--config", str(Path(args.config).resolve()),
               "--geometry-config", str(Path(args.geometry_config).resolve()), "--geometry", "on"]
    for name, value in (("--source-root", args.source_root), ("--output-root", args.output_root)):
        if value:
            command += [name, value]
    if args.datasets:
        command += ["--datasets"] + args.datasets
    if args.limit and args.command != "run":
        command += ["--limit", str(args.limit)]
    if args.available_only or (args.command == "run" and args.limit):
        command.append("--available-only")
    if args.command == "verify-geometry":
        command.append("--verify")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="",
                       OMP_NUM_THREADS=str(settings["cpu_threads"]),
                       MKL_NUM_THREADS=str(settings["cpu_threads"]),
                       OPENBLAS_NUM_THREADS=str(settings["cpu_threads"]))
    libraries = settings.get("library_dirs", [])
    if libraries:
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(libraries +
            ([environment["LD_LIBRARY_PATH"]] if environment.get("LD_LIBRARY_PATH") else []))
    subprocess.run(command, check=True, env=environment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "benchmark", "run", "verify", "postprocess", "verify-geometry"))
    parser.add_argument("--config", default=str(HERE/"config.json"))
    parser.add_argument("--datasets", nargs="+", choices=("sysu", "regdb", "llcm"))
    parser.add_argument("--gpu", type=int, help="Physical nvidia-smi index; must be idle")
    parser.add_argument("--per-stratum", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many newly generated images; 0 = all")
    parser.add_argument("--output-root", help="verify or geometry: alternate output directory")
    parser.add_argument("--source-root", help="geometry: alternate native SR directory")
    parser.add_argument("--geometry-config", default=str(HERE/"geometry_config.json"))
    parser.add_argument("--geometry", choices=("on", "off"), help="Optional post-SR stage; default from geometry_config.json")
    parser.add_argument("--available-only", action="store_true", help="geometry: explicitly select only existing native SR images")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "inventory":
        build_inventory(config)
        return
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    if args.command in ("postprocess", "verify-geometry"):
        if args.command == "postprocess" and not geometry_enabled(args):
            print(json.dumps({"geometry": "off", "generated": 0}))
            return
        launch_geometry(args, config)
        return
    if args.command == "run" and geometry_enabled(args):
        # Parent has not imported torch. The SR child exits completely before
        # CPU postprocessing starts, releasing its CUDA context and all VRAM.
        subprocess.run([sys.executable, str(HERE/"pipeline.py")] + sys.argv[1:]
                       + ["--geometry", "off"], check=True)
        launch_geometry(args, config)
        return
    rows = load_rows(config, args.datasets)
    if args.command == "verify":
        stats = verify(rows, config, args.output_root)
        if stats["missing"] or stats["invalid"] or stats["missing_or_invalid_alias"]:
            sys.exit(1)
        return
    if args.gpu is None or args.limit < 0:
        parser.error("benchmark/run require --gpu; --limit must be nonnegative")
    gpu_uuid = assert_idle_gpu(args.gpu)
    model, metadata = load_model(config)
    if args.command == "benchmark":
        benchmark(model, rows, config, metadata, args.per_stratum, gpu_uuid)
    else:
        run(model, rows, config, metadata, args.limit)


if __name__ == "__main__":
    main()

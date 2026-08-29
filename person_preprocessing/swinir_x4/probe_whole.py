"""Read-only original-image diagnostic: whole-image inference memory, no PNG writes."""
import argparse
import gc
import json
import math
import time

import numpy as np
from PIL import Image

import pipeline as p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    config = p.load_config(p.HERE/"config.json")
    rows = p.load_rows(config)
    uuid = p.assert_idle_gpu(args.gpu)
    import torch
    model, metadata = p.load_model(config)
    warm = min(rows, key=lambda r: r["work_pixels"])
    with Image.open(p.source_path(warm, config)) as im:
        pixels = np.asarray(im.convert("RGB"))[None]
    p.infer(model, pixels, warm["modality"], "cuda:0", config)
    results = []
    for dataset in ("regdb", "llcm", "sysu"):
        group = sorted([r for r in rows if r["dataset"] == dataset],
                       key=lambda r: math.ceil(r["width"]/8)*math.ceil(r["height"]/8))
        selections = [("median", group[len(group)//2]), ("p95", group[int(len(group)*.95)]),
                      ("largest_padded_area", group[-1]),
                      ("widest", max(group, key=lambda r: r["width"])),
                      ("tallest", max(group, key=lambda r: r["height"]))]
        seen = set()
        for label, row in selections:
            if row["source"] in seen:
                continue
            seen.add(row["source"])
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            local_config = dict(config, tile=math.ceil(max(row["width"], row["height"])/8)*8)
            result = {"dataset": dataset, "selection": label, "source": row["source"],
                      "width": row["width"], "height": row["height"]}
            print(json.dumps({"starting": result}), flush=True)
            started = time.perf_counter()
            try:
                with Image.open(p.source_path(row, config)) as im:
                    pixels = np.asarray(im.convert("RGB"))[None]
                output = p.infer(model, pixels, row["modality"], "cuda:0", local_config)
                result.update(status="ok", output_shape=list(output.shape))
                del output
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                result.update(status="oom", error=str(exc))
            result.update(seconds=time.perf_counter()-started,
                          peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                          peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
            results.append(result)
            print(json.dumps(result), flush=True)
    p.write_json(p.HERE/"whole_image_probe.json", {"gpu_uuid": uuid, "model": metadata, "results": results})


if __name__ == "__main__":
    main()

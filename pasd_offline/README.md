# PASD Offline Generator

This package builds fixed-size PASD images before SALT training. It is isolated
from `src/salt_vi` and uses the pinned implementation in `vendor/pasd`.

## Structure

```text
pasd_offline/
├── configs/                 # generation settings
├── pasd_offline/            # config, records, geometry, runtime, generation, scheduler
├── scripts/
│   ├── build_sysu_records.py
│   ├── generate_dataset.py
│   └── validate_dataset.py
├── tests/
├── vendor/pasd/             # pinned PASD source
└── checkpoints/             # local weights, ignored by Git
```

Generated data belongs under `/home/lab929/datasets/derived`, not in this
repository. The formal single-view RGB dataset is:

```text
/home/lab929/datasets/derived/SYSU-MM01-pasd-rgb-x4-blurpad-512x256-1view-v1/
```

## Environment

```bash
conda create -n salt-pasd-offline python=3.10.19 -y
conda activate salt-pasd-offline
python -m pip install -r requirements-lock.txt
```

The model layout is configured in YAML. The standard paths are
`checkpoints/stable-diffusion-v1-5` and `checkpoints/pasd/checkpoint-100000`.

## Build records

One canonical JSONL schema supports either one or five views per source:

```bash
python scripts/build_sysu_records.py \
  --dataset-root /home/cgv841/datasets/SYSU-MM01 \
  --rgb-candidates /home/lab929/ybj/datasets/text_candidates/SYSU-MM01/Blip_RGB_Qwen3_14B_AWQ/caption_qwen3_14b_awq_4x.json \
  --views-per-source 1 \
  --output /home/lab929/datasets/derived/SYSU-MM01-pasd-rgb-x4-blurpad-512x256-1view-v1/source-records.jsonl \
  --pilot-output /home/lab929/datasets/derived/SYSU-MM01-pasd-rgb-x4-blurpad-512x256-1view-v1/pilot-records.jsonl
```

Use `--views-per-source 5` to build the original-caption plus four-paraphrase
variant. The record builder preserves the supplied captions verbatim.

## Generate

The same command handles local and dynamic multi-GPU generation:

```bash
python scripts/generate_dataset.py \
  --config configs/generate_sysu_rgb_single.yaml \
  --records /home/lab929/datasets/derived/SYSU-MM01-pasd-rgb-x4-blurpad-512x256-1view-v1/source-records.jsonl \
  --workers 2
```

`--workers 1` runs in the foreground. Values 2 or 3 use the configured idle
physical GPUs; GPU 0 is rejected. Generation is resumable per source and writes
PNG files, source metadata, `build.json`, `manifest.jsonl`, and `manifest.json`.
The build fingerprint binds the generation config and records file.

Single-view generation preserves the complete source frame over a blurred
same-image background. Both one- and five-view outputs are 256×512 RGB PNGs.

## Validate and consume

```bash
python scripts/validate_dataset.py \
  --config configs/generate_sysu_rgb_single.yaml \
  --records /home/lab929/datasets/derived/SYSU-MM01-pasd-rgb-x4-blurpad-512x256-1view-v1/source-records.jsonl
```

Stage B points `sysu_sr_view_manifest` to the published `manifest.jsonl` and
sets `sysu_sr_views_per_image` to the same value, 1 or 5. The sibling
`manifest.json` records completeness, counts, checksums, and the build
fingerprint.

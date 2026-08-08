# PASD Offline Generator

This directory is an independent offline image-generation module inside
SALT-VI. It does not import `src/salt_vi`, and SALT-VI training does not import
this module.

The runtime uses the vendored PASD snapshot in `vendor/pasd` and local model
assets in `checkpoints/`. It must not reference the separate `PASD` or
`PASD-reproducibility` workspaces.

## Environment

Use the dedicated Python 3.10 environment. `requirements.txt` pins the direct
runtime dependencies; `requirements-lock.txt` pins the full tested dependency
closure, including the CUDA 11.8 PyTorch stack.

```bash
conda create -n salt-pasd-offline python=3.10.19 -y
conda activate salt-pasd-offline
python -m pip install -r requirements-lock.txt
python -m pip check
```

Do not install the separate PASD checkout into this environment. The module
loads the pinned source snapshot from `vendor/pasd`.

## Layout

```text
pasd_offline/
├── configs/generate_sysu.yaml
├── pasd_offline/                 # generation package
├── scripts/generate_one.py       # one image + one caption
├── scripts/generate_dataset.py   # JSONL batch generation
├── vendor/pasd/                  # pinned PASD inference/model source
├── checkpoints/                  # local runtime weights, ignored by Git
└── tests/                        # CPU-only task/config tests
```

Generated datasets belong under the public data root, not in this repository.
The adaptive five-view output is:

```text
/home/lab929/ybj/datasets/derived/SYSU-MM01-pasd-rgb-adaptive-512x256-5view-v1/
```

## Caption input

Single-image generation accepts a caption verbatim:

```bash
/home/lab929/miniconda3/envs/salt-pasd-offline/bin/python \
  scripts/generate_one.py \
  --config configs/generate_sysu.yaml \
  --image /path/to/input.png \
  --caption "a person wearing a red shirt and black trousers" \
  --modality rgb \
  --output previews/red-shirt.png
```

Batch generation reads JSONL. A row may contain either `caption` or
`captions`:

```json
{"image":"/path/to/0001.png","caption":"a person in a red shirt","output":"rgb/0001.png","modality":"rgb"}
{"image":"/path/to/0002.png","captions":["a person in black","a pedestrian carrying a bag"],"output":"rgb/0002.png","modality":"rgb"}
```

Caption modes:

- `first`: use the first caption;
- `random`: choose one caption with a reproducible seed;
- `all`: generate one output per caption.

```bash
/home/lab929/miniconda3/envs/salt-pasd-offline/bin/python \
  scripts/generate_dataset.py \
  --config configs/generate_sysu.yaml \
  --records /path/to/records.jsonl \
  --caption-mode all
```

The batch command writes `manifest.json` and `manifest.jsonl` beside the
generated images. Each entry records the exact caption, seed, input path,
output path, and SHA-256 hashes.

## Build SYSU caption records

Use the existing per-image BLIP captions:

```bash
python scripts/build_sysu_records.py \
  --dataset-root /home/cgv841/datasets/SYSU-MM01 \
  --caption-dict /home/cgv841/datasets/SYSU-MM01/Text/Blip_RGB/caption_dict_Blip_RGB.json \
  --caption-dict /home/cgv841/datasets/SYSU-MM01/Text/Blip_IR/caption_dict_Blip_IR.json \
  --caption-scope image \
  --output /home/lab929/datasets/derived/SYSU-MM01-pasd-x2-v1/source-records.jsonl
```

For identity-level candidate sampling, add the two `id_caption_map` files and
set `--caption-scope identity`. This writes compact records with a
`caption_pool_key` plus a shared `.caption-pool.json` file. Pass that pool to
`generate_dataset.py --caption-pool ...`; the same records can then be generated
with `first`, seeded `random`, or `all` mode.

## Model assets

Expected local layout:

```text
checkpoints/
├── stable-diffusion-v1-5/
│   ├── feature_extractor/
│   ├── scheduler/
│   ├── text_encoder/
│   ├── tokenizer/
│   └── vae/
└── pasd/checkpoint-100000/
    ├── controlnet/
    └── unet/
```

These paths are configurable. Keeping model loading in `runtime.py` and the
model definitions in `vendor/pasd` allows a future training entry point to use
the same checkpoint layout without changing the offline dataset interface.

Install the official PASD archive without using any external checkout:

```bash
python scripts/install_pasd_checkpoint.py --archive /path/to/pasd.zip
```

Official archive identity:

- size: `5,453,737,067` bytes;
- SHA-256: `48eb5f434791f3d0d7c1b36c0aaf1040d935c0a065cae96d7e35336af9df93b7`.

## Adaptive SYSU five-view build

The active contract contains the official `29,033` RGB images. Each RGB source
uses its original caption plus four Qwen paraphrases and produces five PASD
views. IR remains on the original SYSU image path and has no caption
augmentation.

```bash
python scripts/build_sysu_multiview_records.py \
  --dataset-root /home/cgv841/datasets/SYSU-MM01 \
  --rgb-candidates /home/lab929/ybj/datasets/text_candidates/SYSU-MM01/Blip_RGB_Qwen3_14B_AWQ/caption_qwen3_14b_awq_4x.json \
  --output /home/lab929/ybj/datasets/derived/SYSU-MM01-pasd-rgb-adaptive-512x256-5view-v1/source-records.jsonl \
  --pilot-output /home/lab929/ybj/datasets/derived/SYSU-MM01-pasd-rgb-adaptive-512x256-5view-v1/pilot-records.jsonl
```

The dynamic launcher uses only physical GPUs 1, 2, and 3.  GPU0 is rejected by
both the scheduler and every worker.  Idle workers claim one source at a time
with `flock`, write each PNG atomically, and write a source completion marker
only after all five views pass local validation.

```bash
/home/lab929/miniconda3/envs/salt-pasd-offline/bin/python \
  scripts/launch_dynamic.py \
  --config configs/generate_sysu.yaml \
  --records /path/to/pilot-records.jsonl \
  --max-workers 3
```

The scheduler derives the source count and completion state from the supplied
records. The full RGB build contains `145,165` PNG views.

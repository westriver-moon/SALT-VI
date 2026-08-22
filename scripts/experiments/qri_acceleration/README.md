# QRI acceleration pilot

This directory contains the isolated GPU0 pilot used to identify a scalable
QRI flow. Large outputs are written only below:

`/home/lab929/ybj/experiments/qri_acceleration/qri-fast-search-gpu0-20260821`

The pilot has independently restartable measurement stages:

1. `prepare_pilot.py` builds a deterministic camera-balanced SwinIR/ROI set.
2. `qwen_pilot.py` compares one-call planning with the three-round proposer.
3. `pasd_pilot.py` performs a bounded PASD resolution/step study.
4. `quality_audit.py` checks LR-cycle consistency and PMT-ViT identity cosine.
5. `benchmark_swin_batch.py` measures the official SwinIR batch-size curve.

The selected full-data flow is implemented as two resumable phases:

1. `build_fast_dataset.py` runs batch-16 SwinIR, one cached SAM image encode,
   blur-only ranking, and saves only the top two ROI masks per image.
2. `qwen_group_plans.py` selects one anchor per identity-modality group and
   caches one no-thinking Qwen response. Use `--dry-run` to inspect anchors
   without starting a Qwen server.

The locked parameters and guarded optional PASD thresholds are in
`configs/experiments/qri_acceleration/selected_flow_gpu0.yaml`. PASD is off by
default; the conservative all-image output is the SwinIR image. This keeps the
full flow useful even when the diffusion backend is replaced later.

Example phase A invocation (after exporting the same QRI variables required by
`plugins/qwen_imagination/configs/qri_v1_sysu.yaml`):

```bash
python scripts/experiments/qri_acceleration/build_fast_dataset.py \
  --config plugins/qwen_imagination/configs/qri_v1_sysu.yaml \
  --output-root /home/lab929/ybj/experiments/qri_acceleration/qri-fast-search-gpu0-20260821/artifacts/selected_flow \
  --device cuda:0 --batch-size 16 --top-regions 2
```

Phase B is resumable because every group plan is written atomically:

```bash
python scripts/experiments/qri_acceleration/qwen_group_plans.py \
  --config plugins/qwen_imagination/configs/qri_v1_sysu.yaml \
  --output-root /home/lab929/ybj/experiments/qri_acceleration/qri-fast-search-gpu0-20260821/artifacts/selected_flow \
  --endpoint http://127.0.0.1:18080/v1/chat/completions \
  --model-id third-party-qwen3.8-27b-ud-q4-k-xl
```

Production QRI files are not modified by the pilot.

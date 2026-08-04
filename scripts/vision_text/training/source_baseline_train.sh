#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

GPU="${GPU:-0}"
DATA_ROOT="${DATA_ROOT:-/home/cgv841/datasets/SYSU-MM01}"
PRETRAIN="${PRETRAIN:-pretrained/jx_vit_base_p16_224-80ecf9dd.pth}"
OUTPUT="${OUTPUT:-logs/raw/source_baseline/outputs/vision_text/official_reproduction}"

python -m salt_vi.baselines.vision_text.train \
  --config configs/vision_text/sysu_baseline.yaml \
  --data-root "$DATA_ROOT" \
  --pretrained "$PRETRAIN" \
  --output "$OUTPUT" \
  --device "cuda:${GPU}"


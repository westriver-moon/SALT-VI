#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

GPU="${GPU:-0}"
DATA_ROOT="${DATA_ROOT:-/home/cgv841/datasets/SYSU-MM01}"
WEIGHTS="${WEIGHTS:-logs/raw/source_baseline/outputs/vision_text/official_reproduction/checkpoints/best.pth}"
MODE="${MODE:-all}"
GALLERY_MODE="${GALLERY_MODE:-single}"
TRIALS="${TRIALS:-10}"

python -m salt_vi.baselines.vision_text.test \
  --config configs/vision_text/sysu_baseline.yaml \
  --data-root "$DATA_ROOT" \
  --weights "$WEIGHTS" \
  --mode "$MODE" \
  --gallery-mode "$GALLERY_MODE" \
  --trials "$TRIALS" \
  --device "cuda:${GPU}"


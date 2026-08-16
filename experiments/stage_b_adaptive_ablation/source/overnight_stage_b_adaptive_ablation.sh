#!/usr/bin/env bash
set -euo pipefail

cd /home/cgv841/ybj/SALT-VI

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook)"
fi

conda activate clipreid

python scripts/smoke_stage_b_adaptive_ablation.py

python scripts/run_stage_b_adaptive_ablation.py \
  --gpus 0,1,2,3 \
  --max-parallel 4 \
  --max-mem 2000 \
  --max-util 20 \
  --poll-seconds 60

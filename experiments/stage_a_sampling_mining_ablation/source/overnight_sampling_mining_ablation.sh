#!/usr/bin/env bash
set -euo pipefail

cd /home/cgv841/ybj/SALT-VI
source /home/cgv841/anaconda3/etc/profile.d/conda.sh
conda activate clipreid

mkdir -p train_outputs/sampling_mining_ablation

select_idle_gpu() {
  python - <<'SMOKEPY'
import subprocess

out = subprocess.check_output([
    "nvidia-smi",
    "--query-gpu=index,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits",
], text=True)
best = None
for line in out.splitlines():
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 3:
        continue
    idx = int(float(parts[0]))
    mem = int(float(parts[1]))
    util = 0 if parts[2].upper() == "N/A" else int(float(parts[2]))
    if mem < 2000 and util < 20:
        item = (mem, util, idx)
        if best is None or item < best:
            best = item
if best is not None:
    print(best[2])
SMOKEPY
}

SMOKE_GPU=""
while [[ -z "${SMOKE_GPU}" ]]; do
  SMOKE_GPU="$(select_idle_gpu || true)"
  if [[ -z "${SMOKE_GPU}" ]]; then
    echo "No idle GPU for smoke; waiting 60s..."
    sleep 60
  fi
done
echo "Running smoke on physical GPU ${SMOKE_GPU}"
CUDA_VISIBLE_DEVICES="${SMOKE_GPU}" python scripts/smoke_sampling_mining_ablation.py

python scripts/run_sampling_mining_ablation.py \
  --gpus 0,1,2,3 \
  --max-parallel 4 \
  --max-mem 2000 \
  --max-util 20

python scripts/summarize_sampling_mining_ablation.py

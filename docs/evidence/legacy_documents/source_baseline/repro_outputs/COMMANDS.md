# SALT-VI migrated evidence document

> Source document ID: `source_baseline:repro_outputs/COMMANDS.md`  
> Original SHA-256: `a3032a8b3198ca211ea96d043b5e401a53afb643f0d94c3fabe6731283337943`  
> This is read-only experiment evidence, not an active runtime instruction.

# Commands

## Environment

```bash
cd /home/cgv841/ybj/SALT-VI vision-text baseline
/home/cgv841/anaconda3/envs/reid/bin/python -m pip install 'pytest>=7.0'
```

## Static Import Check

```bash
cd /home/cgv841/ybj/SALT-VI vision-text baseline
python -m compileall -q src/salt_vi/baselines/vision_text
```

## Unit Tests

```bash
cd /home/cgv841/ybj/SALT-VI vision-text baseline
/home/cgv841/anaconda3/envs/reid/bin/python -m pytest src/salt_vi/baselines/vision_text/tests -q
```

## Download Weights

```bash
cd /home/cgv841/ybj/SALT-VI vision-text baseline
/home/cgv841/anaconda3/envs/reid/bin/python -m salt_vi.baselines.vision_text.tools.download_weights --imagenet
/home/cgv841/anaconda3/envs/reid/bin/python -m salt_vi.baselines.vision_text.tools.download_weights --official
```

## Preflight

```bash
cd /home/cgv841/ybj/SALT-VI vision-text baseline
CUDA_VISIBLE_DEVICES=0 /home/cgv841/anaconda3/envs/reid/bin/python -m salt_vi.baselines.vision_text.tools.preflight \
  --config configs/vision_text/sysu_baseline.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --pretrained pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  --device cuda:0
```

## Smoke Training

```bash
cd /home/cgv841/ybj/SALT-VI vision-text baseline
CUDA_VISIBLE_DEVICES=0 /home/cgv841/anaconda3/envs/reid/bin/python -m salt_vi.baselines.vision_text.train \
  --config configs/vision_text/sysu_baseline.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --pretrained pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  --output logs/raw/source_baseline/outputs/vision_text/smoke \
  --device cuda:0 \
  --smoke-batches 1
```

```bash
cd /home/cgv841/ybj/SALT-VI vision-text baseline
CUDA_VISIBLE_DEVICES=0 /home/cgv841/anaconda3/envs/reid/bin/python -m salt_vi.baselines.vision_text.train \
  --config configs/vision_text/sysu_baseline.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --pretrained pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  --output logs/raw/source_baseline/outputs/vision_text/smoke_epoch7 \
  --device cuda:0 \
  --smoke-batches 1 \
  --override train.start_epoch=7
```

## Official Checkpoint 1-Trial Evaluation

```bash
cd /home/cgv841/ybj/SALT-VI vision-text baseline
CUDA_VISIBLE_DEVICES=0 /home/cgv841/anaconda3/envs/reid/bin/python -m salt_vi.baselines.vision_text.test \
  --config configs/vision_text/sysu_baseline.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --weights pretrained/pmt_sysu_vit_official.pth \
  --mode all \
  --gallery-mode single \
  --trials 1 \
  --device cuda:0 \
  --output logs/raw/source_baseline/outputs/vision_text/official_weight_trial1
```

## Full Training Command

```bash
cd /home/cgv841/ybj/SALT-VI vision-text baseline
GPU=0 DATA_ROOT=/home/cgv841/datasets/SYSU-MM01 \
  PRETRAIN=pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  OUTPUT=logs/raw/source_baseline/outputs/vision_text/official_reproduction \
  bash scripts/vision_text/training/source_baseline_train.sh
```

## Full 10-Trial Test Command

```bash
cd /home/cgv841/ybj/SALT-VI vision-text baseline
GPU=0 DATA_ROOT=/home/cgv841/datasets/SYSU-MM01 \
  WEIGHTS=logs/raw/source_baseline/outputs/vision_text/official_reproduction/checkpoints/best.pth \
  TRIALS=10 \
  bash scripts/vision_text/testing/source_baseline_test.sh
```

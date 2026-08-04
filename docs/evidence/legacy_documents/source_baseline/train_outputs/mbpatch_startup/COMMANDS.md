# SALT-VI migrated evidence document

> Source document ID: `source_baseline:train_outputs/mbpatch_startup/COMMANDS.md`  
> Original SHA-256: `14648f156ed6c10aa015c39f73da1305ecdcf8356185a834ab759ec39593b0a8`  
> This is read-only experiment evidence, not an active runtime instruction.

# Commands

## Static Check

```bash
cd /home/cgv841/ybj/SALT-VI vision-text baseline
python -m compileall -q src/salt_vi/baselines/vision_text
```

## Unit Tests

```bash
cd /home/cgv841/ybj/SALT-VI vision-text baseline
/home/cgv841/anaconda3/envs/reid/bin/python -m pytest src/salt_vi/baselines/vision_text/tests -q
```

## Preflight

```bash
cd /home/cgv841/ybj/SALT-VI vision-text baseline
CUDA_VISIBLE_DEVICES=0 /home/cgv841/anaconda3/envs/reid/bin/python -m salt_vi.baselines.vision_text.tools.preflight \
  --config configs/sysu_vision_text_mbpatch.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --pretrained pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  --device cuda:0
```

## Epoch 1 Smoke

```bash
cd /home/cgv841/ybj/SALT-VI vision-text baseline
CUDA_VISIBLE_DEVICES=0 /home/cgv841/anaconda3/envs/reid/bin/python -m salt_vi.baselines.vision_text.train \
  --config configs/sysu_vision_text_mbpatch.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --pretrained pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  --output logs/raw/source_baseline/outputs/vision_text/mbpatch_smoke_ep1 \
  --device cuda:0 \
  --smoke-batches 1 \
  --override data.num_workers=0
```

## Epoch 7 Smoke

```bash
cd /home/cgv841/ybj/SALT-VI vision-text baseline
CUDA_VISIBLE_DEVICES=0 /home/cgv841/anaconda3/envs/reid/bin/python -m salt_vi.baselines.vision_text.train \
  --config configs/sysu_vision_text_mbpatch.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --pretrained pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  --output logs/raw/source_baseline/outputs/vision_text/mbpatch_smoke_ep7 \
  --device cuda:0 \
  --smoke-batches 1 \
  --override data.num_workers=0 \
  --override train.start_epoch=7
```

## Full Training Candidate

```bash
cd /home/cgv841/ybj/SALT-VI vision-text baseline
CUDA_VISIBLE_DEVICES=0 /home/cgv841/anaconda3/envs/reid/bin/python -m salt_vi.baselines.vision_text.train \
  --config configs/sysu_vision_text_mbpatch.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --pretrained pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  --output logs/raw/source_baseline/outputs/vision_text/mbpatch_reproduction \
  --device cuda:0
```

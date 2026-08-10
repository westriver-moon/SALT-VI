# R-Text-Visual + LLM augmentation (40 epochs)

This experiment is a controlled variant of
`configs/stage_b/r_text_visual_20260729.yaml`, the configuration associated
with the historical 84.078% result.

Controlled changes:

- `llm_aug: true` with the existing `llm_aug_prob: 0.5`
- `total_train_epoch: 40`
- experiment identity and isolated output/event paths

All loss, optimizer, learning-rate, scheduler, data, and evaluation settings
remain unchanged. Training starts from the original Stage-B initialization
checkpoint (`a3_e4_hpt_l025/e4/model_Fusion_21.pth`, SHA-256
`7cd15b7b12ba138ccf6590f850dd72a11bfea16d7723a4ce101b98b0a3b1996c`),
not from the historical 84.078% trained checkpoint.

Runtime assignment: physical GPU 1 (`CUDA_VISIBLE_DEVICES=1`, local
`gpu_id=0`).

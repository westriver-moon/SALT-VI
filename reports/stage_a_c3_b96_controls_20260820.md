# Stage-A C3 batch-96 controls archive (2026-08-20)

All three runs use C3 (camera-diverse PK sampling + cosine normalized classifier),
effective RGB+IR batch 96, seed 0, and the official SYSU-MM01 all-search,
single-shot, 10-gallery-trial protocol.

| Run | Input | Best epoch | Rank-1 | mAP | mINP |
|---|---|---:|---:|---:|---:|
| C3-B96 | SwinIR RGB+IR | 23 | 0.699343 | 0.681841 | 0.564855 |
| C3-B96-PASD-x4 | PASD-x4 RGB+IR, blur-pad geometry aligned | 21 | 0.701394 | 0.684604 | 0.568300 |
| C3-B96-E36 | SwinIR RGB+IR, 36 epochs | 23 | 0.682093 | 0.666802 | 0.547290 |

The retained model for each run is the training loop's Rank-1-selected best model;
mAP and mINP are from the same evaluation epoch. The shared PMT-ViT starting
weight is stored once through a hard link under `experiments/archive/stage_a/_shared`.
All optimizer/latest checkpoints were hashed, recorded in each retention manifest,
and removed. The E36 original shell launcher was unavailable; its completion is
reconstructed transparently from the resolved configuration and complete epoch
0-35 event stream.

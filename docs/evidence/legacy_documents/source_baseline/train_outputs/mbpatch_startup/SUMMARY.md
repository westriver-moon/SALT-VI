# SALT-VI migrated evidence document

> Source document ID: `source_baseline:train_outputs/mbpatch_startup/SUMMARY.md`  
> Original SHA-256: `5838580103aa5cd406f864632dca9dc0708bef058ad965153a38fec10326a06d`  
> This is read-only experiment evidence, not an active runtime instruction.

# PMT-MBPatch Startup Summary

Date: 2026-06-15

Goal: add a configurable multi-branch patch embedding variant for PMT and verify the training startup path.

Implemented:

- Added `MultiBranchPatchEmbedOverlap` with anchor-grid token count.
- Added `configs/sysu_vision_text_mbpatch.yaml`.
- Kept the original `configs/vision_text/sysu_baseline.yaml` baseline unchanged.
- Added tests for baseline model shape, MBPatch model shape, and single-branch patch weight loading into MBPatch.

Verification:

- `python -m compileall -q src/salt_vi/baselines/vision_text`: passed.
- `/home/cgv841/anaconda3/envs/reid/bin/python -m pytest src/salt_vi/baselines/vision_text/tests -q`: 9 passed.
- MBPatch preflight on SYSU-MM01 with ImageNet ViT weights: passed.
- MBPatch epoch 1 one-batch smoke: passed.
- MBPatch epoch 7 one-batch smoke: passed.

Startup metrics:

- Epoch 1 smoke: `stage=gray_ir`, `loss=24.2250`, `msel=0.0000`, `dcl=0.0000`.
- Epoch 7 smoke: `stage=rgb_ir`, `loss=32.0959`, `msel=20.6255`, `dcl=0.9145`.
- Preflight feature shape: `(64, 768)` for both stages.

Full training status:

- Started full 24-epoch PMT-MBPatch training at `2026-06-15 20:39:07 Asia/Shanghai`.
- The run reached epoch 2, confirming that the formal training path is active.
- The foreground run was moved to tmux after epoch 2 checkpointing.
- Current background session: `pmt_mbpatch_full`, resumed from `latest.pth` at epoch 3.

Conclusion: the PMT-MBPatch variant is implemented and full training has been kicked off. This is not a final training result and does not yet prove metric improvement.

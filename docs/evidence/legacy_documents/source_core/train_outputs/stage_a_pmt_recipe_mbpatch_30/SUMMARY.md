# SALT-VI migrated evidence document

> Source document ID: `source_core:train_outputs/stage_a_pmt_recipe_mbpatch_30/SUMMARY.md`  
> Original SHA-256: `12fddab783ccb0a28a0b9b41dfae28605ad5702a5d0c345942dfb80a9413d1f3`  
> This is read-only experiment evidence, not an active runtime instruction.

# Stage A PMT Recipe MBPatch 30-Epoch Run

Goal: launch the multi-patch PMT ViT Stage A experiment on an idle GPU.

Run mode: full kickoff.

Selected configuration:

- `configs/stage_a/vision_text_encoder_stage_a_vision_text_recipe_288x144_768_mbpatch.yaml`
- Dataset: SYSU-MM01 at `/home/cgv841/datasets/SYSU-MM01/`
- Image size: `288 x 144`
- Backbone: `PMT_VIT`
- Projection/head dimension: `768`
- Patch embedding:
  - anchor branch: `16 x 16`, stride `12 x 12`
  - second branch: `16 x 8`, stride `12 x 6`
- Training schedule: `30` epochs
- Evaluation: IR query to RGB gallery, every 2 epochs from epoch 2.

Startup status:

- GPU0 was selected because it was idle before launch.
- Training started in tmux session `salt_vi_mbpatch_30`.
- Training process PID: `214475`.
- Initial model construction succeeded and the process is running on GPU0.

Main evidence:

- Launcher log: `reports/experiment_registry/source_tables/source_core/train_outputs/stage_a_pmt_recipe_mbpatch_30/a1_mbpatch_launcher.log`
- Training log: `logs/raw/source_core/logs/stage_a_vision_text_encoder_recipe_288x144_768_mbpatch_run1/sysu/Base/Baseline_train[RGB_IR]_pmt_recipe/logs/log.log`
- Config snapshot: `logs/raw/source_core/logs/stage_a_vision_text_encoder_recipe_288x144_768_mbpatch_run1/sysu/Base/Baseline_train[RGB_IR]_pmt_recipe/configs.yaml`

# SALT-VI migrated evidence document

> Source document ID: `source_core:train_outputs/stage_a_pmt_recipe_size_compare/SCIENTIFIC_CHANGELOG.md`  
> Original SHA-256: `7cbe3ad3838f624dcae433c60b893c29eb7df74abc5694324143f7894b1435bf`  
> This is read-only experiment evidence, not an active runtime instruction.

# Scientific Changelog

- Added controlled PMT-recipe input-size comparison configs:
  - `configs/stage_a/vision_text_encoder_stage_a_vision_text_recipe_256x128.yaml`
  - `configs/stage_a/vision_text_encoder_stage_a_vision_text_recipe_288x144.yaml`
- Intended scientific variable: image resize/crop geometry only, `256x128` versus `288x144`.
- Engineering-only fix: `solver/build.py` now uses `AdamWSkipEmptyGrad` for `AdamW` on this older PyTorch build. This is not a method contribution; it preserves AdamW behavior for parameters with gradients and skips groups with no gradients for that step.
This experiment group does not add a new model method. It isolates input resolution for the already implemented A1 PMT-recipe variant.

Controlled variable:

- `256 x 128` versus `288 x 144` train/test input size.

Held fixed:

- `pretrain_choice: PMT_VIT`
- `training_mode: RGB_IR`
- `joint_mode: image_only`
- `loss_names: pmt_recipe`
- PMT ImageNet checkpoint
- PMT progressive epoch count
- PMT loss weights
- AdamW and cosine schedule
- batch size, `num_pos`, seed, SYSU data path, and IR evaluation modality

Text assets, text fusion, and official PMT SYSU checkpoint initialization are not used.

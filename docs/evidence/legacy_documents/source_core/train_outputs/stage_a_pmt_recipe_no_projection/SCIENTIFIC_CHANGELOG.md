# SALT-VI migrated evidence document

> Source document ID: `source_core:train_outputs/stage_a_pmt_recipe_no_projection/SCIENTIFIC_CHANGELOG.md`  
> Original SHA-256: `489087682f6df22b079f18add0f3529beeafd2f7709f39bb3e8d5de4f8e9dba4`  
> This is read-only experiment evidence, not an active runtime instruction.

# Scientific Changelog

Changed:

- Added `configs/stage_a/vision_text_encoder_stage_a_vision_text_recipe_288x144_768.yaml`.
- Set `prj_output_dim: 768`, causing PMT visual projection to become identity.
- Set a new output path: `logs/raw/source_core/logs/stage_a_vision_text_encoder_recipe_288x144_768_run1/`.
- Set GPU selection to physical GPU 0.

Held constant against the 2048-dimensional 288x144 PMT-recipe comparison:

- Dataset and SYSU protocol.
- PMT ImageNet ViT-B/16 initialization.
- `288x144` input size.
- Patch size, stride, ViT depth, heads, MLP ratio, dropout, and drop path.
- PMT recipe transforms.
- PMT progressive schedule, Triplet, MSEL, and DCL weights.
- AdamW, LR, weight decay, warmup, cosine schedule, batch size, `num_pos`, seed.
- Image-only training and IR evaluation.

Meaning:

- This isolates the effect of the 768-dimensional PMT-native head versus the
  previous 2048-dimensional SALT-VI-compatible projected head as much as the
  current codebase allows.

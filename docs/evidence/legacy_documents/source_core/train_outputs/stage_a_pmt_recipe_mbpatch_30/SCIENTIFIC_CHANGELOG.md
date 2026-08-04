# SALT-VI migrated evidence document

> Source document ID: `source_core:train_outputs/stage_a_pmt_recipe_mbpatch_30/SCIENTIFIC_CHANGELOG.md`  
> Original SHA-256: `f897e234f081787573399e8f946ad91452828ba885576a907f875b59933e4c62`  
> This is read-only experiment evidence, not an active runtime instruction.

# Scientific Changelog

This run differs from the completed `288x144 / 768 no-projection` PMT recipe run by adding a multi-patch visual patch embedding and increasing the schedule from 24 to 30 epochs.

Changed:

- Enabled two-branch PMT patch embedding.
- Added a second branch with `16 x 8` patches and `12 x 6` stride.
- Fused branch feature maps with a `1 x 1` convolution before tokenization.
- Set `total_train_epoch: 30`.

Held constant against the previous closest Stage A PMT recipe run:

- SYSU-MM01 dataset path and protocol.
- `288 x 144` image size.
- `PMT_VIT` ViT-B style backbone.
- ImageNet ViT-B/16 checkpoint initialization.
- `prj_output_dim: 768`.
- `pmt_recipe` losses and progressive schedule.
- IR-only evaluation.

Interpretation boundary:

This is not a pure SALT-VI vision-text baseline reproduction. It remains a SALT-VI Stage A integration with a PMT-style backbone and training recipe.

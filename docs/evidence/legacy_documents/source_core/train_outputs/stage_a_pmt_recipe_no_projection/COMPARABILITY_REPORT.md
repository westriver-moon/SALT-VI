# SALT-VI migrated evidence document

> Source document ID: `source_core:train_outputs/stage_a_pmt_recipe_no_projection/COMPARABILITY_REPORT.md`  
> Original SHA-256: `b1081d76ed64b20c657fe7833406eb9fa6c8f9d94c61f5d50b7e155b173094a9`  
> This is read-only experiment evidence, not an active runtime instruction.

# Comparability Report

Most direct comparison:

- Current run: `A1 PMT recipe 288x144, prj_output_dim=768`.
- Reference run: `A1 PMT recipe 288x144, prj_output_dim=2048`.

Comparable:

- Same dataset, split, evaluation modality, and gallery-trial averaging path.
- Same PMT backbone configuration and ImageNet initialization.
- Same PMT recipe losses and augmentation.
- Same training length and evaluation cadence.
- Same seed and batch layout.

Not comparable as a single-factor SALT-VI vision-text baseline reproduction:

- The run still uses SALT-VI/CLIP2ReID training and evaluation code.
- It does not include SALT-VI vision-text baseline MBPatch.
- Text projection is resized to 768 but text is frozen and unused in Stage A.

Primary hypothesis:

- If 768 improves over 2048, the random 2048 projection/head likely hurt PMT
  optimization or retrieval representation.
- If 768 underperforms, the wider SALT-VI head may be helping this integration,
  or the 768 branch may need retuning.

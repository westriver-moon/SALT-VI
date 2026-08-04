# SALT-VI migrated evidence document

> Source document ID: `source_core:train_outputs/stage_a_pmt_recipe_size_compare/COMPARABILITY_REPORT.md`  
> Original SHA-256: `10839309cd22c3261464c22842f06a6ee509d3bc7610cf0d32999a13250da212`  
> This is read-only experiment evidence, not an active runtime instruction.

# Comparability Report

- The two active runs are comparable to each other because all Stage A PMT-recipe settings are held fixed except `img_h/img_w/img_size`.
- Both runs use SYSU-MM01, seed `0`, batch size `32`, `num_pos: 4`, `RGB_IR` image-only training, IR evaluation, PMT ImageNet ViT initialization, AdamW, cosine LR, and 24 total epochs.
- The optimizer patch is shared by both runs, so it does not bias the 256x128 versus 288x144 comparison.
- These runs should not be compared directly to earlier A0/A1 curves without noting the PMT-recipe changes: progressive augmentation, PMT losses, AdamW, cosine schedule, and 24-epoch training.
The two runs are intended to be compared directly against each other.

Comparable settings:

- Same dataset and cached image arrays
- Same PMT_VIT architecture and ImageNet initialization
- Same PMT-recipe training logic
- Same optimizer and LR schedule
- Same batch size, `num_pos`, and seed
- Same IR evaluation mode

Non-comparable or cautionary settings:

- The 288x144 run has more pixels and more ViT tokens, so it uses more memory and compute per epoch.
- The 288x144 run requires a different positional-embedding resize target than the 256x128 run.
- Comparison to A0 RN50, prior A1 40-epoch, or PMT official checkpoint remains contextual rather than strict.

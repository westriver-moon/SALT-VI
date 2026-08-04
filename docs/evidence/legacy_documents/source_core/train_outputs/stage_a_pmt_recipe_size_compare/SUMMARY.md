# SALT-VI migrated evidence document

> Source document ID: `source_core:train_outputs/stage_a_pmt_recipe_size_compare/SUMMARY.md`  
> Original SHA-256: `11a7297d78f92c544a07f67c68e5662d7caa94731700717aa262353145e16eb1`  
> This is read-only experiment evidence, not an active runtime instruction.

# Stage A PMT Recipe Size Comparison Summary

Run mode: full kickoff.

Goal: compare PMT-recipe Stage A runs under only the input-size variable.

- `A1R_256x128`: PMT recipe, image size `256 x 128`, GPU0.
- `A1R_288x144`: PMT recipe, image size `288 x 144`, GPU1.

Held fixed:

- PMT_VIT ImageNet initialization
- SYSU-MM01 dataset
- image-only `RGB_IR` training
- PMT progressive schedule, losses, optimizer, LR schedule, batch size, `num_pos`, and seed
- IR evaluation protocol

Variable:

- train/test input image size and resulting PMT token grid.

Dataset assumption: SYSU-MM01 image arrays are available under `/home/cgv841/datasets/SYSU-MM01/`.

Checkpoint assumption: PMT ImageNet ViT-B/16 checkpoint is available at `/home/cgv841/ybj/SALT-VI/pretrained/jx_vit_base_p16_224-80ecf9dd.pth`.

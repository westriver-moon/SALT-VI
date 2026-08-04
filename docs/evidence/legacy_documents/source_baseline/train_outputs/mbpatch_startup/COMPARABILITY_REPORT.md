# SALT-VI migrated evidence document

> Source document ID: `source_baseline:train_outputs/mbpatch_startup/COMPARABILITY_REPORT.md`  
> Original SHA-256: `8458d0e336b2c2890eef70cd774efa26468ea892423dba9bacc012e10a4a73f7`  
> This is read-only experiment evidence, not an active runtime instruction.

# Comparability Report

The PMT-MBPatch variant is directly comparable to the current SALT-VI vision-text baseline baseline if trained and evaluated with:

- the same SYSU-MM01 data root;
- the same ImageNet ViT initialization;
- the same 24 epoch training schedule;
- the same all-search single-shot 10-trial evaluation.

What remains comparable:

- Data split and evaluation protocol.
- Loss definitions.
- Batch layout.
- Optimizer family and schedule.

What changes:

- Backbone patch embedding capacity increases.
- Parameter count increases to `87,590,400` total parameters.
- ImageNet pretraining is not strict-identical because newly added branch and fusion parameters have no direct original checkpoint keys.

Current evidence supports only startup correctness, not metric improvement.

# SALT-VI canonical document

This document consolidates related legacy material. All configuration, code, data and output references below have been rewritten to the SALT-VI layout.


---

## Migrated source: README

> Source document ID: `source_core:tools/text_consistency/README.md`  
> Original SHA-256: `643ff97e4b88b27e7adb4af66b1e29eef9cb0d25c3425a18f759c507d7678e18`  
> Canonical runtime: `/home/lab929/ybj/SALT-VI/src/salt_vi/` and `/home/lab929/ybj/SALT-VI/scripts/`
> This section is rewritten for the SALT-VI layout; it is not an active compatibility layer.

# SYSU-MM01 identity-conflict text corrector

This module creates a new text tree and never edits the source tree.

Rules are intentionally narrow:

1. Read all caption dictionaries, covering train, validation, and test IDs.
2. An attribute is editable only when at least two normalized values occur for
   the same identity.
3. The selected value must already occur in that identity.
4. Prefer the value that makes the complete identity attribute signature
   collide with fewer other identities. Then prefer the value used by fewer
   other identities, stronger within-ID support, and lexical order.
5. Replace only the exact spans belonging to conflicting attributes. Do not
   insert missing attributes and do not rewrite grammar or unrelated errors.
6. Write the complete compatible tree, decisions, compressed per-string changes,
   validation, source hashes, and output hashes into a new directory.

Example:

```bash
python -m tools.text_consistency.sysu_conflict_corrector \
  --source-root datasets/sysu/Text \
  --dataset-root /home/cgv841/datasets/SYSU-MM01 \
  --output-root datasets/derived/sysu_text_id_consistent_v1
```

The generated directory can be passed directly as `text_data_root` because it
retains the original `Blip_RGB`, `Blip_IR`, and `GIT_RGB` layout.


---

## Migrated source: README

> Source document ID: `source_core:tools/super_resolution/README.md`  
> Original SHA-256: `3909065f1ab00d84954f84a8f30b6c9eaa83ef949611ff994e97e385840ddfbd`  
> Canonical runtime: `/home/lab929/ybj/SALT-VI/src/salt_vi/` and `/home/lab929/ybj/SALT-VI/scripts/`
> This section is rewritten for the SALT-VI layout; it is not an active compatibility layer.

# SYSU-MM01 SwinIR x2 assets

The formal SR ablation uses the official SwinIR-M classical SR x2 model trained on DF2K.
The upstream checkout and checkpoint live outside this repository. The builder enforces
the official Git revision and checkpoint/network SHA-256 values before inference, then
records them in the immutable manifest.

Generation standardizes the historical `384x144` SYSU training arrays and all evaluation
images to `288x144`, so bicubic and SwinIR groups use the same low-resolution source.
SwinIR runs explicitly in FP32, checks finite output before uint8 conversion, and performs
a real-checkpoint smoke forward before allocating full arrays. IR is converted to
luminance before inference and forced back to equal three-channel grayscale afterward.

The retired `SYSU-MM01-swinir-x2-v2` example is intentionally not runnable.
Use the current, fully pinned build and validation commands in
[`SYSU_SWINIR_PMT256_DATASET_BUILD_CONFIG.md`](../protocols/super_resolution/SYSU_SWINIR_PMT256_DATASET_BUILD_CONFIG.md).
That document records the exact source size, resampling policy, checkpoint,
implementation revision, output identity, and rebuild path.

Pinned identity:

- SwinIR revision: `6545850fbf8df298df73d81f3e8cba638787c8bd`
- checkpoint SHA-256: `2032ebf8f401dd3ce2fae5f3852117cb72101ec6ed8358faa64c2a3fa09ed4ac`
- `models/network_swinir.py` SHA-256: `9e143898679ebeebc5d2fc94ad1b89c38aa4a4d43da4e0fcba0f93e476994913`

Before any asset is written, the builder creates an immutable build identity bound to the
source arrays and labels, evaluation tree, checkpoint, SwinIR revision, builder hash,
batch/device settings, and FP32 policy. Interrupted arrays and evaluation images are reused
only when that identity matches exactly; provenance-free finalized files are rejected.
Full validation scans semantic content, minimum/P05/mean source consistency, and per-camera
evaluation coverage; `--quick` skips only hashes. The source dataset is opened read-only and
is never modified.

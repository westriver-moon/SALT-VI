# SALT-VI canonical document

This document consolidates related legacy material. All configuration, code, data and output references below have been rewritten to the SALT-VI layout.


---

## Migrated source: stage a results

> Source document ID: `source_core:docs/stage_a_results.md`  
> Original SHA-256: `23fcb06acafae02f1390036b502f226e2cea4d880d8fd907a44758778d11f787`  
> Canonical runtime: `/home/cgv841/ybj/SALT-VI/src/salt_vi/` and `/home/cgv841/ybj/SALT-VI/scripts/`  
> This section is rewritten for the SALT-VI layout; it is not an active compatibility layer.

# Stage A Results

## Sampling and Mining Ablation Result

Experiment id: `sampling_mining_ablation_20260706`

All six sampling/mining ablation runs completed successfully. The selected
current-best run is `h5_pk8x4_auto_replace_crossmodal_hard`.

Current best metrics:

| Metric | Value |
|---|---:|
| Rank-1 | 67.09% |
| mAP | 65.08% |
| mINP | 52.00% |

The Stage A main configuration is updated from
`identity_current_replace + PK=8x4 + pmt_hard` to
`identity_auto_replace + PK=8x4 + pmt_cross_modal_hard`.

Current main configuration:

```yaml
sampler_type: identity_auto_replace
triplet_mining: pmt_cross_modal_hard
batch_size: 32
num_pos: 4
pmt_triplet_margin: 0.1
```

The stable Stage A backbone and training settings remain unchanged:
`PMT_VIT + PMT recipe + 288x144 + prj_output_dim=768`, AdamW,
`lr_visual=0.0003`, cosine scheduler, `warmup_epochs=3`,
`target_lr_factor=0.01`, `total_train_epoch=24`, `eval_start_epoch=2`,
`eval_epoch=2`, and `seed=0`.

| Run | Rank-1 | mAP | mINP | Label | Decision |
|---|---:|---:|---:|---|---|
| `s0_pk8x4_current_replace_hard` | 65.53 | 64.11 | 51.65 | old baseline / historical baseline | Keep as historical baseline only; do not use as default. |
| `s1_pk8x4_auto_replace_hard` | 66.43 | 64.58 | 51.63 | sampler baseline | Keep as sampler baseline; use `identity_auto_replace`, but not plain `pmt_hard`, as main. |
| `s2_pk16x2_auto_replace_hard` | 65.31 | 64.34 | 52.09 | negative / secondary result | Keep as secondary control; do not select PK=16x2 as default. |
| `s3_pk4x8_auto_replace_hard` | 66.54 | 63.65 | 50.34 | not selected | Keep as record only; do not select PK=4x8 as default. |
| `h1_pk8x4_auto_replace_wrt` | 64.03 | 62.30 | 49.02 | failed / not suitable | Keep as failed control; do not use WRT as default mining. |
| `h5_pk8x4_auto_replace_crossmodal_hard` | 67.09 | 65.08 | 52.00 | current best / selected main config | Select as current main configuration. |

# Stage-A safe-trick C1–C3 archive

- Runs completed: 2026-08-18 20:25 CST
- Archived: 2026-08-19
- Code commit used by the runs: `9f02ba663e449ae3da69cb9a239c281a0b68d5ea`
- Dataset/protocol: SYSU-MM01, IR-to-visible, all-search, single-shot, ten fixed gallery trials
- Seed: 0
- Epochs: 24
- Archived run root: `/home/lab929/ybj/experiments/archive/stage_a/SALT-VI-safe-tricks-pairs-c1-c3-20260818`
- Selection policy: fixed final epoch 23; no test-metric checkpoint selection; no automatic Stage-B promotion

| Variant | Rank-1 | mAP | mINP |
|---|---:|---:|---:|
| C1 EMA + Cosine Softmax | 0.6922956109 | 0.6789477484 | 0.5678756533 |
| C2 EMA + Camera-diverse | 0.6939784288 | 0.6754745298 | 0.5560781198 |
| C3 Camera-diverse + Cosine Softmax | 0.7029975653 | 0.6864646066 | 0.5725203372 |

The archive retains all event files, raw logs, resolved `configs.yaml` files,
run manifests, selected best models, shared starting-weight reference, source-config snapshots,
the pipeline and scheduler snapshots, the environment capture, the exact Git
bundle, and the verified file inventory.

The safe-trick pipeline remains preregistered as
`A0 -> A1-A6 -> C1-C3 -> B0 -> B1-B6`. Variants are selected through
`configs/pipelines/sysu_safe_tricks.yaml` and YAML `extends`; no code branch is
switched between variants. Stage-B still requires an explicitly supplied and
hashed Stage-A checkpoint. This archive does not select or install one.

- Retention update (2026-08-20): optimizer/resume checkpoints were removed after SHA256 recording; the shared PMT start is a single hard-linked physical copy.

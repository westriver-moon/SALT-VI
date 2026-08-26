# Stage-A safe-trick A1–A6 archive

- Completed: 2026-08-18 12:44 CST
- Code commit used by the runs: `85871cf6131e2337f852a3e404821647d9c84680`
- Dataset/protocol: SYSU-MM01, IR-to-visible, all-search, single-shot, ten fixed gallery trials
- Seed: 0
- Epochs: 24
- Archived run root: `/home/lab929/ybj/experiments/archive/stage_a/SALT-VI-safe-tricks-a1-a6-dual-20260817`
- Selection policy: fixed final epoch 23; no test-metric checkpoint selection for the table below

| Variant | Rank-1 | mAP | mINP |
|---|---:|---:|---:|
| A1 EMA | 0.6911123395 | 0.6770329786 | 0.5619058959 |
| A2 Camera-diverse | 0.6953458190 | 0.6774422112 | 0.5579270486 |
| A3 Hetero-Center | 0.6090455055 | 0.5856238473 | 0.4473782055 |
| A4 RFA | 0.6106758118 | 0.5854033755 | 0.4456133438 |
| A5 Cosine Softmax | 0.6880620718 | 0.6742926787 | 0.5630333774 |
| A6 all-in combination | 0.6021825075 | 0.5700305013 | 0.4225537545 |

The server archive retains every event file, raw log, resolved `configs.yaml`, run manifest,
selected best model, shared starting-weight reference, source-config snapshot, environment capture, exact Git
bundle, and SHA-256 inventory. A6 underperformed A1, A2, and A5, so the next preregistered
round tests only the three pairwise combinations C1–C3.

- Retention update (2026-08-20): optimizer/resume checkpoints were removed after SHA256 recording; the shared PMT start is a single hard-linked physical copy.

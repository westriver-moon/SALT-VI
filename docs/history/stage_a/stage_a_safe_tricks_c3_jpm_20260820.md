# Stage-A safe-trick C3-JPM archive

- Runs completed: 2026-08-20
- Archived: 2026-08-20
- Code commit: `32e1b807`
- Dataset/protocol: SYSU-MM01, IR-to-visible, all-search, single-shot, ten fixed gallery trials
- Seed: 0
- Epochs: 24
- Selection policy: fixed final epoch 23; no test-metric checkpoint selection

| Run | Variant | Status | Rank-1 | mAP | mINP |
|---|---:|---|---:|---:|---:|
| run3-gpu2 | C3-JPM | completed | 0.654325 | 0.638471 | 0.512629 |
| run2 | C3-JPM-partmix | completed | 0.594241 | 0.601038 | 0.491310 |
| run2 | C3-JPM | failed during epoch 0 (signal 15) | - | - | - |

Archive roots:

- `/home/lab929/ybj/experiments/archive/stage_a/SALT-VI-safe-tricks-c3-jpm-20260820-run3-gpu2`
- `/home/lab929/ybj/experiments/archive/stage_a/SALT-VI-safe-tricks-c3-jpm-20260820-run2`

Each archive retains event files, raw logs, resolved `configs.yaml`, run manifests,
selected best models, source-config snapshots, pipeline/scheduler snapshots,
environment capture, exact Git bundle, retention manifest, and SHA256 inventory.
Resume/optimizer checkpoints were removed under the starting-and-best-only policy.

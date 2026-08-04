# SALT-VI migrated evidence document

> Source document ID: `source_core:train_outputs/stage_a_group_current/COMMANDS.md`  
> Original SHA-256: `3c5848c018224a619cfdba8d8716a8696ddc84dcff86a87eb12bb3a7e3107342`  
> This is read-only experiment evidence, not an active runtime instruction.

# Stage A Training Commands

Started from repository root:

```bash
cd /home/cgv841/ybj/SALT-VI
source /home/cgv841/anaconda3/etc/profile.d/conda.sh
conda activate clipreid

python scripts/train.py --config_select configs/stage_a/rn50_ori_stage_a_control.yaml
python scripts/train.py --config_select configs/stage_a/vision_text_encoder_stage_a.yaml
```

Launcher log:

```text
/home/cgv841/ybj/SALT-VI/reports/experiment_registry/source_tables/source_core/reports/experiment_registry/source_tables/source_core/train_outputs/stage_a_group_current/launcher.log
```

The original kickoff ran commands sequentially on GPU0 through one background launcher.

Runtime control update on 2026-06-18 21:19 CST:

- The parent launcher PID `2231196` was paused while A0 PID `2231211` continues running, so A1 will not auto-start after A0.
- `run_stage_a_group.sh` now requires `RUN_A1_AFTER_A0=1` before it will launch A1.
- A1 should be launched manually after reviewing A0 completion, using the 40-epoch config now stored in `configs/stage_a/vision_text_encoder_stage_a.yaml`.

Manual A1 launch on 2026-06-18 21:25 CST:

```bash
cd /home/cgv841/ybj/SALT-VI
source /home/cgv841/anaconda3/etc/profile.d/conda.sh
conda activate clipreid
python scripts/train.py --config_select configs/stage_a/vision_text_encoder_stage_a.yaml
```

A1 launcher log:

```text
/home/cgv841/ybj/SALT-VI/reports/experiment_registry/source_tables/source_core/reports/experiment_registry/source_tables/source_core/train_outputs/stage_a_group_current/a1_launcher.log
```

Actual launcher script:

```text
/home/cgv841/ybj/SALT-VI/reports/experiment_registry/source_tables/source_core/reports/experiment_registry/source_tables/source_core/train_outputs/stage_a_group_current/run_stage_a_group.sh
```

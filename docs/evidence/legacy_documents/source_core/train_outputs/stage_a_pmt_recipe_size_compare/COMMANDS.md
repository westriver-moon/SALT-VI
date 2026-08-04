# SALT-VI migrated evidence document

> Source document ID: `source_core:train_outputs/stage_a_pmt_recipe_size_compare/COMMANDS.md`  
> Original SHA-256: `afb3c6b4e33e3c79e94801ed8ecdf8e6b0cf29a400054090ab7b2279bfff2245`  
> This is read-only experiment evidence, not an active runtime instruction.

# Stage A PMT Recipe Size Comparison Commands

Started from repository root:

```bash
cd /home/cgv841/ybj/SALT-VI
source /home/cgv841/anaconda3/etc/profile.d/conda.sh
conda activate clipreid

python scripts/train.py --config_select configs/stage_a/vision_text_encoder_stage_a_vision_text_recipe_256x128.yaml
python scripts/train.py --config_select configs/stage_a/vision_text_encoder_stage_a_vision_text_recipe_288x144.yaml
```

The two commands are launched in parallel because GPU0 and GPU1 were both free at kickoff.

Launcher logs:

```text
/home/cgv841/ybj/SALT-VI/reports/experiment_registry/source_tables/source_core/reports/experiment_registry/source_tables/source_core/train_outputs/stage_a_pmt_recipe_size_compare/a1r_256x128_launcher.log
/home/cgv841/ybj/SALT-VI/reports/experiment_registry/source_tables/source_core/reports/experiment_registry/source_tables/source_core/train_outputs/stage_a_pmt_recipe_size_compare/a1r_288x144_launcher.log
/home/cgv841/ybj/SALT-VI/reports/experiment_registry/source_tables/source_core/reports/experiment_registry/source_tables/source_core/train_outputs/stage_a_pmt_recipe_size_compare/a1r_256x128_run1_launcher.log
/home/cgv841/ybj/SALT-VI/reports/experiment_registry/source_tables/source_core/reports/experiment_registry/source_tables/source_core/train_outputs/stage_a_pmt_recipe_size_compare/a1r_288x144_run1_launcher.log
/home/cgv841/ybj/SALT-VI/reports/experiment_registry/source_tables/source_core/reports/experiment_registry/source_tables/source_core/train_outputs/stage_a_pmt_recipe_size_compare/a1r_256x128_run2_launcher.log
/home/cgv841/ybj/SALT-VI/reports/experiment_registry/source_tables/source_core/reports/experiment_registry/source_tables/source_core/train_outputs/stage_a_pmt_recipe_size_compare/a1r_288x144_run2_launcher.log
```

Active clean run output dirs:

```text
/home/cgv841/ybj/SALT-VI/logs/raw/source_core/logs/stage_a_vision_text_encoder_recipe_256x128_run2/sysu/Base/Baseline_train[RGB_IR]_pmt_recipe
/home/cgv841/ybj/SALT-VI/logs/raw/source_core/logs/stage_a_vision_text_encoder_recipe_288x144_run2/sysu/Base/Baseline_train[RGB_IR]_pmt_recipe
```

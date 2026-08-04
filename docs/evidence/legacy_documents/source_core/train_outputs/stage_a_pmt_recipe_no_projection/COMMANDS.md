# SALT-VI migrated evidence document

> Source document ID: `source_core:train_outputs/stage_a_pmt_recipe_no_projection/COMMANDS.md`  
> Original SHA-256: `1842599740e3566430446cdd26473e224cc06644ab1c3af636728a3a2315a2d1`  
> This is read-only experiment evidence, not an active runtime instruction.

# Commands

Working directory:

```bash
cd /home/cgv841/ybj/SALT-VI
source /home/cgv841/anaconda3/etc/profile.d/conda.sh
conda activate clipreid
```

Pre-start design check:

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
# Loaded configs/stage_a/vision_text_encoder_stage_a_vision_text_recipe_288x144_768.yaml.
# Checked: image-only loader, batch layout, PMT no-projection Identity,
# classifier/text dimensions, frozen text tensors.
PY
```

Forward smoke check:

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
# Checked one full batch for current_epoch=0 and current_epoch=6.
# Both gray_ir and rgb_ir branches produced finite losses.
PY
```

Formal kickoff:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python scripts/train.py \
  --config_select configs/stage_a/vision_text_encoder_stage_a_vision_text_recipe_288x144_768.yaml \
  > /home/cgv841/ybj/SALT-VI/reports/experiment_registry/source_tables/source_core/reports/experiment_registry/source_tables/source_core/train_outputs/stage_a_pmt_recipe_no_projection/a1r_288x144_768_launcher.log 2>&1 &
```

Expected training log:

```text
/home/cgv841/ybj/SALT-VI/logs/raw/source_core/logs/stage_a_vision_text_encoder_recipe_288x144_768_run1/sysu/Base/Baseline_train[RGB_IR]_pmt_recipe/logs/log.log
```

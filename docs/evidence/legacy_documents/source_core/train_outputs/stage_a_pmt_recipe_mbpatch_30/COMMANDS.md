# SALT-VI migrated evidence document

> Source document ID: `source_core:train_outputs/stage_a_pmt_recipe_mbpatch_30/COMMANDS.md`  
> Original SHA-256: `4b9c80bb2c03943b49fa08258eb722f4f156fe084d264aba7f04e599a6af8a42`  
> This is read-only experiment evidence, not an active runtime instruction.

# Commands

GPU check:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits
```

Selected GPU:

```text
GPU0: NVIDIA GeForce RTX 3090, 11 MiB used, 0% utilization, no compute process before launch.
```

Full kickoff command:

```bash
cd /home/cgv841/ybj/SALT-VI
tmux new-session -d -s salt_vi_mbpatch_30 \
  "cd /home/cgv841/ybj/SALT-VI && source /home/cgv841/anaconda3/etc/profile.d/conda.sh && conda activate clipreid && python scripts/train.py --config_select configs/stage_a/vision_text_encoder_stage_a_vision_text_recipe_288x144_768_mbpatch.yaml 2>&1 | tee reports/experiment_registry/source_tables/source_core/train_outputs/stage_a_pmt_recipe_mbpatch_30/a1_mbpatch_launcher.log"
```

Monitor commands:

```bash
tmux attach -t salt_vi_mbpatch_30
tail -f '/home/cgv841/ybj/SALT-VI/logs/raw/source_core/logs/stage_a_vision_text_encoder_recipe_288x144_768_mbpatch_run1/sysu/Base/Baseline_train[RGB_IR]_pmt_recipe/logs/log.log'
nvidia-smi
```

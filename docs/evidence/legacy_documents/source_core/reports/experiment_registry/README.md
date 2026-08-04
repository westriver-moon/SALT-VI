# SALT-VI migrated evidence document

> Source document ID: `source_core:reports/experiment_registry/README.md`  
> Original SHA-256: `a50a705175bd1f83ebf8eb6f07fe4cae2542080018b4cab230059ab1d7cbba56`  
> This is read-only experiment evidence, not an active runtime instruction.

# SALT-VI Experiment Registry

- Generated: 2026-08-03 11:13:37
- Purpose: preserve the legacy Stage-A/early Stage-B subset and selected later experiment records with retained checkpoints.
- Scope: this table does **not** include the full metric-boost, A3-E4 HPT Stage-2/3, multistage-text, or RegDB leaderboards. The archived Qwen grid/pair-equal and SALT ablation entries are included as verified SYSU records; their source experiment directories remain authoritative.
- Bulky logs/raw/source_core/logs/checkpoints are intentionally not copied into Git. A non-empty checkpoint field means that the server-side file is currently retained; an empty field is a metrics-only record whose checkpoint was pruned.

## Highlights

- Best Rank-1 within this legacy Stage-A subset: `PMSR-A3-swinir-both-x2-tail-e32` epoch `30`, Rank-1 `0.69677`, mAP `0.68244`, mINP `0.56912`.
- Best Rank-1 among recorded Stage-B rows: `SALT_R_TEXT_VISUAL` epoch `23`, Rank-1 `0.84078`, mAP `0.81433`, mINP `0.71790`.
- The archived Qwen and SALT results are evidence only and are not promoted as the active default training recipe; the Stage-3 `PAIR-EQUAL` provenance record remains the active mainline reference.
- YAML configs retained in this branch: `58`.
- Local branches audited: `1`.

## Result Table

| stage | group | experiment | YAML | lifecycle | status | best_epoch | Rank-1 | mAP | mINP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stage_a | pmt_recipe_size_projection_mbpatch | pmt_recipe_256_2048_projection | `configs/stage_a/vision_text_encoder_stage_a_vision_text_recipe_256x128.yaml` | active | done | 23 | 0.63681 | 0.61888 | 0.48570 |
| stage_a | pmt_recipe_size_projection_mbpatch | pmt_recipe_288_2048_projection | `configs/stage_a/vision_text_encoder_stage_a_vision_text_recipe_288x144.yaml` | active | done | 23 | 0.64712 | 0.62417 | 0.48981 |
| stage_a | pmt_recipe_size_projection_mbpatch | pmt_recipe_288_768_mbpatch_30 | `configs/stage_a/vision_text_encoder_stage_a_vision_text_recipe_288x144_768_mbpatch.yaml` | active | done | 27 | 0.63847 | 0.62645 | 0.50403 |
| stage_a | pmt_recipe_size_projection_mbpatch | pmt_recipe_288_768_no_projection | `configs/stage_a/vision_text_encoder_stage_a_vision_text_recipe_288x144_768.yaml` | active | done | 21 | 0.65527 | 0.64108 | 0.51647 |
| stage_a | sampling_mining_ablation | h1_pk8x4_auto_replace_wrt | `configs/stage_a/sampling_mining_ablation/h1_pk8x4_auto_replace_wrt.yaml` | active | succeeded |  | 0.64030 | 0.62300 | 0.49020 |
| stage_a | sampling_mining_ablation | h5_pk8x4_auto_replace_crossmodal_hard | `configs/stage_a/sampling_mining_ablation/h5_pk8x4_auto_replace_crossmodal_hard.yaml` | active | succeeded |  | 0.67090 | 0.65080 | 0.52000 |
| stage_a | sampling_mining_ablation | s0_pk8x4_current_replace_hard | `configs/stage_a/sampling_mining_ablation/s0_pk8x4_current_replace_hard.yaml` | active | succeeded |  | 0.65530 | 0.64110 | 0.51650 |
| stage_a | sampling_mining_ablation | s1_pk8x4_auto_replace_hard | `configs/stage_a/sampling_mining_ablation/s1_pk8x4_auto_replace_hard.yaml` | active | succeeded |  | 0.66430 | 0.64580 | 0.51630 |
| stage_a | sampling_mining_ablation | s2_pk16x2_auto_replace_hard | `configs/stage_a/sampling_mining_ablation/s2_pk16x2_auto_replace_hard.yaml` | active | succeeded |  | 0.65310 | 0.64340 | 0.52090 |
| stage_a | sampling_mining_ablation | s3_pk4x8_auto_replace_hard | `configs/stage_a/sampling_mining_ablation/s3_pk4x8_auto_replace_hard.yaml` | active | succeeded |  | 0.66540 | 0.63650 | 0.50340 |
| stage_a | stage_a_group_current | A0_RN50_ORI | `configs/stage_a/rn50_ori_stage_a_control.yaml` | active | stopped_by_user | 107 | 0.52506 | 0.50583 | 0.36471 |
| stage_a | stage_a_group_current | A1_PMT_VIT | `configs/stage_a/vision_text_encoder_stage_a.yaml` | active | done | 31 | 0.24418 | 0.24567 | 0.13726 |
| stage_a | super_resolution_ablation | PMSR-A0-original-256 | `configs/experiments/reproduction/archived_configs/pmsr_a0_original_256.yaml` | archived | succeeded | 24 | 0.66229 | 0.64554 | 0.51810 |
| stage_a | super_resolution_ablation | PMSR-A1-bicubic-x2 | `configs/experiments/reproduction/archived_configs/pmsr_a1_bicubic_x2.yaml` | archived | succeeded | 24 | 0.68701 | 0.67405 | 0.55803 |
| stage_a | super_resolution_ablation | PMSR-A1-bicubic-x2-tail-e32 | `configs/experiments/reproduction/archived_configs/pmsr_a1_bicubic_x2_tail_e32.yaml` | archived | succeeded | 26 | 0.68601 | 0.67394 | 0.55927 |
| stage_a | super_resolution_ablation | PMSR-A3-swinir-both-x2 | `configs/experiments/reproduction/archived_configs/pmsr_a3_swinir_both_x2.yaml` | archived | succeeded | 22 | 0.69519 | 0.67711 | 0.55895 |
| stage_a | super_resolution_ablation | PMSR-A3-swinir-both-x2-tail-e32 | `configs/experiments/reproduction/archived_configs/pmsr_a3_swinir_both_x2_tail_e32.yaml` | archived | succeeded | 30 | 0.69677 | 0.68244 | 0.56912 |
| stage_b | qwen_text_augmentation | QWEN4_PAIR_EQUAL_BALANCED_RGB | `configs/experiments/reproduction/archived_configs/qwen4_pair_equal.yaml` | archived | succeeded | 14 | 0.83508 | 0.81317 | 0.72011 |
| stage_b | qwen_text_augmentation | QWEN_T1_BALANCED100_E30 | `configs/experiments/reproduction/archived_configs/qwen_t1_balanced100.yaml` | archived | succeeded | 24 | 0.83471 | 0.81312 | 0.71925 |
| stage_b | qwen_text_augmentation | QWEN_T2_BALANCED75_E30 | `configs/experiments/reproduction/archived_configs/qwen_t2_balanced75.yaml` | archived | succeeded | 24 | 0.83823 | 0.81537 | 0.72202 |
| stage_b | qwen_text_augmentation | QWEN_T3_BALANCED50_E30 | `configs/experiments/reproduction/archived_configs/qwen_t3_balanced50.yaml` | archived | succeeded | 27 | 0.83952 | 0.81592 | 0.72218 |
| stage_b | qwen_text_augmentation | QWEN_T4_IID100_E30 | `configs/experiments/reproduction/archived_configs/qwen_t4_iid100.yaml` | archived | succeeded | 27 | 0.83852 | 0.81508 | 0.72062 |
| stage_b | qwen_text_augmentation | QWEN_T5_CYCLE5_ORIGINAL_E30 | `configs/experiments/reproduction/archived_configs/qwen_t5_cycle5_original.yaml` | archived | succeeded | 24 | 0.83503 | 0.81298 | 0.71900 |
| stage_b | qwen_text_augmentation | QWEN_T6_DUAL_MEAN_E30 | `configs/experiments/reproduction/archived_configs/qwen_t6_dual_mean.yaml` | archived | succeeded | 20 | 0.83061 | 0.80895 | 0.71449 |
| stage_b | qwen_text_augmentation | QWEN_T7_QUALITY_PAIR_E30 | `configs/experiments/reproduction/archived_configs/qwen_t7_quality_pair.yaml` | archived | succeeded | 24 | 0.83765 | 0.81455 | 0.71993 |
| stage_b | salt_ablation | SALT_F_FULL_FAVTA30 | `configs/experiments/reproduction/archived_configs/salt_f_full_favta30.yaml` | archived | succeeded | 22 | 0.83639 | 0.81077 | 0.71430 |
| stage_b | salt_ablation | SALT_R_TEXT_VISUAL | `configs/experiments/reproduction/archived_configs/salt_r_text_visual.yaml` | archived | succeeded | 23 | 0.84078 | 0.81433 | 0.71790 |
| stage_b | salt_ablation | SALT_R_TWO_ANCHOR_GROUPS | `configs/experiments/reproduction/archived_configs/salt_r_two_anchor_groups.yaml` | archived | succeeded | 24 | 0.83450 | 0.81064 | 0.71571 |
| stage_b | token_aware_direct_supervision | d0_none | `configs/experiments/reproduction/archived_configs/d0_none.yaml` | archived | succeeded | 1 | 0.80153 | 0.77238 | 0.65839 |
| stage_b | token_aware_direct_supervision | d1_token_id | `configs/experiments/reproduction/archived_configs/d1_token_id.yaml` | archived | succeeded | 1 | 0.80150 | 0.77236 | 0.65856 |
| stage_b | token_aware_direct_supervision | d2_token_rgb_cm | `configs/experiments/reproduction/archived_configs/d2_token_rgb_cm.yaml` | archived | succeeded | 1 | 0.80297 | 0.77245 | 0.65725 |
| stage_b | token_aware_direct_supervision | d3_token_text_supcon | `configs/experiments/reproduction/archived_configs/d3_token_text_supcon.yaml` | archived | succeeded | 1 | 0.80032 | 0.77191 | 0.65833 |
| stage_b | token_aware_direct_supervision | d4_id_cm | `configs/experiments/reproduction/archived_configs/d4_id_cm.yaml` | archived | succeeded | 1 | 0.80302 | 0.77296 | 0.65849 |
| stage_b | token_aware_direct_supervision | d5_id_supcon | `configs/experiments/reproduction/archived_configs/d5_id_supcon.yaml` | archived | succeeded | 1 | 0.80145 | 0.77233 | 0.65845 |
| stage_b | token_aware_direct_supervision | d6_cm_supcon | `configs/experiments/reproduction/archived_configs/d6_cm_supcon.yaml` | archived | succeeded | 1 | 0.80400 | 0.77287 | 0.65761 |
| stage_b | token_aware_direct_supervision | d7_all | `configs/experiments/reproduction/archived_configs/d7_all.yaml` | archived | succeeded | 1 | 0.80124 | 0.77205 | 0.65799 |
| stage_b | token_aware_e4_warmstart | W1_token_aware_e4_warmstart | `configs/experiments/reproduction/archived_configs/w1_token_aware_e4_warmstart.yaml` | archived | succeeded | 1 | 0.79997 | 0.77189 | 0.65814 |
| stage_b | token_aware_pa05 | t1_text_query_all_tokens | `[historical config unavailable; see reports/experiment_registry/experiment_registry.csv]` | retired | done | 5 | 0.80084 | 0.76822 | 0.64978 |
| stage_b | token_aware_pa05 | t2_text_query_top32 | `[historical config unavailable; see reports/experiment_registry/experiment_registry.csv]` | retired | done | 5 | 0.80560 | 0.77176 | 0.65196 |
| stage_b | token_aware_pa05 | t3_text_query_top64 | `[historical config unavailable; see reports/experiment_registry/experiment_registry.csv]` | retired | done | 5 | 0.80442 | 0.77091 | 0.65129 |
| stage_b | token_aware_pa05 | t4_base_query_top32 | `[historical config unavailable; see reports/experiment_registry/experiment_registry.csv]` | retired | done | 13 | 0.80358 | 0.77294 | 0.65844 |
| stage_b | token_aware_pa05 | t5_ir_query_top32 | `[historical config unavailable; see reports/experiment_registry/experiment_registry.csv]` | retired | done | 5 | 0.80105 | 0.76750 | 0.64806 |
| stage_b | token_to_token_ablation | g0_add | `configs/experiments/reproduction/runs/source_core-result-archive-20260710/configs/g0_add.yaml` | retired | succeeded_reused | 13 | 0.81767 | 0.78509 | 0.67427 |
| stage_b | token_to_token_ablation | g1_parameter_add_pa05 | `configs/experiments/reproduction/runs/source_core-result-archive-20260710/configs/g1_parameter_add_pa05.yaml` | retired | succeeded | 9 | 0.81620 | 0.78663 | 0.67711 |
| stage_b | token_to_token_ablation | g2_current_token_aware_pa05 | `configs/experiments/reproduction/runs/source_core-result-archive-20260710/configs/g2_current_token_aware_pa05.yaml` | retired | succeeded | 5 | 0.80560 | 0.77176 | 0.65196 |
| stage_b | token_to_token_ablation | g3_cross_attention | `configs/experiments/reproduction/runs/source_core-result-archive-20260710/configs/g3_cross_attention.yaml` | retired | succeeded | 31 | 0.79582 | 0.77033 | 0.65980 |
| stage_b | token_to_token_ablation | n1_token_to_token_direct | `configs/experiments/reproduction/runs/source_core-result-archive-20260710/configs/n1_token_to_token_direct.yaml` | retired | succeeded | 31 | 0.73618 | 0.71861 | 0.60009 |
| stage_b | token_to_token_ablation | n2_token_to_token_residual_pa05 | `configs/experiments/reproduction/runs/source_core-result-archive-20260710/configs/n2_token_to_token_residual_pa05.yaml` | retired | succeeded | 9 | 0.80134 | 0.77270 | 0.65984 |

## Branch Consolidation Notes

| branch | sha | worktree | recommended_action |
| --- | --- | --- | --- |
| `main` | `91774670e` | `` | keep as default branch |

## Files

- `experiment_results.csv`: generated normalized view of this registry subset; it is not the global project leaderboard.
- `yaml_inventory.csv`: existing active/archived YAML configs with lifecycle, SHA-1, and key fields.
- `archived_results.csv`: source/input rows imported from completed runs outside the main worktree. Its rows intentionally reappear in the generated `experiment_results.csv`; the validator enforces equality to prevent drift.
- `configs/experiments/reproduction/archived_configs/`: YAML snapshots for completed external runs.
- `branch_audit.csv`: local branch/worktree audit used for consolidation.

# Stage-A C3 recipe and loss research: final archive

Date: 2026-08-31 (Asia/Shanghai)
Experiment group: `stage_a_c3_recipe_loss_research_20260829`
Dataset and protocol: SYSU-MM01, all-search, single-shot, infrared-to-visible, official aggregate over gallery trials 0-9
Training budget: 24 epochs, seed 0

## Final status

| Variant | Archival status | Selected epoch | Rank-1 | mAP | mINP |
|---|---|---:|---:|---:|---:|
| R0 current | completed | 23 | 70.0447% | 68.4326% | 56.7630% |
| R1 gray4 coupled | completed | 19 | 68.1909% | 66.9998% | 55.4992% |
| R2 gray4 decoupled ramp3 | completed | 23 | 67.9989% | 66.8993% | 55.6562% |
| L0 no auxiliary | completed | 23 | 63.6313% | 61.8771% | 49.0734% |
| L1 MSEL only | completed | 23 | 67.1707% | 64.2487% | 50.9349% |
| L2 DCL only | completed | 23 | 63.9837% | 64.1965% | 53.4158% |
| L4 MSEL0.50 DCL0.10 | completed | 23 | 68.1593% | 65.8003% | 53.2761% |
| M1 cross-modal margin0.05 | completed | 23 | 69.2848% | 66.3069% | 53.3065% |
| M2 cross-modal margin0.10 | completed | 19 | 68.7957% | 65.5661% | 52.2745% |
| M3 cross-modal margin0.20 | completed | 23 | 68.5038% | 65.8413% | 53.0060% |
| S0 selective control | completed | 23 | 58.8930% | 57.6024% | 44.8647% |
| S1 relation0.05 | failed_user_stopped（中止前部分指标） | 21 | 58.8246% | 57.5229% | 44.5028% |
| S2 relation0.10 | failed_user_stopped（首轮中止，无验证） | — | — | — | — |

Eleven variants completed the full schedule. S1 and S2 were explicitly stopped by the
user and are archived as failures (`exit_code=-15`), not as completed experiments.
S1's epoch-21 values are retained only as partial interrupted-run evidence; S2 has no
evaluation result.

## Conclusions

- R0 remains the group best: Rank-1 70.0447%, mAP 68.4326%, mINP 56.7630%.
- M1 is the strongest modified mining/loss recipe: Rank-1 69.2848%. M2 and M3 do not
  improve on M1 or R0, so increasing the cross-modal margin is not supported.
- L4 reaches Rank-1 68.1593%; its reduced DCL recipe is viable but does not beat R0.
- S0 is substantially below the main C3 line. S1 is essentially level with S0 before
  interruption, and S2 was stopped before evaluation. These runs do not support further
  selective Relation Loss hyperparameter search.
- D0 and D1 diagnostics are retained as supporting evidence and are not counted as
  training-result rows.

## Archive layout

- Retained original run root: `/home/lab929/ybj/experiments/stage_a/SALT-VI-c3-recipe-loss-research-20260829`
- Completed batch 1 (R0/R1/R2/L0/L1/L2/L4/M1): `/home/lab929/ybj/experiments/archive/stage_a/SALT-VI-c3-recipe-loss-research-20260829-completed-batch1`
- Final batch 2 (M2/M3/S0/S1/S2): `/home/lab929/ybj/experiments/archive/stage_a/SALT-VI-c3-recipe-loss-research-20260829-final-batch2`
- Web-readable Git index: `reports/experiment_archives/c3_recipe_loss_research_20260829/`

Both archive batches preserve the original run tree. Scientific payload copies were
verified byte-for-byte; no new checksum was calculated. The Git index contains the
full final metrics table, terminal scheduler state, experiment plan, diagnostics, and
all compact evaluation event streams. Large model/optimizer files remain in the server
archives and are intentionally not committed to Git.

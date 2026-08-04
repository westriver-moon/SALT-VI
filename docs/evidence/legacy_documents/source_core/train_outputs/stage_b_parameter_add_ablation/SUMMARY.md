# SALT-VI migrated evidence document

> Source document ID: `source_core:train_outputs/stage_b_parameter_add_ablation/SUMMARY.md`  
> Original SHA-256: `d37b4557bb3b6293573bbccc0deedd75754372eee143b5d62bf811c3d4a718af`  
> This is read-only experiment evidence, not an active runtime instruction.

# Stage B Parameter-Add Ablation Summary

| experiment | pa | best_epoch | Rank-1 | mAP | mINP | status |
| --- | --- | --- | --- | --- | --- | --- |
| e4_parameter_add_pa05_baseline | 0.5 | 9 | 0.81620 | 0.78663 | 0.67711 | done |
| e7_parameter_add_pa07 | 0.7 | 13 | 0.77910 | 0.74919 | 0.62987 | done |
| e10_parameter_add_pa03 | 0.3 | 13 | 0.77628 | 0.75698 | 0.64669 | done |

- Best by Rank-1: e4_parameter_add_pa05_baseline (pa=0.5, Rank-1=0.81620)
- Best by mAP: e4_parameter_add_pa05_baseline (pa=0.5, mAP=0.78663)
- Best by mINP: e4_parameter_add_pa05_baseline (pa=0.5, mINP=0.67711)
- Conclusion: pa=0.5 remains best across Rank-1 / mAP / mINP among pa in {0.3, 0.5, 0.7}: yes

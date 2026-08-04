# SALT-VI migrated evidence document

> Source document ID: `source_core:reports/token_aware_e4_warmstart/README.md`  
> Original SHA-256: `c7434523abe24d44dfca67b3797bfd0fcd1d483744340d079c41fcac4fce5874`  
> This is read-only experiment evidence, not an active runtime instruction.

# Token-aware E4 Warm-start W1

Official selection uses the epoch with the highest Rank-1; mAP and mINP are reported from that same epoch.

| Experiment | Initialization | Best epoch | Rank-1 | mAP | mINP | Delta vs E4 | Delta vs cold T4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E4 | E4 parameter_add pa=0.5 checkpoint | 9 | 0.81620 | 0.78663 | 0.67711 | - | - |
| T4 cold-start | Stage A checkpoint | - | 0.80358 | 0.77294 | 0.65844 | - | - |
| W1 | E4 best checkpoint warm-start | 1 | 0.79997 | 0.77189 | 0.65814 | R1 -1.623 pp; mAP -1.474 pp; mINP -1.897 pp | R1 -0.361 pp; mAP -0.105 pp; mINP -0.030 pp |

## Evaluation trace

| epoch | Rank-1 | mAP | mINP | gamma | id loss | wrt loss | total loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.79997 | 0.77189 | 0.65814 | -0.02531416155397892 | 0.06204411 | 0.2915326 | 0.35357678 |
| 3 | 0.79997 | 0.77189 | 0.65814 | -0.027608051896095276 | 0.10552909 | 0.29124686 | 0.39677596 |
| 5 | 0.79997 | 0.77189 | 0.65814 | -0.024270830675959587 | 0.06942765 | 0.2703239 | 0.33975166 |
| 7 | 0.79997 | 0.77189 | 0.65814 | -0.023806236684322357 | 0.051640432 | 0.25781184 | 0.30945224 |
| 9 | 0.79997 | 0.77189 | 0.65814 | -0.02247723937034607 | 0.044847433 | 0.25160816 | 0.29645568 |
| 11 | 0.79997 | 0.77189 | 0.65814 | -0.023146333172917366 | 0.044010896 | 0.24358703 | 0.28759807 |
| 13 | 0.79997 | 0.77189 | 0.65814 | -0.022044802084565163 | 0.038645178 | 0.23708595 | 0.27573115 |
| 15 | 0.79997 | 0.77189 | 0.65814 | -0.02292061783373356 | 0.028888326 | 0.2289446 | 0.25783262 |
| 17 | 0.79997 | 0.77189 | 0.65814 | -0.022825580090284348 | 0.02663242 | 0.22374903 | 0.25038126 |
| 19 | 0.79997 | 0.77189 | 0.65814 | -0.022556385025382042 | 0.02361744 | 0.21991515 | 0.24353264 |
| 21 | 0.79997 | 0.77189 | 0.65814 | -0.02411118522286415 | 0.020388905 | 0.2184635 | 0.23885235 |
| 23 | 0.79997 | 0.77189 | 0.65814 | -0.023708999156951904 | 0.018274456 | 0.21360533 | 0.23187971 |
| 25 | 0.79997 | 0.77189 | 0.65814 | -0.024687714874744415 | 0.017923726 | 0.21091107 | 0.22883452 |
| 27 | 0.79997 | 0.77189 | 0.65814 | -0.02463444322347641 | 0.016264156 | 0.21033135 | 0.22659561 |
| 29 | 0.79997 | 0.77189 | 0.65814 | -0.024571282789111137 | 0.016600404 | 0.20687035 | 0.22347078 |
| 31 | 0.79997 | 0.77189 | 0.65814 | -0.02453179657459259 | 0.014387992 | 0.20805693 | 0.22244523 |

E4 checkpoint: /home/cgv841/ybj/experiments/stageb-fusion-ablation/e4_no_sff_parameter_add_pa05/results/sysu/FV/Baseline_train[RGB_IR_Text]_joint[uni]_Blip_parameter_add_id,wrt_Fix_Visual/models/model_Fusion_9.pth
E4 checkpoint SHA256: 62ed940bf0fcb8ae5d87807d9c080ab64fe3ad2edad1dc83fc8d2530b54005e7
W1 best checkpoint: /home/cgv841/ybj/SALT-VI/logs/raw/source_core/logs/stage_b/token_aware_e4_warmstart/w1_base_query_top32_e4_warm/sysu/FV/Baseline_train[RGB_IR_Text]_joint[uni]_Blip_token_aware_parameter_add_id,wrt_Fix_Visual/models/model_Fusion_1.pth
Training time: 6:27:46
Git commit at training start: cb4be596896be0ab21438778321e60ac6b31d7c8

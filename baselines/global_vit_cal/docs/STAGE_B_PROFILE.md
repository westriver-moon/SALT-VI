# Stage-B code profile and Qwen offline bridge

`configs/stage_b_hard_mining.yaml` contains a standalone Stage-B training
profile: RGB, IR, text and parameter-add fusion views; identity plus
bidirectional cross-modal hard mining; a hard-loss ramp from epoch 3 through
epoch 7; and only RGB--IR, RGB--Text and IR--Text pairs enabled.

`stage_b.py` implements the model-independent hard-mining ramp and pair loss.
It can be used with the global CAL backbone without importing a legacy runner.

## Qwen use is offline only

`configs/stage_b_hard_mining_qwen_offline.yaml` enables an offline annotation
bridge. Feed each `QwenRegionalPipeline` JSON record to `retained_world_texts`
and tokenize its returned candidate strings before training. If a text encoder
returns `[B,K,D]` features and the corresponding `selected_weight` tensor is
`[B,K]`, use `weighted_world_feature_mean` to form the Stage-B `Text` feature.
The RGB/IR/Fusion/Text features then go to `weighted_cross_modal_hard_loss`.

No Qwen request is made during model training or retrieval.

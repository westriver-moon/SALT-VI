# SALT core refactor

This refactor separates experiment orchestration from model mathematics while
preserving existing YAML, checkpoints, result keys, and training entry points.

## Boundaries

### Training recipe

`salt_vi.training.recipes` owns the supported loss graphs:

- `PMTRecipe`
- `LegacyRGBIRRecipe`
- `LegacyRGBIRTextRecipe`
- `IRToRGBTextRecipe`

`CLIP2ReID` retains encoders, classifiers, fusion primitives, checkpoint I/O,
and trainability controls. Its `forward()` delegates to one recipe selected at
construction time.

### Retrieval protocol

Every retrieval mode is an explicit module in `salt_vi.retrieval`, including
`legacy`. A protocol owns:

- runtime validation specific to that retrieval mode;
- train/query/gallery text contracts;
- the associated training recipe;
- query and gallery names, result key, and evaluator dispatch.

Adding another retrieval mode requires a protocol module plus one registry
entry. Central validation, loader policy, model forward, and evaluator dispatch
do not gain another mode-specific branch.

### Data source

`salt_vi.data.sources` isolates image and caption access from
`SYSU_Tri_Data`. Array and PASD multiview inputs implement the same `sample()`
interface. Caption sources independently represent absent, array-backed, and
multiview captions. Dataset code only applies transforms and assembles the
batch contract.

## Preserved contracts

- `scripts/train.py --config_select ...` remains the entry point.
- Existing configuration keys and values retain their meaning.
- Feature encoding order remains RGB original, RGB augmented, then IR.
- Supported loss equations and weights are unchanged.
- Evaluation result keys remain `IR`, `Fusion`, `Text`, and `IR-RGBText`.
- Checkpoint parameter names are unchanged; recipes and protocols add no
  trainable state.
- `pasd_offline` remains an independent offline generator.

Unsupported historical `dual_text`, `ir_selffusion`, and `rgb_selffusion`
branches were removed from the active implementation. They were already
rejected by runtime configuration validation and remain available in Git
history.

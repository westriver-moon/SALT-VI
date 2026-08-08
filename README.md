# SALT-VI

SALT-VI is the canonical research implementation for visible-infrared person re-identification with RGB text supervision, two-stage training, super-resolution inputs, and LLM-based caption augmentation.

## Canonical default

The formal Stage-B default is `SALT_R_TEXT_VISUAL`, configured by
`configs/stage_b/r_text_visual_20260729.yaml`. It uses frozen visual features,
offline RGB+IR SwinIR x2 inputs, direct RGB-IR/RGB-Text/IR-Text pair losses,
multi-branch patch embedding, and learnable PatchGeM. The promotion decision,
evidence, and validation boundary are recorded in
`docs/protocols/SALT_R_TEXT_VISUAL_DEFAULT.md`.

## Repository layout

- `src/salt_vi/`: canonical implementation
- `configs/`: reproducible configurations and archived experiment YAML files
- `reports/experiment_registry/experiment_registry.csv`: experiment registry
- `docs/`: protocols, evidence indexes, and operational notes
- `scripts/`: training, validation, and analysis entry points
- `pasd_offline/`: independent caption-driven PASD dataset generator; not imported by the training package
- `checkpoints/`, `logs/`, `pretrained/`, `runtime/`: local runtime assets; intentionally excluded from Git

## Installation

```bash
python -m pip install -e ".[test]"
python -m pytest src/salt_vi/tests
PYTHONPATH=pasd_offline python -m pytest pasd_offline/tests
```

The repository does not distribute datasets or model weights. Configure public dataset roots, pretrained initialization, canonical checkpoints, and output paths before training. The experiment registry records checkpoint identities and SHA-256 values for retained runs.

## Training

Run the formal default with its explicit YAML configuration:

```bash
python scripts/train.py --config_select configs/stage_b/r_text_visual_20260729.yaml
```

`DataParallel` is intentionally unsupported. Use one process per GPU, or the validated `fixed_visual_data_parallel` mode when the visual branch is frozen.

## Reproducibility scope

The source repository contains code, configurations, tests, and experiment metadata. SYSU-MM01/RegDB/LLCM datasets, derived data arrays, pretrained weights, checkpoints, and raw logs remain local assets and must be obtained separately according to their respective licenses.

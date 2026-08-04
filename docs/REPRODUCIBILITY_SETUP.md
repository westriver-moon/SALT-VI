# External Reproducibility Setup

SALT-VI distributes source code, configurations, tests, and experiment metadata. It deliberately does **not** distribute third-party datasets, derived arrays, pretrained initialization files, model checkpoints, or raw training logs.

## Required local assets

1. Obtain SYSU-MM01, RegDB, and/or LLCM from their respective official sources and comply with their licenses.
2. Obtain the PMT ViT initialization file and place it below a local pretrained-assets directory.
3. Obtain the canonical Stage-A checkpoint required by a Stage-B configuration. Its experiment identity and SHA-256 are recorded in `reports/experiment_registry/experiment_registry.csv`.
4. Choose a writable output directory outside any source checkout if desired.

## Portable configuration

YAML loading expands `${VAR}` placeholders. Start from `configs/templates/runtime_paths.example.yaml`, set the following environment variables, and merge the needed fields into an explicit local YAML:

- `SALT_VI_SYSU_ROOT`
- `SALT_VI_REGDB_ROOT`
- `SALT_VI_LLCM_ROOT`
- `SALT_VI_TEXT_ROOT`
- `SALT_VI_PRETRAINED_ROOT`
- `SALT_VI_OUTPUT_ROOT`

Historical YAML files intentionally retain the paths recorded at the time of their runs. Do not edit those files for a new machine; create a new runtime YAML instead.

## Validation order

1. `python -m pip install -e .`
2. `python -m pytest -q src/salt_vi/tests src/salt_vi/baselines/vision_text/tests`
3. Run `python scripts/train.py --help`.
4. Run the project preflight for the selected configuration before launching a full experiment.

`DataParallel` is not supported. Use one process per GPU, or use the documented frozen-visual replica mode when applicable.

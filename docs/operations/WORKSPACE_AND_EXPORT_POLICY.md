# Workspace and Export Policy

This document describes the current SALT-VI layout and replaces the retired
workspace-level upload manifest.

## Canonical roots

- Project: /home/cgv841/ybj/SALT-VI
- Workspace Git root: /home/cgv841/ybj
- Public datasets: /home/cgv841/datasets
- Project code: SALT-VI/src/salt_vi/ and SALT-VI/scripts/
- Configs: SALT-VI/configs/
- Experiment records: SALT-VI/experiments/
- Logs and reports: SALT-VI/logs/ and SALT-VI/reports/
- Weights: SALT-VI/checkpoints/ and explicitly documented pretrained paths

## Git policy

Track source code, configs, tests, lightweight reports, resolved manifests, and
checksums. Do not track datasets, checkpoints, large logs, caches, generated
arrays, or temporary runtime state.

Before a commit, inspect:

    git -C /home/cgv841/ybj status --short
    git -C /home/cgv841/ybj diff --check

## Experiment policy

Each new run gets a directory under SALT-VI/experiments/<experiment_id>/.
Its configuration, status, metrics, seed, code state, and log locations must
be recorded in the canonical experiment registry. Training should use
SALT-VI/src/salt_vi/ and SALT-VI/scripts/, not retired source trees.

## Safety

Do not delete or relocate data, checkpoints, or files outside
/home/cgv841/ybj without explicit authorization. Preserve historical evidence
under SALT-VI/docs/archive/ or SALT-VI/reports/evidence/.

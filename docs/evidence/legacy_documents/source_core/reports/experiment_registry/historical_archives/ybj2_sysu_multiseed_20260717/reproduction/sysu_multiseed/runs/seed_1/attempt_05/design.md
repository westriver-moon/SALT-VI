# SALT-VI migrated evidence document

> Source document ID: `source_core:reports/experiment_registry/historical_archives/ybj2_sysu_multiseed_20260717/reproduction/sysu_multiseed/runs/seed_1/attempt_05/design.md`  
> Original SHA-256: `2cfb17befae0b7ceb8a6af9728d05ee1f7a3cf6c2cb422b9959384dbf01e6941`  
> This is read-only experiment evidence, not an active runtime instruction.

# Experiment design

- Hypothesis: The official SALT-VI SYSU configuration is reproducible under an independently recorded random seed.
- Declared baseline: official SALT-VI commit `682742130f2fb7bca26dabd92bc5a788225d7541` and the author configuration.
- Exact intervention: Change only the training seed to 1; train for 120 epochs from the shared official base checkpoint.
- Controlled variables: code commit, base checkpoint, dataset view, dependencies, batch size, optimizer configuration, GPU model, and evaluation protocol.
- Expected effect: successful completion with metrics recorded as observed; no numerical pass/fail threshold.
- Selection rule: retain and report every declared seed; no selective exclusion.
- Validity label: `exploratory-test-set-tuned`.

# SALT-VI migrated evidence document

> Source document ID: `source_core:reports/experiment_registry/historical_archives/ybj2_sysu_multiseed_20260717/reproduction/sysu_multiseed/preflight/smoke_seed1/design.md`  
> Original SHA-256: `25c0ad1f9e6de0a75343520f2f167c3ce56b398078c5535cddc29336f0f8944f`  
> This is read-only experiment evidence, not an active runtime instruction.

# Experiment design

- Hypothesis: The isolated environment and data view can complete one training epoch.
- Declared baseline: official SALT-VI commit `682742130f2fb7bca26dabd92bc5a788225d7541` and the author configuration.
- Exact intervention: Run seed 1 for one epoch with evaluation disabled; do not treat metrics as a formal result.
- Controlled variables: code commit, base checkpoint, dataset view, dependencies, batch size, optimizer configuration, GPU model, and evaluation protocol.
- Expected effect: successful completion with metrics recorded as observed; no numerical pass/fail threshold.
- Selection rule: retain and report every declared seed; no selective exclusion.
- Validity label: `standard`.

# SALT-VI migrated evidence document

> Source document ID: `source_core:reports/experiment_registry/historical_archives/ybj2_sysu_multiseed_20260717/reproduction/sysu_multiseed/preflight/aborted_official_mer_attempt_04_20260717T072347/design.md`  
> Original SHA-256: `1f8962a4c6e4a8ad6d9927aab3831481f03a99ee1168c25a8b748e8f1a2408df`  
> This is read-only experiment evidence, not an active runtime instruction.

# Experiment design

- Hypothesis: The official MER evaluation protocol executes deterministically on the supplied checkpoint.
- Declared baseline: official SALT-VI commit `682742130f2fb7bca26dabd92bc5a788225d7541` and the author configuration.
- Exact intervention: Run Fusion MER evaluation without modifying model code or checkpoint state.
- Controlled variables: code commit, base checkpoint, dataset view, dependencies, batch size, optimizer configuration, GPU model, and evaluation protocol.
- Expected effect: successful completion with metrics recorded as observed; no numerical pass/fail threshold.
- Selection rule: retain and report every declared seed; no selective exclusion.
- Validity label: `standard`.

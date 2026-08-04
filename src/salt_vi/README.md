# SALT-VI canonical implementation

????????????????? data?engine?models?optim?utils ? config ???

?????salt_vi.entrypoints.train:main?

??????? Token interaction / bidirectional token-cycle ???RGB/IR/Text ?????????????????????

## 2026-08-04 removal status

The Token interaction / bidirectional token-cycle implementation has been removed from both the canonical source and the retained vendor/source_core code copy. No active training entrypoint imports or calls it. Dedicated token-cycle YAML rows are retired in the experiment registry. Historical YAML/checkpoint/audit records remain only as evidence and are not executable entrypoints; see `runtime/migrations/token_interaction_removal_20260804/`.


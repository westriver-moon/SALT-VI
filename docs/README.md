# SALT-VI 文档索引

本目录收纳已按新项目结构改写的协议、结果、操作手册和历史证据。运行入口始终是 `scripts/train.py`，核心代码始终是 `src/salt_vi/`。

## Canonical documents
- [`docs/operations/GETTING_STARTED_AND_BASELINES.md`](operations/GETTING_STARTED_AND_BASELINES.md)
- [`docs/operations/TEXT_GENERATION.md`](operations/TEXT_GENERATION.md)
- [`docs/operations/TOKEN_CYCLE.md`](operations/TOKEN_CYCLE.md)
- [`docs/operations/VISION_TEXT_AND_SR_TOOLS.md`](operations/VISION_TEXT_AND_SR_TOOLS.md)
- [`docs/protocols/data_quality_and_external_screening.md`](protocols/data_quality_and_external_screening.md)
- [`docs/protocols/experiment_governance.md`](protocols/experiment_governance.md)
- [`docs/protocols/hyperparameter_search.md`](protocols/hyperparameter_search.md)
- [`docs/protocols/super_resolution/A3_E4_HPT_STAGE3.md`](protocols/super_resolution/A3_E4_HPT_STAGE3.md)
- [`docs/protocols/super_resolution/SYSU_SR_PROTOCOL_AND_AUDIT.md`](protocols/super_resolution/SYSU_SR_PROTOCOL_AND_AUDIT.md)
- [`docs/reference/RETIRED_IMPLEMENTATIONS.md`](reference/RETIRED_IMPLEMENTATIONS.md)
- [`docs/results/STAGE_A_AND_SAMPLING_RESULTS.md`](results/STAGE_A_AND_SAMPLING_RESULTS.md)

## Evidence
`docs/evidence/legacy_documents/` 保存逐实验叙述性证据；它们不作为新的训练入口。指标和配置的权威索引仍是 `reports/experiment_registry/experiment_registry.csv`。

## Path rules
- 配置：`configs/`；旧 `config/` 已按实验族改写。
- 代码：`src/salt_vi/` 与 `scripts/`。
- 数据：公共根目录 `/home/cgv841/datasets/`，项目不复制数据集。
- 日志/结果：`logs/` 与 `reports/`。
- 权重：本次不移动、不改名。
- [Workspace and export policy](operations/WORKSPACE_AND_EXPORT_POLICY.md)

# SALT-VI 文档迁移报告（2026-08-03）

## 结论

旧项目文档已按语义合并或作为只读证据改写到 `SALT-VI/docs/`。新文档中的配置、代码、数据和日志路径均指向 SALT-VI canonical 结构；权重和数据实体未移动。

## 合并文档

| 新文档 | 合并来源数 | 处理 |
|---|---:|---|
| `docs/operations/GETTING_STARTED_AND_BASELINES.md` | 2 | 逐段重写并合并重复协议/说明 |
| `docs/operations/TEXT_GENERATION.md` | 2 | 逐段重写并合并重复协议/说明 |
| `docs/operations/TOKEN_CYCLE.md` | 1 | 逐段重写并合并重复协议/说明 |
| `docs/operations/VISION_TEXT_AND_SR_TOOLS.md` | 2 | 逐段重写并合并重复协议/说明 |
| `docs/protocols/data_quality_and_external_screening.md` | 2 | 逐段重写并合并重复协议/说明 |
| `docs/protocols/experiment_governance.md` | 2 | 逐段重写并合并重复协议/说明 |
| `docs/protocols/hyperparameter_search.md` | 1 | 逐段重写并合并重复协议/说明 |
| `docs/protocols/super_resolution/A3_E4_HPT_STAGE3.md` | 2 | 逐段重写并合并重复协议/说明 |
| `docs/protocols/super_resolution/SYSU_SR_PROTOCOL_AND_AUDIT.md` | 5 | 逐段重写并合并重复协议/说明 |
| `docs/reference/RETIRED_IMPLEMENTATIONS.md` | 1 | 逐段重写并合并重复协议/说明 |
| `docs/results/STAGE_A_AND_SAMPLING_RESULTS.md` | 1 | 逐段重写并合并重复协议/说明 |

## 逐实验证据

共有 82 个 Markdown/RST 证据文件保留在 `docs/evidence/legacy_documents/`；它们不参与运行。

## 未迁移文件

requirements 文件和环境依赖输入没有伪装成文档迁移；它们由 `runtime/` 环境快照和 canonical 配置负责。

## 权威入口

- 代码：`src/salt_vi/`
- 配置：`configs/`
- 总表：`reports/experiment_registry/experiment_registry.csv`
- 迁移清单：`runtime/document_migration_manifest_20260803.json`

## 自动审计

- 文档数：95
- 旧项目/旧入口残留：0
- 无法解析的可运行配置引用：0

## 旧源文档清理

已删除旧目录中被 canonical 文档完整覆盖的 21 个协议、README 和工具手册；每个源文件的 SHA-256 与目标文档映射保存在 `runtime/document_migration_manifest_20260803.json`。

保留项：旧 `train_outputs/**`、`reports/**` 和 `repro_outputs/**` 下的逐实验 Markdown/RST 证据；它们已在 `docs/evidence/legacy_documents/` 中改写，旧副本暂不清理。权重、数据、代码和日志未触碰。

## 迁移后规范化

已修正视觉-文本基线输出目录的嵌套路径，并在 canonical 入门文档中加入 `pip install -e .`，确保 `salt_vi` 模块命令可解析。配置引用和旧入口审计重新计算，结果写入迁移清单。

## 源文档覆盖审计

- 旧项目 Markdown/RST 源文档：82 份。
- 已登记到 canonical 合并文档、证据副本或删除映射：82/82。
- 证据目标哈希校验：82/82 通过。
- 合并文档目标哈希校验：11/11 通过。
- 旧项目 `TVI-LFM/docs/` 和 `PMT-SYSU/docs/` 不再保留核心文档；旧 `train_outputs/`、`reports/`、`repro_outputs/` 中的原始逐实验记录作为只读证据保留。

# SALT-VI 统一实验总表

experiment_registry.csv 是 SALT-VI 唯一对外使用的实验总表。它按行保留全部实验记录，并用 record_type 区分汇总结果、逐 epoch 指标、训练曲线和清单记录。

source_tables/ 是总表所依赖的原始结果表集合，按 source_core、source_baseline、experiments 等语义子目录组织；它们与总表属于同一项目，不再设置旧项目/新项目的层级。总表通过 source_table、source_row_number、source_sha256 和 extra_metrics_json 保留每条记录与原始表的对应关系。

数据集和模型权重不复制到这里；总表只记录它们的路径、哈希和状态。

2026-08-10 完成的近期协议实验集中记录在
`source_tables/experiments/completed_20260810.csv`，人类可读摘要位于
`reports/evidence/RECENT_COMPLETED_EXPERIMENTS_20260810.md`。该批次统一采用
“最高 Rank-1 轮次及其同轮 mAP/mINP”作为权重保留规则。

## 当前正式默认方案

自 2026-08-08 起，`SALT_R_TEXT_VISUAL` 被提升为 SALT-VI 正式 Stage-B
默认方案。其权威配置是 `configs/stage_b/r_text_visual_20260729.yaml`，
SYSU-MM01 all-search / single-shot / 10-trial 的保留结果为 Rank-1
84.0783477%、mAP 81.4333938%、mINP 71.7898627%。

总表中该次运行的 `archived` lifecycle 保持不变，因为它表示原始实验运行已
完成并归档，不表示方案不再有效。当前默认身份及不要求多随机种子验证的人工
决策记录在 `docs/protocols/SALT_R_TEXT_VISUAL_DEFAULT.md`。

# SALT-VI 统一实验总表

experiment_registry.csv 是 SALT-VI 唯一对外使用的实验总表。它按行保留全部实验记录，并用 record_type 区分汇总结果、逐 epoch 指标、训练曲线和清单记录。

source_tables/ 是总表所依赖的原始结果表集合，按 source_core、source_baseline、experiments 等语义子目录组织；它们与总表属于同一项目，不再设置旧项目/新项目的层级。总表通过 source_table、source_row_number、source_sha256 和 extra_metrics_json 保留每条记录与原始表的对应关系。

数据集和模型权重不复制到这里；总表只记录它们的路径、哈希和状态。

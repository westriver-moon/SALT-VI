# SALT-VI 统一实验总表

`experiment_registry.csv` 是 SALT-VI 跨 Stage-A、Stage-B、消融、复现和数据派生实验的唯一总表。不要从 README、日志摘要或论文草稿建立第二份“当前排行榜”。

本次文档整合不修改该 CSV、其 source tables 或任何结构化实验结果。

## 使用规则

- 一行可以表示实验结果、单 epoch 指标、归档记录或来源记录；先查看 `record_type`。
- `lifecycle` 表达配置当前是否可运行，`status` 表达那一次运行的状态；两者不能互相替代。
- checkpoint 非空不代表文件仍存在，结合 checkpoint 状态、校验状态和哈希字段判断。
- SYSU 正式结果必须注明 all-search、single-shot、10 gallery trials；总表只记录 10-trial 聚合指标。
- RegDB 必须区分单个 numbered trial 与多 trial 均值。
- 同一保留结果的 Rank-1、mAP 和 mINP 应来自同一选择 epoch；独立最佳指标放在补充字段，不覆盖选择结果。
- 历史配置存在不代表其数据、初始化权重或 topology 仍完整；以 `lifecycle` 和复现状态为准。

## 当前项目状态

- 正式 Stage-B 默认仍为 `SALT_R_TEXT_VISUAL`，配置 `configs/stage_b/r_text_visual_20260729.yaml`。
- 当前活跃研究是 geometry-matched PASD RGB+IR 的 Stage-A 视觉底座，详见 [`../../docs/README.md`](../../docs/README.md)。
- `SALTVI-STAGEA-PASD-NOMB-B16-20260811`、RN50 Direct 与停止的 PostTrain60 均已登记；Direct 保留最佳 checkpoint，PostTrain60 仅保留配置、指标和日志。

## 维护流程

完成实验后同时登记：

1. 稳定 experiment ID、阶段、数据集与协议；
2. 配置快照和代码提交；
3. 实际命令、运行目录和日志；
4. 结构化指标及选择规则；
5. checkpoint 路径、身份、哈希和保留状态；
6. 单 seed、测试集调参、缺失资产或其他有效性边界。

原始日志、配置快照、`experiments/` 元数据和 Git 历史承担细节追溯；README 只说明总表语义，不复制表内结果。

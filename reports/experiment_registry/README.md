# SALT-VI 统一实验总表

`experiment_registry.csv` 是 SALT-VI 跨 Stage-A、Stage-B、消融、复现和数据派生实验的唯一总表。不要从 README、日志摘要或论文草稿建立第二份“当前排行榜”。

该 CSV 是唯一权威的结构化记录；历史专项报告和原始来源不删除，作为可追溯证据保留。当前批次的状态入口是 [`../EXPERIMENT_STATUS_20260822.md`](../EXPERIMENT_STATUS_20260822.md)，CSV 只保存结构化字段，不复制叙述性排行榜。

## 使用规则

- 一行可以表示实验结果、单 epoch 指标、归档记录或来源记录；先查看 `record_type`。
- `lifecycle` 表达配置当前是否可运行，`status` 表达那一次运行的状态；两者不能互相替代。
- checkpoint 非空不代表文件仍存在，结合 checkpoint 状态、校验状态和哈希字段判断。
- SYSU 正式结果必须注明 all-search、single-shot、10 gallery trials；总表只记录 10-trial 聚合指标。
- RegDB 必须区分单个 numbered trial 与多 trial 均值。
- 同一保留结果的 Rank-1、mAP 和 mINP 应来自同一选择 epoch；独立最佳指标放在补充字段，不覆盖选择结果。
- 历史配置存在不代表其数据、初始化权重或 topology 仍完整；以 `lifecycle` 和复现状态为准。

## 当前项目状态（截至 2026-08-22）

- 当前活跃对照是 `configs/stage_a/plugins/hybrid_loss/` 下的 PMT-MSCM H1–H4 及 H3-35 独立延长实验；精确结果和归档状态见当前状态总览。
- QRI 加速、眼镜、像素控制、文本标注和文本想象是独立专项证据，必须以 smoke/诊断/未启动生产的真实状态登记，不能混入 Stage-A ReID 排行榜。
- 正式 Stage-B 默认和旧路线仍为历史对照，配置 `configs/stage_b/r_text_visual_20260729.yaml`；它不是本批次活跃实验入口。

- geometry-matched PASD RN50 Stage-A 已接入 Stage-B 并完成 30 epoch 运行；该实验及其基础配置均已归档，配置位于 `configs/experiments/reproduction/archived_configs/stage_b_rn50_pasd_r_text_visual_30.yaml`。
- RN50 两阶段 40-epoch 网格已全部归档；r3 是组内最佳并保留唯一结果权重，r0/r1/r2 的非改进权重均已清理。网格配置和 runner 已移入复现归档区。
- PMT-ViT、No-MBPatch、batch 128、FlashAttention 的 24 epoch 修复版与 70 epoch 延长版均已恢复并归档；24 epoch 最佳模型保留，70 epoch 未替代它，详见唯一 CSV 与 [`../../docs/README.md`](../../docs/README.md)。
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

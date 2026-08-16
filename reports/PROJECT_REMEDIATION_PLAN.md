# SALT-VI 后续修正记录

> 核查基线：`main@c9cbc1af3b7a5d8fc80f9db02801a89c5f995d09`。
> 本文记录 2026-08-16 整改队列的完成状态与剩余证据缺口。

## 已完成

### 1. Golden evaluation 冻结入口与真实证据

`scripts/evaluation/run_golden_evaluation.py` 在 test 模式下写出结构化
`golden_evaluation.json`，包含 checkpoint SHA-256、resolved config hash、data
manifest hash、完整 `ProtocolSpec`、`eval_caption_seed` 和 Rank-1/mAP/mINP。

已用保留 checkpoint 完成：

- `r3_switch24_fullpairs`，SYSU all-search/single-shot/10 trials，
  `identity_text` IR/Fusion/Text：
  - IR: Rank-1 0.7068, mAP 0.6614, mINP 0.5122
  - Fusion: Rank-1 0.8375, mAP 0.7927, mINP 0.6682
  - Text: Rank-1 0.4514, mAP 0.4688, mINP 0.3458
- 同一 checkpoint 的 multi-shot 边界样例：Fusion Rank-1 0.9024,
  mAP 0.7543, mINP 0.2706。
- `ir_to_rgb_text_fusion84_30`，SYSU IR -> RGB+Text：Rank-1 0.5378,
  mAP 0.5389, mINP 0.4085。

证据目录：`reports/golden_evaluations/`（迁移前冻结）与
`reports/golden_evaluations_identity_text/`（迁移后冻结）。

RegDB 仍无保留 checkpoint，未编造 numbered-trial 指标；这是唯一剩余证据缺口。

### 2. Run manifest 与完整 resume 一致性校验

`run_manifest.json` 记录 run UUID、resolved config SHA、data manifest SHA、初始化
checkpoint SHA 和 `ProtocolSpec`；checkpoint schema 升级到 2，完整 resume 在加载
model/optimizer 前校验 run identity，任一不一致直接失败。

### 3. Active retrieval 协议迁移

`src/salt_vi/retrieval/legacy.py` 已迁移为
`src/salt_vi/retrieval/identity_text.py`，`NAME="identity_text"`，删除
`IS_LEGACY`。上层训练/测试改为按 `RESULT_KEYS` 遍历，不再按 legacy 名称分流。
`legacy` 仅保留为 reproduction/archived 配置的兼容别名，active 配置与默认值已
切换到 `identity_text`。

### 4. 退役 model-only / metric-boost resume

`resume_train_epoch >= 0` 与 `metric_boost_resume_epoch > 0` 在运行时显式拒绝；
model-only 加载分支已删除。49 个 model-only checkpoint 只盘点、未删除，清单见
`reports/checkpoint_inventory/model_only_checkpoints_20260816.csv`。

### 5. 配置 schema 收口

active config 增加数值类型/范围校验，`qbn_freeze_running_stats_epoch=-1` 作为
合法 sentinel 放行；argparse 默认值保持为显式 CLI 覆盖，不重写 reproduction
配置。

## 验收状态

- 服务器 `PYTHONPATH=src pytest -q src/salt_vi/tests`：75 passed。
- `git diff --check`：通过。
- 未删除任何 checkpoint 资产。

## 待办

- 恢复或重建 RegDB 保留 checkpoint 后，补一条明确 numbered trial 的 golden
  evaluation，再关闭本文档。

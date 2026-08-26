# Stage-B Ramp–EMA Causal Ablation 归档报告

> 实验 ID：`stage-b-ramp-ema-causal-ablation-20260825`。本报告对应 2026-08-26 完成的 G0–G3 Phase 1 实验；权重文件按用户明确要求不保留。

## 归档结论

- G0–G3 共 12 个 Phase 1 job 全部完成；每个 job 运行至配置的 30 epoch 上限或 12 epoch 早停窗口，实际完成记录为 12 epoch 训练预算。
- G4–G7（`G4_s0, G5_s0, G6_s0, G7_s0`）未启动，原因是预注册的 Phase-1 Rank-1 EMA 交互效应门控为负，因此不能称为“已完成”。
- 最佳主指标 job：`G3_s2`，模型 `ema_best.pth`（权重不保留），epoch 7；Rank-1=83.6603%，mAP=80.0120%，mINP=68.8018%。
- 归档目录：`/home/lab929/ybj/experiments/archive/stage_b/SALT-VI-stageb-ramp-ema-causal-ablation-20260825`。验证清单：`/home/lab929/ybj/experiments/archive/stage_b/SALT-VI-stageb-ramp-ema-causal-ablation-20260825/provenance/inventory.sha256`；验证报告：`/home/lab929/ybj/experiments/archive/stage_b/SALT-VI-stageb-ramp-ema-causal-ablation-20260825/provenance/verification.json`。

## Phase 1 实验结果

| Job | 组别 | seed | primary | 最佳 epoch | Rank-1 (%) | mAP (%) | mINP (%) | 状态 |
|---|---|---:|---|---:|---:|---:|---:|---|
| G0_s0 | G0 | 0 | online | 6 | 82.9293 | 79.6061 | 68.6203 | completed |
| G0_s1 | G0 | 1 | online | 4 | 82.6164 | 79.5261 | 68.4817 | completed |
| G0_s2 | G0 | 2 | online | 5 | 83.3605 | 79.6181 | 68.3635 | completed |
| G1_s0 | G1 | 0 | online | 4 | 82.7373 | 79.2218 | 67.9392 | completed |
| G1_s1 | G1 | 1 | online | 3 | 83.0792 | 79.4689 | 67.9767 | completed |
| G1_s2 | G1 | 2 | online | 5 | 83.4552 | 79.5660 | 68.0581 | completed |
| G2_s0 | G2 | 0 | ema | 7 | 83.3710 | 79.8693 | 68.7362 | completed |
| G2_s1 | G2 | 1 | ema | 7 | 83.3605 | 79.9877 | 68.9023 | completed |
| G2_s2 | G2 | 2 | ema | 6 | 83.4946 | 79.9843 | 68.9613 | completed |
| G3_s0 | G3 | 0 | ema | 7 | 83.4815 | 79.8963 | 68.6824 | completed |
| G3_s1 | G3 | 1 | ema | 8 | 83.2396 | 79.8077 | 68.6008 | completed |
| G3_s2 | G3 | 2 | ema | 7 | 83.6603 | 80.0120 | 68.8018 | completed |

## 交互效应

公式：`I_EMA = G3_EMA - G1_online - G2_EMA + G0_online`。

| 指标 | I_EMA 均值 | sample std | I_online 均值 |
|---|---:|---:|---:|
| Rank-1 | -0.00070117 | 0.00459616 | 0.00000000 |
| mAP | 0.00122787 | 0.00269627 | 0.00000000 |
| mINP | 0.00325565 | 0.00262904 | 0.00000000 |

Rank-1 的 EMA 交互均值为负（约 -0.0701 个百分点），所以 Phase 2 的 G4–G7 条件门控未触发；I_online 为 0 是因为 online 端在这组设计中没有启用 EMA 分支，属于设计中的对照记录。

## 设计与复现信息

- 2×2 因子：hard cross-modal loss 为 constant 1.25 vs. 5 epoch ramp；EMA 为 off vs. decay=0.999、late start=6、average k=3。
- 统一数据/评估：SYSU-MM01，RGB/IR/Text，all-search、single-gallery、10 trials；输入 512×256；Stage-A 初始化为 RN50/PASD checkpoint。
- 调度器：只允许物理 GPU 1、3，最大并发 2；本报告不把 GPU 映射误写成实验因子。
- 早停语义：是 30 epoch 训练配置下的 12 epoch 早停/完成窗口，不是把学习率计划改成“真正的 12 epoch 实验”。
- 精确设计源码位于 `provenance/source_files/`，源码快照位于 `provenance/shared_snapshot/`；非权重运行证据按清单逐文件 SHA-256 验证。
- autoresearch 外层 wrapper 有一次结束后的 bookkeeping crash，但 scheduler/state.json 为 completed，且 results.json 明确记录 12 个 Phase 1 job，故归档状态按实际 scheduler 结果判定。

## 归档内容

- 原始运行根：`/home/lab929/ybj/autoresearch-v2/runs/stage-b-ramp-ema-causal-ablation-20260825`（保留非权重证据）。
- `provenance/final_metrics.json`：results、interaction、scheduler 状态与事件摘要。
- `provenance/omitted_weights.json`：按后缀统计被明确排除的权重文件，不保存权重本体。
- `provenance/inventory.sha256`：完整非权重归档清单。

完整的逐 job JSON 结果仍以归档内的 `results.json` 为准；本报告只做便于网页端阅读的摘要。

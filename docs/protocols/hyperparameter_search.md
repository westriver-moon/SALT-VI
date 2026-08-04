# SALT-VI canonical document

This document consolidates related legacy material. All configuration, code, data and output references below have been rewritten to the SALT-VI layout.


---

## Migrated source: h1 hyperparameter search plan

> Source document ID: `source_core:docs/h1_hyperparameter_search_plan.md`  
> Original SHA-256: `99eb2a7c9b407a7b2d9d90e39faeb4754bd7720b13625e496a204c094ebe8ed8`  
> Canonical runtime: `/home/cgv841/ybj/SALT-VI/src/salt_vi/` and `/home/cgv841/ybj/SALT-VI/scripts/`  
> This section is rewritten for the SALT-VI layout; it is not an active compatibility layer.

# TRAIN-3-H1 最终超参数搜索长期指导

## 1. 目标与固定协议

本计划以 `TRAIN-3-H1` 为当前最佳单模型配置，分三阶段完成局部超参数搜索。训练仍从 E4 epoch-9 checkpoint warm-start，最终选择基于 SYSU-MM01 all-search、single-shot、10 gallery trials 的 Rank-1、mAP 和 mINP。

以下设置在全部实验中保持固定：

| 项目 | 固定值 |
|---|---|
| 基线配置 | `logs/raw/source_core/reports/metric_boost/runs/TRAIN-3-H1/runtime_config.yaml` |
| Warm-start | E4 epoch-9 checkpoint |
| 训练模式 | `RGB_IR_Text` |
| 融合方式 | `parameter_add` |
| 视觉骨干 | `Fix_Visual=true` |
| 损失族 | `id,cross_modal_hard` |
| 输入尺寸 | 288×144 |
| 协议 | SYSU all-search、single-shot、10 gallery trials |
| 后处理 | 无 TTA、无多尺度、无重排、无模型/种子集成 |
| 初筛 seed | 0 |
| 评测频率 | 每个 epoch |
| checkpoint 选择 | 最高 Rank-1；mAP、mINP 必须取同一 epoch |

## 2. 第一阶段：单因素局部粗筛

每次只改变一个参数，其余参数保持 H1 基准。四组共享同一个 `HPT-C0`，因此共有 12 个唯一实验。

| 组别 | 参数 | 搜索值 | H1 基准值 | 唯一实验数 |
|---|---|---|---:|---:|
| W | `cross_modal_hard_weight` | 0.50、0.75、1.00、1.25、1.50 | 1.00 | 5 |
| L | `lr_txt` | 2.5e-6、5e-6、7.5e-6、1e-5 | 1e-5 | 4 |
| P | `pa` | 0.45、0.50、0.55 | 0.50 | 3 |
| B | BN 策略 | shared BN、QBN、QBN + epoch 5 冻结统计量 | shared BN | 3 |

| 实验 ID | 相对 H1 的唯一改动 |
|---|---|
| `HPT-C0` | 无，精确复现 H1 |
| `HPT-W050` | `cross_modal_hard_weight=0.50` |
| `HPT-W075` | `cross_modal_hard_weight=0.75` |
| `HPT-W125` | `cross_modal_hard_weight=1.25` |
| `HPT-W150` | `cross_modal_hard_weight=1.50` |
| `HPT-L025` | `lr_txt=2.5e-6` |
| `HPT-L050` | `lr_txt=5e-6` |
| `HPT-L075` | `lr_txt=7.5e-6` |
| `HPT-PA045` | `pa=0.45` |
| `HPT-PA055` | `pa=0.55` |
| `HPT-QBN` | `uni_BN=true` |
| `HPT-QBN-F5` | `uni_BN=true` 且 `qbn_freeze_running_stats_epoch=5` |

第一阶段只负责局部筛选，不把不同因素组合在同一实验中。调度器检测物理 GPU 的显存、利用率和计算进程；空闲一张即启动一个实验，空闲多张即并行启动相同数量，且每张卡最多一个训练进程。

## 3. 第二阶段：学习率 × hard-loss 权重局部网格

从第一阶段选出的最佳 `lr_txt` 和 `cross_modal_hard_weight` 各取“最佳值及两个相邻值”，形成 3×3 网格。`pa` 与 BN 策略固定为第一阶段胜出项。

若第一阶段支持 `lr_txt=5e-6`、`cross_modal_hard_weight=1.0`，则实例化为：

| `lr_txt` \ hard weight | 0.75 | 1.00 | 1.25 |
|---|---:|---:|---:|
| 2.5e-6 | `L025-W075` | `L025-W100` | `L025-W125` |
| 5e-6 | `L050-W075` | `L050-W100` | `L050-W125` |
| 7.5e-6 | `L075-W075` | `L075-W100` | `L075-W125` |

与第一阶段完全相同的参数组合必须复用已有结果，不重复训练。若最佳值位于粗筛边界，则只向仍有实验覆盖的一侧取邻值，不盲目外推。

## 4. 第三阶段：跨模态 pair 权重结构搜索

六种 pair 不独立做笛卡尔积，而使用四个具有明确几何假设的结构。学习率、hard-loss 总权重、`pa` 和 BN 均固定为前两阶段胜出项。

| 实验 | RGB-IR | RGB-Fusion | IR-Fusion | RGB-Text | IR-Text | Fusion-Text |
|---|---:|---:|---:|---:|---:|---:|
| `PAIR-EQUAL` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `PAIR-MILD` | 0.75 | 1.00 | 0.50 | 0.25 | 0.25 | 0.25 |
| `PAIR-STRONG` | 0.50 | 1.00 | 0.25 | 0.10 | 0.10 | 0.10 |
| `PAIR-NOTEXT` | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |

## 5. 最终复现与选择规则

| 候选配置 | seed 0 | seed 1 | seed 42 |
|---|---:|---:|---:|
| 最佳组合 A | 运行 | 运行 | 运行 |
| 最佳组合 B | 运行 | 运行 | 运行 |
| 最佳组合 C | 运行 | 运行 | 运行 |

最终按三种子均值选配置，不选择单个幸运 seed。内部排序分数为：

```text
S = 0.5 × Rank-1 + 0.4 × mAP + 0.1 × mINP
```

若综合分差小于 0.1 个百分点，优先选择 mAP 更高且跨种子方差更小的配置。相对 H1，任一主指标下降超过 0.3 个百分点的配置不得仅凭另一指标的小幅增益进入下一阶段。

## 6. 运行与记录要求

- 每次正式训练必须在进程启动前生成不可变 `manifest.json`。
- 每个 run 必须保存 `design.md`、resolved runtime config、config diff、exact source state/code patch、环境指纹、数据集与 warm-start 指纹、command、artifact hashes、`events.jsonl`、日志和原子更新的状态文件。
- 重试使用新的 attempt 目录或新的 experiment ID，不覆盖既有 provenance bundle。
- 调度器可持续等待 GPU；“当前无空闲卡”不是失败条件。
- 不得与已有实验共享同一个物理 GPU；GPU 是否空闲必须同时检查显存、利用率和 compute process。

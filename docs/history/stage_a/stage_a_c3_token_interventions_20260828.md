# C3 token 干预实验归档（2026-08-27/28）

本页归档 rounded-rectangle 固定 token pruning 与 Pose-guided Counterfactual
Token Intervention（CTI）。两者均保留原 C3 ID、triplet、MSEL、DCL 目标；前者
在单一路径中固定裁掉圆角矩形外 token，后者在训练期增加共享后缀参数的反事实分支。
推理始终使用普通完整路径。

## 方法与固定条件

- 输入最终均为 512×256，PMT PatchEmbed 为 16×16、stride 12×12，完整网格
  42×21（882 个 patch token）。
- Rounded pruning 分别删除 10%、15%、20%，保留 token 原顺序和原位置编码。
- CTI 在第 9 层后，每次从 pose/anatomy 人体候选中无放回随机删除固定 32 个
  token；删除候选不足的样本仍参加原 C3 loss，只不参加 CTI loss。
- CTI 使用 true-ID logit 相对其他类别 log-sum-exp 的 evidence drop margin；
  `margin=0.5`、最大权重 0.2、epoch 6 启动、3 epoch 线性 warm-up。
- 所有运行均为 seed 0、24 epochs。这里固定报告最终 epoch 23，而非用测试集挑 epoch。
- SYSU 协议为 all-search、single-shot、红外 query 到可见光 gallery、10 gallery trials，
  无 reranking/TTA。

## 两种输入数据

`person_fit` 使用审核后的 V1 + YOLO26 person bbox；714 个 SYSU fallback 保留完整
画面。`resize` 使用同一批 native realSR SwinIR-M ×4 输出，直接 bicubic 到
512×256，不做人框裁剪。两套均覆盖完整 44,745 个 SYSU source key，并各自有
完整 pose 与 anatomy sidecar。

它们都不同于旧 C3 的 classical SwinIR-M ×2 数据。因此下面结果能比较当前 ×4
管线内的裁剪/干预差异，不能单独归因于相对旧 C3 的裁剪状态。

## 最终指标

| 输入 | 干预 | Rank-1 | mAP | mINP |
| --- | --- | ---: | ---: | ---: |
| person-fit | rounded pruning 10% | 67.8701% | 66.1821% | 54.5619% |
| person-fit | rounded pruning 15% | 66.6763% | 64.9668% | 53.2465% |
| person-fit | rounded pruning 20% | 66.3660% | 64.7354% | 52.9775% |
| resize | rounded pruning 10% | 67.5940% | 66.3765% | 54.8531% |
| person-fit | pose-guided CTI, K=32 | 66.7499% | 64.6963% | 52.1852% |
| resize | pose-guided CTI, K=32 | 67.9963% | 66.1729% | 54.2364% |

同一当前 resize 输入上，CTI 比 10% pruning 高 0.40 个 Rank-1 百分点，但 mAP
低 0.20、mINP 低 0.62，整体视为接近而非明确胜出。CTI 的 person-fit→resize
变化为 Rank-1 +1.25、mAP +1.48、mINP +2.05，说明当前 person-fit 几何对 CTI
不利。Rounded pruning 从 10% 增至 15%/20% 后连续下降，10% 是本轮较合理的效率点。

旧 classical SwinIR ×2 的 ellipse-C3 对照最终约为 70.4–70.6 Rank-1、68.8–68.9
mAP、57.5 mINP；当前六个实验没有超过它。但该差距同时包含 ×2/×4 模型、SR
顺序、IR 处理和几何管线变化，不能仅解释为 CTI 或 pruning 失败，也不能靠事后
超参搜索宣称消除。后续严格消融应先在同一旧 ×2 输入上重跑原 C3、10% pruning
和 CTI，再调整 intervention layer、K、margin、weight 与 start epoch。

## 主线代码与配置

- Rounded pruning：`src/salt_vi/models/vision_adapter.py`、
  `src/salt_vi/models/vision_transformer.py`、`src/salt_vi/tests/test_token_pruning.py`；
- CTI：`src/salt_vi/data/pose_tokens.py`、`src/salt_vi/training/recipes.py`、
  `src/salt_vi/tests/test_pose_guided_cti.py`；
- 可运行配置：`configs/stage_a/person_assets/sysu_{person_fit,resize}_cti.yaml` 与
  `sysu_{person_fit,resize}_rounded_rect_10.yaml`；15%/20% person-fit 配置也保留。

禁用 CTI/pruning 时保持原 C3 路径。CTI 与固定 pruning、ellipse attention、
quadruple input 不允许同时启用，配置校验会直接拒绝。

## 归档与复核证据

- Rounded pruning：
  `/home/lab929/ybj/experiments/archive/stage_a/SALT-VI-c3-rounded-token-pruning-20260827`
  （34 个文件通过 SHA-256 清单校验）；
- Pose-guided CTI：
  `/home/lab929/ybj/experiments/archive/stage_a/SALT-VI-pose-guided-cti-20260828`
  （18 个文件通过 SHA-256 清单校验）；
- 每个变体保留最终 `model_IR_23.pth`、可恢复的 `checkpoint_latest.pth`、
  `configs.yaml`、`run_manifest.json`、事件流与原始日志；
- 结构化指标写入 `reports/experiment_registry/experiment_registry.csv`，资产位置与
  哈希写入 `reports/assets/person_assets_20260829.json`。

本批次均为单 seed，结果用于工程消融，不作为统计显著性结论。

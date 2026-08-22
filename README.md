# SALT-VI

SALT-VI 是可见光—红外行人重识别研究仓库。当前实现包含两阶段训练、离线 PASD 视觉生成、RGB 文本监督，以及面向模糊观测的加权语义想象接口。

## 当前状态

- 2026-08-21/22 的当前实验主线是 SYSU-MM01 PMT-MSCM 混合损失 H1–H4；H1–H4 已完成 24 epoch，H3 QCT-0.40 的 35 epoch 独立延长实验仍以服务器状态和完成门禁为准。
- H1–H4 的最高/最终结果、协议、配置和归档状态统一见 [`reports/EXPERIMENT_STATUS_20260822.md`](reports/EXPERIMENT_STATUS_20260822.md)；不要从旧报告中的“当前状态”句子建立排行榜。
- QRI 加速、眼镜验证、像素控制、文本标注和文本想象均保留为独立专项证据；smoke/负面结果不等同于全数据集生产或 ReID 训练结果。
- RN50、PMT-ViT/No-MBPatch、safe-trick、PASD 和 Stage-B 结果是历史已归档路线，仍可由实验总表和归档目录追溯，不能误写成当前运行任务。
- 实验指标与 checkpoint 身份的唯一总表：`reports/experiment_registry/experiment_registry.csv`。

项目架构、数据契约、当前实验、运行命令和结果解释统一见 [`docs/README.md`](docs/README.md)。
- 本轮架构审计与清理记录见 docs/ARCHITECTURE_AND_RESTRUCTURE.md。
- 本轮可复现归档规范和双账号资产核验记录见 `reports/archive_plan_20260822.md`。

## 主要入口

```bash
python -m pip install -e ".[test]"
python -m pytest src/salt_vi/tests
PYTHONPATH=. python -m pytest -q pasd_plugin/tests
python scripts/train.py --config_select <config.yaml>
```

- `src/salt_vi/`：训练、模型、数据和评估的唯一实现。
- `configs/stage_a/`、`configs/stage_b/`、`configs/super_resolution/`：当前可运行配置；各目录中的 `reproduction/` 与 `configs/experiments/reproduction/` 只保存已运行实验快照。
- `pasd_plugin/`：统一 PASD 数据生成、验证插件（SYSU-MM01、RegDB、LLCM），见 [`pasd_plugin/README.md`](pasd_plugin/README.md)。
- `semantic_imagination/`：加权语义假设插件，见 [`semantic_imagination/README.md`](semantic_imagination/README.md)。
- `plugins/qwen_imagination/`：集中管理 QRI v1/v2 及后续 Qwen 想象力插件；SALT 主线通过 `src/salt_vi/imagination.py` 的统一接口调用。
- `feature_analysis/`：特征分析工具，见 [`feature_analysis/README.md`](feature_analysis/README.md)。
- `reports/experiment_registry/`：实验总表及其字段说明。

数据集、派生图像、预训练权重、checkpoint 和原始日志是服务器本地资产，不随源码发布。

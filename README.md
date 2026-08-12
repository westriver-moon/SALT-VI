# SALT-VI

SALT-VI 是可见光—红外行人重识别研究仓库。当前实现包含两阶段训练、离线 PASD 视觉生成、RGB 文本监督，以及面向模糊观测的加权语义想象接口。

## 当前状态

- 当前研究主线：在 SYSU-MM01 上使用 geometry-matched PASD RGB 与 IR 输入重建 Stage-A 视觉底座。
- 当前对照：RN50 从头直接训练（Direct）与官方 SYSU 权重低学习率后训练（PostTrain60）。
- 正式 Stage-B 默认：`configs/stage_b/r_text_visual_20260729.yaml`（`SALT_R_TEXT_VISUAL`）。它是已晋升的历史默认，不等于当前正在运行的 Stage-A 研究线。
- 实验指标与 checkpoint 身份的唯一总表：`reports/experiment_registry/experiment_registry.csv`。

项目架构、数据契约、当前实验、运行命令和结果解释统一见 [`docs/README.md`](docs/README.md)。

## 主要入口

```bash
python -m pip install -e ".[test]"
python -m pytest src/salt_vi/tests
PYTHONPATH=pasd_offline python -m pytest pasd_offline/tests
python scripts/train.py --config_select <config.yaml>
```

- `src/salt_vi/`：训练、模型、数据和评估的唯一实现。
- `configs/`：活跃配置与复现实验快照。
- `pasd_offline/`：独立 PASD 数据生成器，见 [`pasd_offline/README.md`](pasd_offline/README.md)。
- `semantic_imagination/`：加权语义假设插件，见 [`semantic_imagination/README.md`](semantic_imagination/README.md)。
- `feature_analysis/`：特征分析工具，见 [`feature_analysis/README.md`](feature_analysis/README.md)。
- `reports/experiment_registry/`：实验总表及其字段说明。

数据集、派生图像、预训练权重、checkpoint 和原始日志是服务器本地资产，不随源码发布。

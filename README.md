# SALT-VI

SALT-VI 是可见光—红外行人重识别研究仓库。当前实现包含两阶段训练、离线 PASD 视觉生成、RGB 文本监督，以及面向模糊观测的加权语义想象接口。

## 当前状态

- 当前研究主线：在 SYSU-MM01 上用 geometry-matched PASD RGB 与 IR 重建 Stage-A，并把保留的 RN50 Stage-A 初始化接入 Stage-B；首个 30 epoch 衔接实验已完成并登记。
- PMT-ViT、No-MBPatch、batch 128、FlashAttention 路线：24 epoch 修复版已恢复结构化结果并保留最佳模型；70 epoch 延长版已完成并归档，但未替代该结果。
- 已完成对照：RN50 Direct 完成 120 epoch；PostTrain60 在 epoch 28 停止并归档为失败。
- 当前 Stage-B 主线使用 PASD-RN50 衔接方案；已完成运行的精确配置快照归档于 `configs/experiments/reproduction/archived_configs/`。`configs/stage_b/r_text_visual_20260729.yaml`（`SALT_R_TEXT_VISUAL`）仅保留为历史对照，不再作为默认入口。
- 实验指标与 checkpoint 身份的唯一总表：`reports/experiment_registry/experiment_registry.csv`。

项目架构、数据契约、当前实验、运行命令和结果解释统一见 [`docs/README.md`](docs/README.md)。

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
- `feature_analysis/`：特征分析工具，见 [`feature_analysis/README.md`](feature_analysis/README.md)。
- `reports/experiment_registry/`：实验总表及其字段说明。

数据集、派生图像、预训练权重、checkpoint 和原始日志是服务器本地资产，不随源码发布。

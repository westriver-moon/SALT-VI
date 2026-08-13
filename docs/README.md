# SALT-VI 统一项目指南

本文档是 SALT-VI 当前项目状态、架构、数据契约、训练入口和实验解释的唯一综合说明。历史决策和逐实验叙述不再维护为独立文档；需要追溯时使用 Git 历史、配置快照、原始日志和实验总表。

## 1. 当前研究主线

当前工作的核心不是继续堆叠 Stage-B 文本融合模块，而是先修复其上游视觉域：PASD 生成后的 RGB 与原始 IR 在分辨率、人物尺度和背景布局上不一致，会把预处理偏差混入跨模态差异。

当前数据集为：

```text
/home/lab929/datasets/derived/SYSU-MM01-pasd-rgb-ir-geomatched-512x256-1view-v1
```

数据契约：

- 共 44,745 个源图像视图：RGB 29,033，IR 15,712。
- RGB 使用单视图 PASD 输出；IR 不做语义生成。
- 两种模态均保持人物比例，以相同的 blurred-background 几何方式适配到 256×512。
- `manifest.jsonl`、`manifest.json`、`build.json` 和 `validation-report.json` 共同固定来源、大小、校验和与完整性；当前校验错误数为 0。
- 训练配置使用 `sysu_sr_backend: pasd_multiview`、`sysu_sr_modalities: [rgb, ir]`、`sysu_sr_view_sampling: paired` 和 `sysu_sr_exact_size: true`。

已完成两种 RN50 Stage-A 适配路线的工程比较：

| 路线 | 配置 | 初始化与调度 | 归档最佳 SYSU 结果 |
| --- | --- | --- | --- |
| Direct | `configs/stage_a/reproduction/source_core/stage_a_tvilfm_rn50_pasd_rgb_ir_geomatched_512x256_direct.yaml` | ImageNet RN50 初始化；120 epoch，batch 32 | Rank-1 71.5698%，mAP 67.8598%，mINP 53.9621%（epoch 115；完成 120 epoch） |
| PostTrain60 | `configs/stage_a/reproduction/source_core/stage_a_tvilfm_rn50_pasd_rgb_ir_geomatched_512x256_posttrain_60ep.yaml` | `pretrained/tvi_lfm/sysu/VI_sysu_BASE.pth`；60 epoch，batch 16，低学习率 | Rank-1 66.6448%，mAP 61.9430%，mINP 46.6493%（epoch 19；在 epoch 28 停止） |

Direct 已完成并保留 epoch 115 最佳 checkpoint；PostTrain60 已停止、删除实验权重并按失败归档。由于两条路线的初始化、batch size、学习率和调度同时不同，它们是工程路线比较，不是只隔离“是否 warm start”的严格单因素消融。

已完成的 PMT-ViT、No-MBPatch、geometry-matched PASD 单视图 Stage-A 结果为 Rank-1 67.9174%、mAP 64.9257%、mINP 51.3045%（epoch 23）。对应实验 `SALTVI-STAGEA-PASD-NOMB-B16-20260811` 已归档到总表。

## 2. 正式默认与活跃研究的区别

正式 Stage-B 默认仍是：

```text
configs/stage_b/r_text_visual_20260729.yaml
```

启动命令：

```bash
python scripts/train.py --config_select configs/stage_b/r_text_visual_20260729.yaml
```

`SALT_R_TEXT_VISUAL` 冻结视觉分支，使用离线 RGB+IR SwinIR x2 输入、RGB-IR/RGB-Text/IR-Text direct pair loss、双分支 patch embedding 和 learnable PatchGeM，不启用 LLM caption augmentation。其保留结果来自 SYSU-MM01 all-search、single-shot、10 gallery trials、单 seed：Rank-1 84.0783%、mAP 81.4334%、mINP 71.7899（epoch 23）。

它是结果引用和 Stage-B 复现的正式默认；当前 geometry-matched Stage-A 工作是在寻找更好的上游视觉初始化，尚未替代该默认。

## 3. 系统架构

```text
原始 SYSU 图像与 captions
        │
        ├─ pasd_offline/ ──> PASD RGB / geometry-only IR manifests
        │
        ├─ semantic_imagination/ ──> 动态加权语义假设（尚未进入活跃配置）
        │
        ▼
Stage A：RGB/IR image-only 视觉表征
        │
        ▼
Stage B：冻结或受控视觉分支的 RGB/IR/Text 融合
        │
        ▼
SYSU/RegDB/LLCM 评估、checkpoint、日志与实验总表
```

代码边界：

- `src/salt_vi/data/`：公开数据索引、PASD manifest 读取、视图加权采样和训练数据适配。
- `src/salt_vi/models/`：RN50、PMT-ViT、CLIP 视觉/文本分支与 pooling。
- `src/salt_vi/engine/`：模型构建、训练和测试。
- `src/salt_vi/training/`：活跃训练 recipe。
- `src/salt_vi/retrieval/`：活跃检索后端；未登记别名不作为入口。
- `src/salt_vi/entrypoints/train.py`：唯一训练入口实现；外部使用 `scripts/train.py`。
- `pasd_offline/` 与 `semantic_imagination/` 是离线包，不导入训练包。

## 4. Semantic Imagination 的当前边界

该插件把一个模糊观测转换为若干语义等价簇：VLM 进行多次扰动采样，文本嵌入聚类，簇内 medoid 作为代表，簇频率作为 `hypothesis_weight`。权重会原样进入 PASD manifest，并由 SALT sampler 加权采样。

当前代码已支持 `sysu_sr_views_per_image: 0` 的动态视图数，但所有活跃训练 YAML 仍使用单视图 `1`；当前 geometry-matched 数据的权重均为 1。因此 Semantic Imagination 是下一阶段接口，不是当前运行实验的自变量。数学语义和不变量见 [`../semantic_imagination/MATHEMATICAL_SPEC.md`](../semantic_imagination/MATHEMATICAL_SPEC.md)。

## 5. 仓库与资产位置

| 路径 | 作用 |
| --- | --- |
| `src/salt_vi/` | 当前实现 |
| `configs/stage_a/`、`configs/stage_b/` | 活跃阶段配置 |
| `configs/experiments/reproduction/` | 历史运行配置快照；不能因存在而视为可运行 |
| `scripts/` | 训练、验证、分析、归档入口 |
| `pasd_offline/` | PASD records、生成、验证和 geometry-matched 构建 |
| `semantic_imagination/` | 离线语义假设与 PASD record 导出 |
| `feature_analysis/` | 特征提取和分析 |
| `experiments/` | 运行元数据与归档材料 |
| `reports/experiment_registry/experiment_registry.csv` | 唯一实验总表 |
| `checkpoints/`、`pretrained/`、`logs/`、`runtime/` | 服务器本地运行资产 |
| `vendor/` | 上游来源和许可证边界 |

公共原始数据根为 `/home/cgv841/datasets/`；当前 PASD 派生数据位于 `/home/lab929/datasets/derived/`。配置必须显式记录实际绝对路径，不把数据复制进仓库。

## 6. 安装、检查与运行

```bash
python -m pip install -e ".[test]"
python -m pytest src/salt_vi/tests
PYTHONPATH=pasd_offline python -m pytest pasd_offline/tests
python -m pytest semantic_imagination/tests
```

训练统一使用显式 YAML：

```bash
python scripts/train.py --config_select <config.yaml>
```

运行前至少核对：数据根、派生数据 manifest、`training_weight_init`、输出目录、GPU 映射、图像尺寸、backbone topology 和评估 trial 数。`DataParallel` 不作为通用入口；冻结视觉分支时只使用已验证的 `fixed_visual_data_parallel`，其余情况一进程一 GPU。

## 7. 评估与结果解释

- SYSU 正式结果使用 all-search、single-shot、10 gallery trials 的聚合指标。
- 不把一次 run/split ID 当作 gallery trial。
- RegDB 结果必须注明单个 numbered trial 或多 trial 均值。
- 最佳 checkpoint 的 Rank-1、mAP、mINP必须来自同一 epoch；若另报独立最佳指标，必须明确标注未被选择。
- stdout 日志用于诊断；正式归档指标以结构化结果文件和总表为准。
- 测试集调参、单 seed 和不完全可复现实验必须在总表状态或备注中明确表达。

## 8. 实验总表与证据

`reports/experiment_registry/experiment_registry.csv` 是唯一跨阶段总表，不从 README 复制出第二份排行榜。每次完成实验时，保存配置快照、代码提交、运行命令、指标文件、日志、checkpoint 路径和校验值，再更新总表。

历史 Markdown 已删除，因为它们包含重复结果、失效路径和过期“当前状态”。历史事实仍可从以下位置恢复：

1. Git 历史；
2. `reports/experiment_registry/experiment_registry.csv` 及 source tables；
3. `configs/experiments/reproduction/`；
4. `experiments/`、`logs/raw/` 和结构化 runtime manifests。

这些原始实验产物不是当前运行说明，不应重新链接成平行文档体系。

## 9. 文档规则

当前人工维护的项目文档只有：

- `/README.md`：项目入口和状态摘要；
- `/docs/README.md`：本统一指南；
- `/pasd_offline/README.md`：PASD 独立模块；
- `/semantic_imagination/README.md` 与 `MATHEMATICAL_SPEC.md`：语义想象接口和数学规范；
- `/feature_analysis/README.md`：特征分析模块；
- `/reports/experiment_registry/README.md`：总表字段和维护边界；
- vendor/source 与 checkpoint 放置说明：第三方和运行资产边界。

新的运行过程不要再建立独立的“当前状态”“修复报告”“结果汇总”Markdown；将事实写入配置、结构化结果、总表和 Git 提交。

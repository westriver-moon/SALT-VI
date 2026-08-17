# SALT-VI 统一项目指南

本文档是 SALT-VI 当前项目状态、架构、数据契约、训练入口和实验解释的唯一综合说明。历史决策和逐实验叙述不再维护为独立文档；需要追溯时使用 Git 历史、配置快照、原始日志和实验总表。

## 1. 当前研究主线

统一 PASD 插件负责生成跨模态的派生图像，而不改变既有训练加载器。它覆盖 SYSU-MM01、RegDB 与 LLCM，显式保留各自官方索引和评估归属，避免把评估协议混入训练数据。

当前数据集为：

```text
/home/lab929/datasets/derived/PASD-v2/<dataset>-rgb-ir-512x256-1view
```

数据契约：

- 覆盖 SYSU-MM01、RegDB、LLCM 的官方训练与评估索引；RegDB 每张源图只生成一次，并记录全部 10 个 trial 归属。
- RGB 与 IR/NIR 均使用单视图 PASD；IR/NIR 输出在生成后转换为三通道灰度图。
- 两种模态均以保持比例的 `fit` 几何适配到 256×512，空白区域使用同源 cover-crop 模糊背景填充。
- `records.jsonl`、`manifest.jsonl`、`manifest.json`、`build.json` 和 `validation-report.json` 固定来源、大小、校验和、几何参数和协议归属。
- 本轮只生成与验证派生产物；训练加载器保持不变，后续接入必须只消费 `train` 归属。

RN50 Direct 已完成并保留最佳 checkpoint；PostTrain60 已停止、删除实验权重并按失败归档。两条路线同时改变初始化、batch size、学习率和调度，因此只是工程路线比较，不是严格单因素消融。PMT-ViT、No-MBPatch 的已完成结果也已登记；精确指标、选择 epoch、配置和 checkpoint 身份只查实验总表。

保留的 RN50 Direct Stage-A 初始化已经接入 geometry-matched PASD Stage-B，30 epoch 运行已完成，配置归档于 `configs/experiments/reproduction/archived_configs/stage_b_rn50_pasd_r_text_visual_30.yaml`。后续两阶段网格表明，24 epoch ID+WRT 对齐后接 16 epoch all-pairs 跨模态细化的 r3 是组内最佳并保留唯一结果权重。PMT-ViT、No-MBPatch、batch 128、FlashAttention 的 24 epoch 修复版与 70 epoch 延长版均已从 TensorBoard 同 step 指标恢复；24 epoch 最佳模型已保留，70 epoch 版本未替代它。精确指标和证据路径只记录在实验总表。

## 2. 正式默认与活跃研究的区别

正式 Stage-B 默认仍是：

```text
configs/stage_b/r_text_visual_20260729.yaml
```

启动命令：

```bash
python scripts/train.py --config_select configs/stage_b/r_text_visual_20260729.yaml
```

`SALT_R_TEXT_VISUAL` 冻结视觉分支，使用离线 RGB+IR SwinIR x2 输入、RGB-IR/RGB-Text/IR-Text direct pair loss、双分支 patch embedding 和 learnable PatchGeM，不启用 LLM caption augmentation。其保留结果来自 SYSU-MM01 all-search、single-shot、10 gallery trials、单 seed；精确指标和选择 epoch 只查实验总表。

它只用于历史结果引用。当前 Stage-B 主线是 PASD-RN50 衔接方案；可运行候选配置位于 `configs/stage_b/` 根目录，已完成运行的精确快照位于 `configs/experiments/reproduction/archived_configs/`，两者的结果统一查实验总表。

## 3. 系统架构

```text
原始 SYSU 图像与 captions
        │
        ├─ pasd_plugin/ ──> SYSU/RegDB/LLCM 的 PASD RGB+IR/NIR manifests
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
- `pasd_plugin/` 与 `semantic_imagination/` 是离线包，不导入训练包。

## 4. Semantic Imagination 的当前边界

该插件把一个模糊观测转换为若干语义等价簇：VLM 进行多次扰动采样，文本嵌入聚类，簇内 medoid 作为代表，簇频率作为 `hypothesis_weight`。权重会原样进入 PASD manifest，并由 SALT sampler 加权采样。

当前代码已支持 `sysu_sr_views_per_image: 0` 的动态视图数，但所有活跃训练 YAML 仍使用单视图 `1`；当前 geometry-matched 数据的权重均为 1。因此 Semantic Imagination 是下一阶段接口，不是当前运行实验的自变量。数学语义和不变量见 [`../semantic_imagination/MATHEMATICAL_SPEC.md`](../semantic_imagination/MATHEMATICAL_SPEC.md)。

## 5. 仓库与资产位置

| 路径 | 作用 |
| --- | --- |
| `src/salt_vi/` | 当前实现 |
| `configs/stage_a/`、`configs/stage_b/`、`configs/super_resolution/` | 当前可运行配置；配置根目录是主线入口 |
| 配置目录内的 `reproduction/`、`configs/experiments/reproduction/` | 历史运行的解析配置与证据快照；不作为新实验入口 |
| `scripts/train.py` | SALT-VI 两阶段训练入口 |
| `scripts/vision_text/super_resolution/` | 当前独立视觉超分消融的预检与启动入口 |
| `pasd_plugin/` | 统一 PASD records、生成、验证与免拉伸几何构建 |
| `semantic_imagination/` | 离线语义假设与 PASD record 导出 |
| `feature_analysis/` | 特征提取和分析 |
| `experiments/` | 运行元数据、归档材料及已完成实验的一次性 `source/` |
| `reports/experiment_registry/experiment_registry.csv` | 唯一实验总表 |
| `checkpoints/`、`pretrained/`、`logs/`、`runtime/` | 服务器本地运行资产 |
| `vendor/` | 上游来源和许可证边界 |

公共原始数据根为 `/home/cgv841/datasets/`；当前 PASD 派生数据位于 `/home/lab929/datasets/derived/`。配置必须显式记录实际绝对路径，不把数据复制进仓库。

## 6. 安装、检查与运行

```bash
python -m pip install -e ".[test]"
python -m pytest src/salt_vi/tests
PYTHONPATH=. python -m pytest -q pasd_plugin/tests
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
- `/pasd_plugin/README.md`：统一 PASD 插件接口；
- `/semantic_imagination/README.md` 与 `MATHEMATICAL_SPEC.md`：语义想象接口和数学规范；
- `/feature_analysis/README.md`：特征分析模块；
- `/reports/experiment_registry/README.md`：总表字段和维护边界；
- vendor/source 与 checkpoint 放置说明：第三方和运行资产边界。

新的运行过程不要再建立独立的“当前状态”“修复报告”“结果汇总”Markdown；将事实写入配置、结构化结果、总表和 Git 提交，历史过程只留在 Git。

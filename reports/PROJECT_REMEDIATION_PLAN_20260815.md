# SALT-VI 实验工程审查与修正计划（2026-08-15）

> 基线：`main@c8580d6`。本文只记录工程问题、修正边界和后续顺序；指标仍以 `reports/experiment_registry/experiment_registry.csv` 为唯一事实来源。

## 当前主线

当前主线不是继续叠加文本模块，而是先消除上游视觉域偏差，再稳定 Stage-B：

1. 用 geometry-matched PASD RGB 与对应 IR 固定 SYSU Stage-A 数据契约；
2. 保留 RN50 Direct Stage-A 最佳初始化，并接入历史 `R_TEXT_VISUAL` Stage-B；
3. 先做 ID+WRT 对齐，再做跨模态细化；当前网格中 24+16 epoch 的 all-pairs r3 为组内最佳；
4. PMT-ViT/No-MBPatch/batch128/FlashAttention 是归档对照，没有替代 RN50 主线；
5. Semantic Imagination 仍是下一阶段接口，本轮不并入正式训练。

## 问题核对

| 问题 | 判断 | 本轮处理 |
| --- | --- | --- |
| P1-1 协议写死 | 属实 | 统一 `ProtocolSpec`；事件记录标识及结构化字段。 |
| P1-2 `gallery_trials` 无效 | 属实 | SYSU/LLCM 的样本、loader、聚合共用同一字段；非正数拒绝。 |
| P1-3 gallery 污染 RNG | 属实 | 主实现和 baseline 改为 trial 局部 RNG；身份 caption 使用 seed 绑定的局部 RNG。 |
| P1-4 六个 Shell 入口错误 | 属实 | 从正式 `scripts/` 删除，历史可由 Git 恢复。 |
| P1-5 归档 runner 断路 | 属实 | 修正 `parents[3]`、runner 的 `archived_configs` 路径，以及 active TVI-LFM 配置残留的旧 `extends`。 |
| P1-6 输出目录混用 | 属实，部分完成 | fresh train 拒绝非空目录；run UUID/manifest/config hash 留待下一阶段。 |
| P1-7 依赖与 checkpoint | 属实，部分完成 | CI 固定 Torch 2.2.1/TorchVision 0.17.1/NumPy 1.26.4；显式非重入 checkpoint 并兼容旧 API。 |
| sampler 旧实现、旧 scheduler、永真条件 | 属实 | 已删除或简化，不改变算法边界。 |
| baseline/monorepo CI 边界 | 属实，部分完成 | CI 加入 baseline 和 semantic tests；wheel 排除 tests；目录迁移延后。 |
| 配置多事实来源 | 属实，未完成 | 本轮只修 `gallery_trials`；严格 schema 放到第二阶段。 |
| train/data 职责过载 | 属实，未重构 | golden evaluation 冻结前不拆模型、loss、optimizer。 |
| README 旧路径 | 属实 | 已更新为真实归档路径并补充 r3 主线定位。 |

没有发现要求整体作废既有标准 SYSU all-search、single-shot、10-trial 结果的 P0 证据。multi-shot、非 10 trial、目录复用、协议不清或旧 Shell 启动的结果，应使用保留 checkpoint 重新评估。

## 本轮边界

已修改：协议元数据、trial、随机性、输出目录入口、归档 runner、CI/打包、死代码和文档。

未修改：模型结构、loss 数学定义、optimizer、checkpoint、数据集、原始日志、指标和总表历史事实。

## 后续顺序

### A. 完成运行身份闭环

1. fresh run 创建不可覆盖的 `run_manifest.json`；
2. 记录 run UUID、commit、配置哈希、seed、协议、数据 manifest 和初始化 checkpoint；
3. checkpoint、事件和 metrics 使用同一 UUID；
4. resume 校验 manifest/config 后才允许追加日志；
5. 用论文候选 checkpoint 做冻结式 golden evaluation，不重新训练。

### B. 收敛配置事实来源

1. 建立严格 schema，默认值只保留一处；
2. argparse 默认值改为 `None`，只表达显式覆盖；
3. 未知 YAML/CLI 字段直接报错，backend 专属字段分别校验；
4. active 配置不保存主机绝对路径；归档配置保留历史路径并标注可运行性。

### C. 补齐动态防线

1. 增加 `gallery_trials=3` 的真实 loader 测试；
2. 增加 identity-text/image-caption 事件快照测试；
3. 增加 fresh/resume manifest 与日志测试；
4. 增加微型 CPU train-eval-save-resume 闭环；
5. 有数据和 GPU 时增加真实 CUDA smoke，正式指标只读结构化结果文件。

### D. 职责拆分

golden result 固定后再拆 config、data、protocol、training、evaluation、runtime；每一步用同 checkpoint 验证结果容差。baseline 与 research 原型的顶层迁移最后进行。

## 验收标准

- 事件明确记录 search/gallery mode、trial 数、查询/gallery 模态、caption 来源与 lookup；
- 相同 trial 的 gallery 与全局 Python/NumPy RNG 消耗无关；
- `gallery_trials=N` 只构造并聚合 N 个 gallery；
- fresh run 不能写入非空目录，resume 必须显式；
- 归档 runner 能解析到真实归档配置；
- 核心、PASD、baseline、semantic-imagination 测试均进入 CI；
- wheel 不携带 tests；
- 保留结果继续由总表、配置、commit、checkpoint 和协议唯一定位。

## 本轮验收记录

验收环境：3090 服务器 `/home/lab929/ybj/.conda-envs/salt-vi-flash/bin/python3.9`；工作树基线仍为 `main@c8580d6`，所有修正尚未提交。

- `git diff --check`：通过；
- 协议、局部 RNG、输出目录新增回归测试：8 passed；
- SALT-VI 核心与 vision-text baseline：73 passed；
- PASD offline：24 passed；
- Semantic Imagination：24 passed；
- 两个已归档 RN50 runner 的 `--validate-only`：通过，Stage-A checkpoint SHA-256 匹配；
- `scripts/train.py --help`：通过；
- wheel 构建：通过，保留默认配置与 tokenizer 资源，未打包 tests。

本轮未运行真实数据训练、完整真实数据评估或 CUDA backward，因此不能用以上结果替代论文候选 checkpoint 的冻结式 golden evaluation。正式提交前仍需检查最终 diff 和工作树范围；是否提交由项目维护者决定。

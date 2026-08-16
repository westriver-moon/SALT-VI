# SALT-VI 受控重整计划

> 核查基线：`main@e2f39c62d76f1eb6326b7f2a42bdcccf2d74cf61`。指标事实仍只写入
> `reports/experiment_registry/experiment_registry.csv`；本文只保留尚未完成的工程队列和本轮审查结论。

## 主线边界

当前主线保持不变：geometry-matched PASD RGB/IR 固定 Stage-A 数据契约，保留 RN50 Direct
Stage-A 最佳初始化，Stage-B 先做 ID+WRT 对齐再做跨模态细化；24+16 epoch、all-pairs 的 r3
仍是当前组内最佳。PMT-ViT、No-MBPatch、batch128、FlashAttention 是归档对照，Semantic
Imagination 仍是下一阶段原型。

本轮不修改网络数学定义、loss、optimizer、checkpoint 内容、数据资产、历史指标或总表事实。

## 新审查逐项核对

| 报告问题 | 判定 | 本轮处理 |
| --- | --- | --- |
| evaluation caption seed 与训练 seed 绑定 | 属实 | 新增独立 `eval_caption_seed`，query/gallery caption 均只读该字段；默认固定为 0。 |
| `ProtocolSpec` 不能唯一标识 Fusion/Text/IR | 属实 | identifier 现包含 dataset、search/gallery、trial IDs、方向、backend、test modality、query/gallery modalities、caption lookup/source/seed。 |
| RegDB 未记录具体 numbered trials | 属实 | 记录连续 `trial_ids`，并校验范围必须位于 1–10。 |
| `official=True` 语义含混 | 属实 | 改名为 `official_sampling_protocol`，只表达采样协议，不再暗示检索任务本身“官方”。 |
| 配置未知/失效字段可静默存在 | 属实 | canonical active `configs/stage_a|stage_b` 启用未知键拒绝；reproduction 快照保持历史原样。活跃 YAML 的 TTA/rerank/ensemble/旧 PMT no-op 字段已删除。 |
| `Return_B4_BN` 是 dead flag | 属实 | 从 CLI、输出命名和 canonical classifier 移除；历史配置设为 true 时明确报错，false 仅为快照兼容。当前正式主线原值为 false，历史结果不受影响。 |
| 初始化 checkpoint SHA 只记录不核验 | 属实 | 活跃配置必须同时给出 `training_weight_init_sha256`；训练在构造数据和模型前核验文件 SHA-256。 |
| main/baseline 的 SYSU multi-shot 已漂移 | 属实 | main 在单 camera/ID 少于 10 张时与 baseline 一样使用 replacement；标准 single-shot 结果不受影响。 |
| `models/model.py` 整体是死代码 | 不属实 | RGB/IR/shared RN50 stem 仍被 `clip_model.py` 调用；只删除其中真正无引用的旧 Classifier/初始化工具，并改为显式导出。 |
| `engine/build.py::FM_cat` 无调用 | 属实 | 已删除。 |
| 文档规则与一次性报告并存 | 属实 | 删除已完成清理报告和旧日期型整改报告；本文成为唯一临时整改队列。Semantic Imagination 的有效机制内容暂不误删。 |
| 顶层 `scripts/` 堆积阶段性实验入口 | 属实 | 已完成的 fusion/adaptive/sampling-mining runner、smoke、summarizer、overnight 脚本迁入对应 `experiments/*/source/`。 |
| CI 没有静态语法检查 | 属实 | 增加 `compileall`；完整 Ruff/type/dead-code 规则仍列入后续。 |
| CI 足以证明真实实验正确 | 不属实 | 现有 CPU contract 只能证明接口与小规模行为；真实数据、CUDA、resume 与 golden metrics 仍需单独冻结。 |

同身份 RGB caption 作为 query identity text 是当前明确协议，不按 identity leakage 处理；它现在会在
`ProtocolSpec` 中完整记录，且其随机选择由独立评估 seed 控制。

## 尚未完成的修正

### 1. 冻结论文候选 golden evaluation

使用保留 checkpoint 进行只读真实数据评估，保存 commit SHA、checkpoint SHA、resolved config hash、
data manifest hash、完整 `ProtocolSpec`、`eval_caption_seed` 和结构化 metrics。覆盖：

- SYSU all-search/single-shot/10 trials；
- 至少一个 SYSU multi-shot 边界样例；
- RegDB 明确 numbered trial IDs；
- IR、Text、Fusion 与 IR-to-RGBText 协议快照。

### 2. 运行身份与 resume 闭环

为 fresh run 建立不可覆盖的 run UUID/manifest；checkpoint、events、metrics 共用同一 UUID；resume
必须核验 manifest、config hash、初始化来源与数据清单后才能追加日志。

### 3. 配置 schema 第二阶段

当前已阻止活跃 YAML 的未知键和缺失 checkpoint hash。后续补类型/范围 schema、backend 专属字段
约束，并逐步把 argparse 默认值收敛为“只表达显式覆盖”。不要重写 reproduction 快照。

### 4. 历史边界与大文件拆分

- 将 baseline SYSU protocol 收口为共享实现或增加永久一致性 contract，避免再次漂移；
- 将 `semantic_imagination/KNOWN_LIMITATIONS_AND_CORRECTED_MECHANISM.md` 的仍有效机制并入
  `README.md`/`MATHEMATICAL_SPEC.md`，再删除过程性 audit；
- golden evaluation 固定后，再拆 `entrypoints/train.py`、`engine/build.py`、`dataset.py`、
  `loader.py`；每一步使用相同 checkpoint 做数值回归。

## 验收门槛

- canonical active YAML 出现未知/no-op 键时启动失败；
- 训练初始化 checkpoint 与声明 SHA 不同则在数据/模型构造前失败；
- 改训练 seed 不改变 identity caption；改 `eval_caption_seed` 才改变 prompt 抽样；
- 协议 identifier 对 query modality、caption seed、RegDB trial 不再碰撞；
- SYSU multi-shot 在 camera/ID 少于 10 张时可复现且不污染全局 RNG；
- `scripts/` 不再承载已完成实验的一次性 launcher；
- CPU tests、compileall、wheel 检查通过；
- 最终以真实数据 golden evaluation 证明重整前后保留 checkpoint 指标在容差内一致。

## 本轮验证记录

- 19 份非 reproduction 活跃配置通过 extends 后 strict schema 检查；
- 协议/配置/checkpoint/output-path 针对性测试：33 passed；
- SALT-VI core + vision-text baseline：77 passed；
- PASD offline：24 passed；
- Semantic Imagination：24 passed；
- 总计 125 个独立测试通过；
- `compileall` 覆盖 `src/`、`scripts/`、`experiments/`、PASD 与 Semantic Imagination；
- `scripts/train.py --help` 通过；
- wheel 构建通过，包含 default config 与 tokenizer，排除 tests；
- `git diff --check` 通过。

本轮不运行真实训练、真实数据完整评估或 CUDA backward，不把 CPU contract 当作论文指标证明。

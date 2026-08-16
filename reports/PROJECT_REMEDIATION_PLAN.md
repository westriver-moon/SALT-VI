# SALT-VI 后续修正计划

> 核查基线：`main@c9cbc1af3b7a5d8fc80f9db02801a89c5f995d09`。
> 本文只保留仍未完成、且已由当前代码证据确认的问题；已修正事项不再保留，
> 完成记录由 Git 历史和测试结果承担。

## 主线边界

当前主线仍是 geometry-matched PASD RGB/IR 的 Stage-A 数据契约、RN50 Direct
Stage-A 初始化，以及 Stage-B 的 ID+WRT 对齐与跨模态细化。不得修改已经成立的
网络数学定义、loss、optimizer、同身份 RGB identity caption 协议、reproduction
快照或实验总表事实。

在真实数据 golden evaluation 冻结前，不进行 `train.py`、`build.py`、
`dataset.py`、`loader.py` 的职责拆分，也不删除仍承载当前检索或 checkpoint
兼容行为的路径。

## 尚未完成且核查属实的问题

### 1. 冻结论文候选 golden evaluation

现有 CPU CI 只能证明接口、语法和小规模逻辑，不能证明真实论文指标没有漂移。
使用明确选定的保留 checkpoint 做只读评估，并为每次快照保存：

- commit SHA 与 checkpoint SHA-256；
- resolved config hash 与 data manifest hash；
- 完整 `ProtocolSpec` 和 `eval_caption_seed`；
- Rank-1、mAP、mINP 结构化结果。

最低覆盖 SYSU all-search/single-shot/10 trials、一个 multi-shot 边界样例、
RegDB 明确 numbered trial，以及 IR、Text、Fusion、IR-to-RGBText。后续重构须在
同 checkpoint、同协议、同 seed 下满足预先规定的指标容差。

### 2. 建立最小 run manifest 与完整 resume 一致性校验

当前 full-state checkpoint 保存 model、optimizer、scheduler、scaler 和 RNG，
但仍未绑定 run identity、resolved config、数据清单、初始化来源和协议身份。

先定义可复算的 data manifest 范围，再建立小型 `run_manifest.json`：

- fresh run 生成不可覆盖的 run UUID；
- manifest 记录 resolved config hash、data manifest hash、初始化 checkpoint
  SHA、protocol identifier；
- checkpoint schema 升级后保存 run UUID 和 manifest hash；
- complete resume 在加载 model/optimizer 前复算并比较，不一致直接失败；
- events、metrics、checkpoint 必须共享同一 run UUID。

不得以只记录路径或部分 YAML 代替数据与最终 resolved config 的内容哈希。

### 3. 将 active retrieval 的模糊 legacy 名称迁移为明确协议

`src/salt_vi/retrieval/legacy.py` 不是死代码；它仍实现 IR、Text 和
Fusion(IR + same-identity RGB caption) 到 RGB gallery 的当前协议。启动约束已经
补齐，但命名和上层分支仍待迁移。

完成 golden evaluation 后按同 checkpoint 数值回归：

1. `legacy` 重命名为 `identity_text`，不改变 caption lookup、特征或 metric；
2. 协议统一声明 result keys，上层遍历结果，不再判断 `IS_LEGACY`；
3. 删除 `IS_LEGACY`、legacy-specific logging/save 分支和旧 registry 名称；
4. 保持 `ir_to_rgb_text` 为另一套语义明确的协议。

同身份 RGB caption 是允许的实验协议，不得在迁移中删除。

### 4. 退役历史 model-only resume 与 metric-boost 特殊入口

`resume_train_epoch` 的 model-only 路径和 `metric_boost_resume_epoch` 仍在运行时，
不能在未盘点旧 checkpoint 前直接删除。处理顺序：

1. 盘点仍需继续训练的旧 checkpoint；
2. 将需要保留的 checkpoint 转换为新的 full-state/manifest 契约；
3. 确认 active 配置和论文候选实验不再依赖旧入口；
4. 删除 model-only resume、`metric_boost_resume_epoch` 及其专用分支。

### 5. 配置 schema 与大文件职责收口

- 为 active config 补类型、范围和 backend 专属字段约束；
- 逐步使 argparse 默认值只表达显式 CLI 覆盖，避免覆盖 resolved YAML；
- golden evaluation 固定后再拆大文件，每一步使用相同 checkpoint 做数值回归；
- reproduction 配置保持只读历史资产，不按当前 schema 重写。

## 验收门槛

- golden evaluation 证据完整且可由 hash 唯一复现；
- complete resume 对 run/config/data/init/protocol 任一不一致均在状态加载前失败；
- `identity_text` 与迁移前当前协议在规定容差内指标一致；
- active runtime 不再包含 model-only/metric-boost 特殊恢复分支；
- CPU tests、compileall、active config schema、wheel 边界和 `git diff --check`
  全部通过。

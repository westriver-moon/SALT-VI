# Semantic Imagination v2 重构计划

## 目标

将当前集中在 `plugin.py` 和实验脚本中的采样、解析、重试、聚类、权重及 PASD
转换逻辑拆分为可测试的稳定层，同时保持现有 `build_hypothesis_manifest`、
`cluster_hypothesis_samples` 和 `to_pasd_record` 公共接口兼容。

## 分层

1. `taxonomy.py`：类别、受控状态及状态/value 语义规则；
2. `schema.py`：原子假设、校验问题和校验结果的数据模型；
3. `validator/parser.py`：结构化解析与可唯一确定的表面修复；
4. `validator/semantic.py`：类别约束、state 主导语义、明确冲突和观测重复校验；
5. `validator/feedback.py`：把失败代码转换成下一次重试的定向提示；
6. `validation.py`：只保留旧导入路径兼容层；
7. `sampling.py`：分层调度、确定性种子、带反馈重试、失败溯源及 manifest 构建；
8. `clustering.py`：受控状态精确分组、非结构化文本完全链接回退与 Wilson 区间；
9. `pasd.py`：向 PASD 动态视图记录的兼容转换；
10. `plugin.py`：仅作为旧导入路径的兼容门面。

## 关键不变量

- 不同 category 或 canonical state 永不合并；
- 非结构化文本默认使用完全链接，单链接只用于旧实验复现；
- 校验失败样本保留完整尝试记录，但不进入语义聚类和概率分母；
- 模型主动输出 `no_additional_detail` 与校验耗尽必须分别记账；
- 每个类别分别报告有效率和条件频率；有效类别齐全时全局权重和为1；
- PASD 原有 `hypothesis_weight` 字段保持兼容，新诊断字段只做增量扩展；
- sampling contract 包含验证器、重试和失败处理策略并参与哈希。

## 迁移顺序

1. 在独立 worktree 建立新模块和兼容测试；
2. 运行包单测、PASD 多视图集成测试和小规模真实 InternVL 测试；
3. 将现有机制文档、数学规范及精简实验产物迁入可追踪报告目录；
4. 将实验入口改为调用框架验证器，不再在模型 backend 内部自行聚合计数；
5. 提交新分支，备份并清理 `main` 上原有未提交想象力修改；
6. 合并回 `main`，复测后移除临时 worktree 和已合并主题分支。

## 验收条件

- 旧 API 测试全部通过；
- 新增失败溯源、主动弃权/强制失败分离、状态语义校验测试通过；
- SALT/PASD 现有消费者测试通过；
- 真实模型 smoke test 中 `forced_failure` 不再计入假设权重；
- `git diff --check`、测试和迁移清单均无异常。

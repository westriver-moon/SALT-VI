# 项目修复计划

记录时间：2026-08-17 16:00（Asia/Shanghai）
复核基线：`6be70a785fa52dcd4ca4592a383c2796d56efeb8`
执行状态：本计划已在后续框架重整提交中完成；RegDB checkpoint 缺口按用户要求排除。

## 已确认问题

1. RegDB 索引适配器没有核验 `Visible`/`Thermal` 与 RGB/IR 模态是否对应。
2. Golden evaluation 使用 `--only` 时会用子集覆盖总索引，且旧索引保存临时工作树绝对路径。
3. 运行配置数值校验没有拒绝 `NaN`、无穷值和部分越界概率、种子及学习率因子。
4. PASD 构建指纹只覆盖主要权重，没有覆盖 tokenizer、scheduler、feature extractor 和网络配置。
5. 训练实现保留了已经失效的 model-only resume 与错误的内部 DataParallel 保存分支。
6. 旧 SYSU 超分入口依赖已删除的加载器和配置；其重复配置不再可运行。
7. RegDB golden checkpoint 当前不在服务器上。这是外部资产缺口，按当前范围不下载、不伪造结果。

审查中关于旧 `pasd_offline` 导致 CPU CI 导入失败的问题，在当前基线已由统一 `pasd_plugin` 替换；本轮只通过测试确认，不恢复旧模块。

## 修复顺序

1. 收紧协议、数值与 PASD 资产指纹契约，并补最小回归测试。
2. 修正 Golden evaluation 全量索引的稳定写入和仓库相对路径。
3. 删除失效 resume、DataParallel、旧超分入口和无引用重复配置。
4. 明确配置根目录为当前入口，`reproduction/` 只保存已登记运行快照。
5. 运行 CPU 聚焦测试、服务器环境测试和真实 PASD 资产校验后再推送 `main`。

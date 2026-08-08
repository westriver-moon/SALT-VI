# PR 修正指导 NO.2（2026-08-08）

## 核验结论

本轮以 `agent/refactor-pasd-contracts-20260808` 的 PR #2 为基线。PR #1 仍停留在
`98154bd`，PR #2 的修复提交为 `e844f02`。上一份指导中的八项问题已经修复，
本轮复审提出的六项问题均存在：

1. `Test_Tri_Data` 在读取显式 gallery manifest 前仍解析 `PASD_RGB` 目录，阻断正式
   PASD Stage-B 配置初始化。
2. generation contract 绑定模型路径，不绑定权重、records、输入图像与实现内容。
3. source 完成判断只检查 PNG 存在，不验证 marker 中的内容 SHA。
4. 通用 batch CLI 声称写出 manifest，实际仅五视图路径写出。
5. `worker_chunk_size` 未限定为正整数。
6. 数据集验证器将输出尺寸固定为 `256×512`，未使用生成配置。

## 总体修正结构

修正分为资源解析、构建身份、生成生命周期和接口一致性四层。训练 loader 只负责消费
已解析的 caption 资源；PASD 构建在 worker 启动前生成一次内容契约；调度器区分廉价的
生成态与最终的验证态；通用 batch 与五视图 batch 分别输出符合自身结构的 manifest。

## 阶段一：统一 caption 资源解析

- 新增 caption 路径解析函数。
- 显式 `caption_manifest` 直接作为数据源。
- 未提供 manifest 时才按 `captioner_name + modality` 发现文本目录。
- 使用真实 `Test_Tri_Data` 构造测试覆盖 `captioner_name=PASD`、BLIP manifest、无
  `PASD_RGB` 目录的正式组合。

## 阶段二：建立一次性 build contract

- 在调度器或 batch 生成启动时创建 `build-contract.json`。
- 契约绑定 Stable Diffusion snapshot、PASD checkpoint、YOLO 权重、records 文件、
  records 引用的全部输入图像以及 PASD 实现源码的内容 SHA-256。
- worker 读取已准备的 build contract，不在轮询和断点恢复时重复扫描大权重。
- 每个 source 的 generation contract 绑定 build-contract SHA；任一构建输入变化后，
  旧 marker 不再属于当前构建。

## 阶段三：拆分生成态与验证态

- `source_is_generated` 用于调度轮询，仅匹配 generation contract、view 结构、路径与
  文件大小。
- `validate_source` 用于最终收敛，统一校验输入 SHA、输出 SHA、PNG/RGB/尺寸、非恒定
  图像以及 IR 三通道一致性。
- `consolidate_manifest` 仅收录通过内容验证的 source，并分别报告
  `generated_complete` 与 `validated_complete`；`complete` 等同于最终验证完成。
- scheduler 只有在 `validated_complete=true` 时返回完成。

## 阶段四：统一 CLI 和配置契约

- 通用单 caption、随机 caption 和任意多 caption batch 均写出任务级
  `manifest.jsonl` 与 `manifest.json`。
- 五视图 batch 保留 source 级 marker 和最终视图 manifest。
- 配置加载时要求 `worker_chunk_size > 0`。
- 验证尺寸读取 `target_width` 与 `target_height`。

## 验收

- SALT CPU tests 与 PASD offline tests 全部通过。
- loader 集成测试实际实例化正式 PASD + BLIP gallery 组合。
- build contract 内容变化测试覆盖 records、输入图像、模型权重和实现标识。
- 损坏或替换 PNG 时，生成态与验证态给出不同结果，最终 manifest 不报告完成。
- 通用 batch 的所有 caption mode 均产出 manifest。

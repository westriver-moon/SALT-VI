# PR 修正指导 NO.3（2026-08-08）

## 核验结论

本轮以 PR #3 `agent/refactor-pasd-lifecycle-20260808`、head `ec4fd78` 为基线。
复审提出的五项问题均存在：

1. 同字节数损坏的 PNG 保持 generated 状态，scheduler 最终验证失败后无法进入重新生成。
2. generation 启动后，模型、源码、records 或输入图像发生变化时，最终发布不会重新核对初始身份。
3. generation identity 未包含 Python、核心包、CUDA、cuDNN、GPU 架构和依赖锁文件。
4. pilot records 与 full records 共用一个全局 build identity，full 启动时会使已审核 pilot marker 失效。
5. generic 五 caption 任务按数量被误判为 SYSU five-view；绝对 source key 可使 metadata 和 lock 路径离开输出根目录。

## 总体重构

单一 build contract 拆分为三层：

- generation identity：模型、实现、运行环境和生成参数；
- source contract：generation identity、单张输入内容、caption、seed、模态和输出；
- dataset scope identity：当前 records 集合及其输入集合。

source marker 不再依赖 dataset scope。pilot 与 full 使用相同 generation identity 时，重合 source
的 source contract 保持不变，已经审核的输出直接复用。最终 manifest 独立绑定 full dataset scope。

## 阶段一：显式任务类型与安全路径

- `GenerationTask` 增加明确的 `task_kind`。
- 只有带 `views` schema 的 record 产生 `five_view` 任务。
- generic caption 数量不参与 five-view 判定。
- five-view `source_key` 必须是规范相对路径，不允许绝对路径、盘符或 `..`。
- metadata 与 lock 路径只接受通过规范化的 source key。

## 阶段二：拆分 generation identity 与 dataset scope

- generation identity 绑定 Stable Diffusion、PASD、YOLO、实现源码、生成配置和 environment identity。
- environment identity 绑定依赖锁文件、Python、PyTorch、diffusers、transformers、xformers、CUDA、cuDNN、GPU 型号和计算能力。
- dataset scope 绑定 records 内容和当前输入图像集合。
- source contract 绑定 generation identity 与单 source 内容，不包含 records 全集。

## 阶段三：最终身份复核

- worker 读取 generation identity 后生成 source。
- manifest 发布前重新计算 generation identity 和 dataset scope identity。
- 当前内容与启动身份一致时才写入完成 manifest。
- generation identity 变化后，旧 source marker 自动失效；dataset scope 扩大不影响重合 source marker。

## 阶段四：scheduler repair cycle

- 调度轮询继续使用廉价 generated 状态。
- 全部 generated 后执行 source 内容验证。
- invalid source 的 marker 被统一失效，输出由 worker 原子覆盖。
- scheduler 重新进入生成循环，直到 source 验证和最终身份复核同时完成。

## 验收

- 同字节数 PNG 损坏后，repair cycle 使该 source 回到 pending 并完成重新生成。
- pilot scope 切换到 full scope 后，重合 source marker 仍为 generated 和 validated。
- 模型、实现、环境、records 或输入图像在启动后变化时，最终发布被 generation/scope identity 拒绝。
- generic 五 caption 使用任务级 manifest，不创建 source metadata 或 lock。
- 非法 source key 在 records 加载阶段被拒绝。
- SALT tests、PASD offline tests 与 GitHub Python 3.9/3.10/3.11 checks 全部通过。

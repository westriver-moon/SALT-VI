# PR #1 修正指导

日期：2026-08-08

## 核验结论

PR #1 需要修正后再合并。八项审查意见的核验结果如下。

1. PASD 依赖声明缺少 `ultralytics`。问题属实。人物检测器与安装脚本直接依赖该包，现有 direct requirements 和 lock requirements 均未声明。
2. 动态调度器按 metadata 文件数量判断完成。问题属实。旧 marker、缺失 PNG 和变更后的任务契约都可能被计入完成数量；manifest 汇总也会混入当前 records 之外的 marker。
3. `IR→RGB+Text` 配置沿用旧实验的 metric event 路径和 experiment ID。问题属实。
4. legacy 单任务生成路径把 modality 固定为 RGB。问题属实。IR task 没有把 modality 传入生成器。
5. PASD 文档采用 RGB-only 协议，Stage-B 配置和 records builder 却要求 RGB+IR。问题属实。统一后的协议为：RGB 使用 PASD 五视图和五条 caption；IR 使用原始图像，不生成 IR 改写文本。
6. 训练入口把 `output_path` 同时当作输出根目录和最终实验目录。问题属实。PASD 配置会被重复拼接实验路径。
7. `GenerationConfig.upscale` 不参与几何或输出计算。问题属实。固定输出协议由 `target_height` 与 `target_width` 定义，删除 `upscale`。
8. 当前 SYSU NPY 与 PASD source 顺序没有发生错位。服务器逐位置核验结果为 RGB `22,258/22,258`、IR `11,909/11,909` 全部匹配。现有实现缺少共享的 canonical source index，重构后由 NPY 构建与 PASD loader 共用同一套 source record，并逐位置校验 identity label。

CI 只运行 SALT 测试，未运行 `pasd_offline/tests`。PASD CPU 合同测试将并入同一 workflow；真实模型验证在服务器 PASD 环境执行。

## 阶段修复方案

### 阶段一：统一生成契约

- 将 modality 设为单任务生成接口的必传参数。
- 将任务字段与影响输出的 GenerationConfig 字段合并为一个 generation contract SHA。
- marker 只表示该 generation contract 下的一组完整五视图。
- 删除无效的 `upscale` 字段。
- 将 `ultralytics==8.4.116` 纳入 direct requirements，并把已测试依赖闭包写入 lock requirements。

### 阶段二：以 records 驱动调度与 manifest

- scheduler 启动时加载 records，并构造唯一的 source groups。
- pending、完成判断和最终退出条件全部调用同一套 `source_is_complete(group, config)`。
- `consolidate_manifest` 接收当前 source groups，只读取这些 group 对应且合同匹配的 marker。
- records 数量成为唯一任务规模，不再从 metadata 数量或手工 `expected_sources` 推导。

### 阶段三：统一 RGB-only PASD 插件协议

- records builder 接收按 modality 命名的候选文本输入；当前正式配置只启用 RGB。
- PASD loader 按 modality 独立选择 multiview store 或原始 array，取消“启用 PASD 就必须同时替换 RGB 与 IR”的全局开关。
- RGB 的视觉 view 与文本 view继续支持 independent/paired 采样；IR 直接读取原始训练与评测图像。
- Stage-B 配置使用 `ir_to_rgb_text` retrieval backend，形成 `IR → PASD-RGB+RGB-Text` 实验。

### 阶段四：拆分输出根目录与最终运行目录

- 新增纯函数运行目录解析模块，集中生成实验名称与目录层级。
- fresh train 使用 `output_root`；解析后 `output_path` 只表示最终运行目录。
- resume 和 test 继续使用明确的最终 `output_path`。
- 当前正式 Stage-B 配置迁移到 `output_root`，不再预先写入完整实验目录。

### 阶段五：建立 canonical source index

- 新增共享 SYSU source record 模块，统一 identity、camera、filename 的顺序和 label 映射。
- NPY 预处理与 PASD train view store 共用该模块。
- NPY 构建同时输出 RGB/IR source manifest。
- PASD store 按位置比较 source record label 与 NPY label。

### 阶段六：验证与发布

- 单元测试覆盖 stale marker、缺失 PNG、records 外 marker、IR legacy 输出、RGB-only multiview、输出目录解析和多身份索引顺序。
- GitHub Actions 同时运行 SALT 与 PASD CPU tests。
- 服务器执行人物检测器安装脚本、真实 YOLO 加载、真实 PASD 单图 20-step GPU 推理。
- 修正结果从 PR #1 head 派生到新分支并推送远端。

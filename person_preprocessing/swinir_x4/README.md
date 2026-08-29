# 原尺寸 SwinIR ×4 数据管线

主线代码：`person_preprocessing/swinir_x4/`。生产资产统一位于
`/home/lab929/ybj/datasets/person-assets-512x256/upstream/swinir-real-sr-x4-v1/`。
这是独立离线预处理，不修改原始数据；SALT-VI 可通过 image-tree backend
直接读取原生 ×4 PNG，或读取进一步构建的 512×256 person-fit/resize 数据。

## 已固定的处理约定

1. 原图解码为 RGB 三通道，不读取旧的 resized NPY，不作输入 resize。
2. 红外输入严格沿用旧逻辑：`rint(0.299 R + 0.587 G + 0.114 B)`，再复制为三个通道。
3. 使用指定的 realSR SwinIR-M ×4 PSNR 权重；官方网络配置为 `nearest+conv`，不是 classical SR 的 pixelshuffle。严格加载全部权重，FP32，关闭 autocast。
4. 默认 `tile=0`，整图推理，**不分块**。仅在右侧、底部作对称补边以对齐 8 像素窗口。保留分块选项但不自动启用；若整图实测 OOM，先确定新的配置并使用独立输出目录，再决定是否启用。分块边界结果可能与整图一次推理略有不同，不能把两种结果悄悄混在一起。
5. 裁去补边对应区域，**原生 SR 产物**必须是原始 `W×4, H×4`，不作输出 resize，也不强制 512×256。可选几何后处理另存结果，不改变这一原生 SR 约定。
6. 模型输出先 clamp、×255、round 转为 uint8 RGB。仅 IR 再按旧逻辑对三个量化通道取均值并 `rint`，复制为相同的三通道。
7. 无损 PNG、RGB 模式。IR 三个通道相等；RGB 保留模型的彩色输出。原文件不变。

这是通道兼容处理，不是针对 IR 微调模型，也不意味着恢复了真实红外高频细节。

## 可开关的后处理：等比例缩放、YOLO 引导摆放、模糊补背景

默认关闭，配置独立存放在 `geometry_config.json`，不会使既有 SR 清单、配置或结果失效。

- **关闭**：仅得到 `outputs/` 中原生宽高 ×4 的 PNG，不调用 YOLO 或几何阶段。
- **开启**：先保留上述原生 PNG，再在独立目录 `outputs_geometry_256x512_yolo/` 生成宽256、高512的画布。可修改 `target_width`、`target_height` 和 `output_dir`，不必固定此尺寸。
- 自动两阶段运行时，SR 在子进程执行；该进程完全退出、释放 GPU 后，才启动 CPU 后处理。后处理复用 `.venvs/qri-v1/bin/python` 和现有 YOLO 权重，不安装/升级 SR 环境。配置中的动态库路径仅作用于后处理子进程；CUDA 设备被隐藏，实际 YOLO 推理固定 `device=cpu`。

YOLO 的实际用途不再仅是元数据：

1. 在超分后的图片上检测行人。默认只采用置信度≥0.35、框面积≥原图2%的候选，按置信度、面积与接近画面中心的程度选主要行人；前两名评分过于接近则视为歧义，回退整图居中。这是主体选择启发式，不是身份识别，也没有假定 COCO 检测器在 IR 上可靠。
2. **始终按完整原图计算等比例 fit，不按检测框裁剪，也不为迁就检测框额外缩小。** 只在已有补边余量内平移整张前景，让主要行人中心尽量靠近画布中心。没有余量的轴不移动；如 RegDB 原图比例已是1:2，检测框不会强迫它产生位移。
3. 背景沿用旧实现：同图等比例 cover、居中裁切背景副本、Gaussian blur（默认24像素），前景边缘羽化（默认2像素）。裁切只针对模糊背景副本，**完整前景不裁切**。
4. 检测到的可靠行人框外扩5%，映射到目标画布后作为矩形保护区，令该区使用完整前景而非羽化混合，避免边缘人体被混入模糊背景。这不是人体分割掩膜。
5. 无可靠检测/多人歧义时不按框移动。正常未检出是可记录的回退；缺模型、缺依赖、推理异常会明确报错，不会静默假装完成 YOLO。IR 输出继续保持相同的三个通道。

YOLO 开关由 `geometry_config.json` 的 `detector.enabled` 控制。设为 `false` 时，几何处理退回旧插件的等比例整图居中＋模糊补边；此路径已做逐像素一致性测试。改变检测器、阈值或目标尺寸时用新的后处理目录，以免混合不同配置。**几何总开关与 YOLO 开关是两层独立开关。**

### 使用命令

```bash
cd /home/lab929/ybj/SALT-VI/person_preprocessing/swinir_x4
PY=/home/cgv841/anaconda3/envs/clipreid/bin/python

# 原管线：关闭后处理（目前默认行为）
$PY pipeline.py run --gpu 0 --geometry off

# 一次命令：原生 SR 完成后，再接 CPU 几何阶段
$PY pipeline.py run --gpu 0 --geometry on

# 已有原生 SR 时，只补做后处理，不再运行 SwinIR，也不需要 GPU
$PY pipeline.py postprocess --geometry on
$PY pipeline.py verify-geometry

# 只处理已经完成的 SR 图片（显式允许生产数据尚未齐全）
$PY pipeline.py postprocess --geometry on --available-only

# 更改默认行为：将 geometry_config.json 顶层 enabled 改为 true/false。
# 命令行 --geometry on/off 优先于配置。
```

`--datasets sysu regdb llcm` 可选子集。几何阶段的 `--limit N` 限制选取的已有 SR 图片数；`run --limit N --geometry on` 则先最多新生成 N 张 SR，再为所有已完成 SR 补齐几何产物。`--source-root`、`--output-root` 可用于独立后处理/验证；正式输入和几何输出必须是互不嵌套的独立目录。

每张图的后处理记录位于 `metadata/{dataset}/原相对路径.json`，保存检测框、选择原因、保护区、等比尺寸、补边、实际平移和源图到目标画布的变换。`contract.json` 固定生成约定；`progress.jsonl`/`last_run.json` 记录进度。续跑核对原生输入的大小/mtime、结果尺寸、图像完整性、IR通道和完成记录；缺失或损坏的本阶段产物重算。LLCM 的相机目录别名在新结果目录中重新建立，不指向或覆盖原生 SR 结果。

### 本次后处理验证

- 共28项单元/集成测试通过，涵盖旧通道逻辑、默认关闭、开关编排、旧插件像素一致性、YOLO平移与保护区、异常配置、损坏重算及原生结果不被覆盖。
- CPU 处理既有 78 张真实 SR 样本（每个数据集的 RGB/IR 各13张），全部通过输出、IR通道和别名校验。
- 55 张选出主要行人，37 张实际改变平移位置；23 张没有可靠候选，回退整图居中。RegDB 的26张图都没有平移；同画布尺寸、同宽高比时像素保持不变。
- 对同一批样本再次运行，78张全部跳过，没有重复生成；另检查了RGB/IR位移较大样本的前后画面。功能与几何约定已验证，尚未通过ReID实验验证该后处理对识别指标的影响。
- 样本位于服务器 `geometry_checks/yolo_v1_78/`。只生成测试样本，未启动全量 SR 或全量几何处理。下方旧的6.25小时估计**只含原生 SwinIR SR，不含新 CPU 后处理**。

## 2026-08-26 验证记录

- 11 项测试通过，包含从旧代码提取通道函数的逐像素一致性测试、整图/可选分块的尺寸和像素位置测试、NaN 拒绝、损坏结果检测、别名与划分去重。
- 真权重严格加载成功：`params_ema`，11,715,559 个可训练参数；模型以 eval/no_grad 执行。
- 在实际空闲的物理 GPU 0 上，FP32、batch=1 整图测试了典型图、95% 分位图、最大面积图、最宽图和最高图，没有 OOM。最大 SYSU 原图 485×877 → 1940×3508；PyTorch 已分配峰值 4.28 GiB，预留峰值 8.21 GiB（预留包含已分配，不应相加，也不含全部 CUDA 上下文开销）。因此默认无需分块。
- 完成 72 张分层样本及 6 张各组最大图的端到端测速；测试结束后 GPU 已释放，**没有启动全量生产**。
- 详细服务器记录：`whole_image_probe.json`、`cpu_smoke_test.json`、`benchmarks/20260826-131715/report.json`。CPU 记录中的 GPU 被占用状态是该测试当时的历史状态；随后短任务退出，已完成 GPU 实测。

| 数据集 | 唯一原图数 | 单张 3090 点估计 | 粗略安排范围 | PNG 存储点估计 |
|---|---:|---:|---:|---:|
| SYSU-MM01 | 44,745 | 4.45 小时 | 3.6–6.7 小时 | 11.4 GiB |
| RegDB | 8,240 | 0.24 小时（约 14.5 分钟） | 12–22 分钟 | 0.5 GiB |
| LLCM | 46,767 | 1.57 小时 | 1.3–2.4 小时 | 3.7 GiB |
| 合计 | 99,752 | 6.25 小时 | 约 5–9.5 小时 | 15.6 GiB |

LLCM 另有 15,846 条测试相机路径别名，不重复推理。时间范围是小样本外推的规划范围，非保证或统计置信区间；不含排队、别人占卡、重试和训练接入。存储量也为小样本估计，建议至少预留 25 GiB，原始数据保持不动。

## 数据覆盖

- SYSU-MM01：`exp/train_id.txt + val_id.txt + test_id.txt` 的原始 cam1–6 图片。cam3/6 是 IR。覆盖全部测试 gallery 候选，不限于某一次随机抽样。
- RegDB：10 个 trial 的 train/test × visible/thermal 全部索引并集；同一图只推理一次，各 trial 的原标签保留在清单中，不混用重编号。
- LLCM：train/test × vis/nir 全部索引，并核对真实测试读取的 `test_vis/test_nir/cam*/pid`。字节完全相同的相机副本映射到同一超分结果，以相对符号链接覆盖相机路径，不重复推理/存储图像；若不相同则单独处理。
- 不读取已有 PASD、SwinIR、`*_modify`、Text 等派生文件。

清单：`manifests/images.jsonl`，包含源路径、模态、尺寸、划分、trial 标签、测试路径别名及输出路径；统计见 `manifests/summary.json`。

## 运行

以下命令在 3090 服务器执行。Python 环境复用已有依赖，不修改环境：

```bash
cd /home/lab929/ybj/SALT-VI/person_preprocessing/swinir_x4
PY=/home/cgv841/anaconda3/envs/clipreid/bin/python
$PY pipeline.py inventory
$PY -m unittest -v test_pipeline
# GPU 是 nvidia-smi 物理编号；仅在实际空闲时填写，例如 0。
$PY pipeline.py benchmark --gpu 0 --per-stratum 3
# 若先单独检查整图显存（含各数据集最大图，不保存超分图片）：
$PY probe_whole.py --gpu 0
# 下面才是全量生产，需明确启动；同一命令可断点续跑。
$PY pipeline.py run --gpu 0
# 或只做一个数据集：
$PY pipeline.py run --gpu 0 --datasets regdb
$PY pipeline.py verify
```

启动前两次检查指定 GPU 上没有计算进程且低占用；遇到别人占用即退出，不杀进程、不自动换卡。检查不是跨用户资源锁，不能阻止别人稍后启动任务。全量运行前应协调独占 GPU。单进程、单卡、batch=1；不要并发写同一个输出目录。

产物：`outputs/{sysu,regdb,llcm}/原相对路径.png`。`contract.json` 记录配置、模型结构、所加载 checkpoint 键和版本；有效的已完成 PNG 跳过，缺失/损坏/尺寸或 IR 通道不符的本管线产物重算，先写临时文件再原子替换。异常立即退出，可原命令续跑。`outputs/progress.jsonl` 记录逐图时间，不靠它决定是否完成。更改生成约定时应使用新的 manifest/output 目录，不能混入旧产物。

测速结果与样本另存 `benchmarks/时间戳/`，不混入生产结果。按数据集、模态、推理面积四分位分层抽样；另测各组最大图，验证大图显存与尺寸。耗时是单卡空闲、当前整图/FP32配置下的粗估，区间不是统计置信区间，不含排队时间。

注意：输出改为 `.png`。本任务只构建超分数据；训练接入时应通过 manifest 中的 `output`/`aliases` 映射更新索引。不能直接把仍指向 `.jpg/.bmp` 或 resized NPY 的现有训练配置改个根目录就使用。本管线没有修改训练阶段自身的输入变换。

## 模型来源

- [指定官方权重](https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth)
- [官方推理配置](https://github.com/JingyunLiang/SwinIR/blob/main/main_test_swinir.py)
- vendor 文件由服务器上已有的官方源码副本复制，来源版本见 config.json；保留 LICENSE。
- 旧通道逻辑来源：`/home/lab929/ybj/SALT-VI/src/salt_vi/utils/super_resolution/build_sysu_swinir_x2.py` 的 `normalize_ir`、`prepare_images`、`finalize_images`。测试会从旧源文件提取函数，逐像素比较，而不运行旧构建流程。

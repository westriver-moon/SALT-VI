# SALT-VI migrated evidence document

> Source document ID: `source_core:reports/experiment_registry/historical_archives/ybj2_sysu_multiseed_20260717/SYSU_MULTI_SEED_REPRODUCTION_PLAN.md`  
> Original SHA-256: `65363522c5b4c5d363ff5ff66030159bb444eb60cf6a185243e247f1de925a02`  
> This is read-only experiment evidence, not an active runtime instruction.

# SALT-VI：SYSU-MM01 四种子复现方案

## 目标

在 `/home/cgv841/ybj2` 使用官方 SALT-VI 提交
`682742130f2fb7bca26dabd92bc5a788225d7541`、同一份官方 SYSU base checkpoint，
分别以训练种子 `1、2、3、4` 在四张 RTX 3090 上复现 Tri-SYSU-MM01。

本轮只覆盖 SYSU-MM01，不重训 base model，不覆盖 LLCM 或 RegDB。官方 clone
保持干净，控制文件、环境记录、权重和结果全部放在 clone 外部；禁止读取或写入
`/home/cgv841/ybj`。

## 目录和接口

```text
/home/cgv841/ybj2/
├── SALT-VI/
├── data/sysu/
├── artifacts/official/
└── reproduction/sysu_multiseed/
    ├── requirements.lock.txt
    ├── manifest.json
    ├── preflight.sh
    ├── run_seed.sh
    ├── launch_all.sh
    ├── collect_metrics.py
    └── runs/seed_<n>/attempt_<nn>/
```

- `preflight.sh check|official-eval|smoke`：门槛检查、官方模型 MER 推理、seed 1 单轮训练。
- `run_seed.sh <seed> <physical_gpu> <attempt>`：完整训练并执行 MER 测试，拒绝覆盖。
- `launch_all.sh [attempt]`：按 `1→0、2→1、3→2、4→3` 创建 tmux 会话。
- `collect_metrics.py --root <runs>`：生成 CSV、JSON、Markdown 汇总。

## 固定资源与环境

- `VI_sysu_BASE.pth`：Google Drive ID `1aY4BcXhhbHt2XOnxkj7U8s0Rh4wK7vQS`，
  期望大小 354,597,422 字节。
- `SYSU_SALT-VI.pth`：Google Drive ID `1BmvOP-kLAwRVxI4bissQqYDYmLl3HDBc`，
  期望大小 354,598,388 字节。
- 作者日志：Google Drive ID `1jMmAd3Sw1TfnYoB9ga-_lm8oYikeIbNd`。
- Conda 环境名为 `tvilfm-ybj2`，Python 3.10.4、PyTorch 2.0.1、
  torchvision 0.15.2，使用 CUDA 11.7 wheel。

SYSU 数据通过 `/home/cgv841/ybj2/data/sysu` 视图读取：原始相机目录、划分和
预处理数组来自 `/home/cgv841/datasets/SYSU-MM01`；`Text` 来自官方 clone。
不重新运行预处理，不复制数据，也不建立任何指向 `ybj` 的链接。

## 启动门槛和预飞

正式运行前必须满足：隔离检查通过；官方 Git 根、origin、commit 和干净状态正确；
权重大小及 SHA-256 已记录且可加载；数据与文本资源存在；磁盘剩余至少 25GB；
目标 GPU 无计算进程；运行时导入和 `scripts/train.py --help` 通过。

先运行官方 `SYSU_SALT-VI.pth` 的 MER 推理，再运行 seed 1 的单 epoch smoke。
任一步失败即停止，不启动四种子训练，不静默升级依赖或改变 batch size。

## 正式配置

四个种子共享以下参数：`dataset=sysu`、`training_mode=RGB_IR_Text`、
`joint_mode=uni`、`captioner_name=Blip`、`llm_aug=true`、`llm_aug_prob=0.5`、
`Feat_Filter=true`、`Fix_Visual=true`、`fusion_way=add`、`loss_names=id,wrt`、
`lr_txt=0.00035`、`text_weight_decay=0.0005`、
`text_weight_decay_bias=0.0005`、`batch_size=32`、`total_train_epoch=120`、
`eval_epoch=2`、`eval_start_epoch=0`、`test_modality=Text,IR,Fusion`。

每个进程只看见一张物理 GPU，并在进程内使用逻辑 `gpu_id=0`。每次重试使用新的
attempt 目录；单个种子失败不终止其他种子；OOM 不自动降低配置；不得删除或排除
表现较差的种子。

## 汇总与完成标准

输出 `metrics_by_seed.csv`、`metrics_summary.json`、`reproduction_report.md`，包含
每个种子的无 MER/MER Rank-1、mAP、mINP，以及四种子的均值、样本标准差、
最小值、最大值和与作者参考结果的差值。

不设数值通过门槛。技术完成要求是四个种子均完成 120 epoch、各有唯一可加载的
最佳 Fusion checkpoint、四次 MER 测试完成、日志无 traceback/NaN、统计包含全部
四个种子，且官方源码仓库保持干净。

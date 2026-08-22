# SALT-VI 2026-08-21/22 实验可复现归档计划

最后更新：2026-08-22 18:54 CST（H3-35 按用户要求留给服务器 watcher 运行至次日）
服务器：`lab929-3090`
仓库：`/home/lab929/ybj/SALT-VI`
分支：`codex/pmt-mscm-phased-pasd-20260821-v2`
基线提交：`27ff03a16f2d0d99c1511a484e9aa0c891365207`

## 1. 目标与不可违反的约束

本次归档覆盖 PMT-MSCM 混合损失训练、H3 QCT-0.40 的 35 epoch 独立延长实验，以及
2026-08-21/22 的 QRI 加速、眼镜验证、像素控制、文本标注和文本想象实验。归档必须满足：

1. 完整保留每个实验实际使用的源码、原始配置、解析后的 `configs.yaml`、
   `run_manifest.json`、启动命令、日志、事件、指标、checkpoint 和最终模型；
2. 归档快照时工作区是带 20 个 tracked 修改和 86 个 untracked 文件的实验工作区；随后仅增加了当前状态文档、总表登记和说明性修订，不能仅记录
   Git HEAD；必须冻结工作树的逐文件快照、二进制 patch 和 Git bundle；
3. 大模型、数据集和第三方源码不复制进 Git，通过绝对路径、字节数、SHA-256、目录树摘要
   或第三方 Git 提交进行不可变引用；
4. 只执行“复制 → 生成 provenance → 全量 SHA-256 → 再校验”，不移动或删除原实验，
   不回退、不清理当前仓库；
5. 运行中的实验只能在完成门禁通过后归档，不能把中间 checkpoint 误标为最终结果。

## 2. 当前仓库审计

审计时分支与远端 `origin/codex/pmt-mscm-phased-pasd-20260821-v2` 完全同步，ahead/behind
为 `0/0`；无 staged 文件、无冲突，`git diff --check` 和 `git diff --cached --check` 通过。

工作区仍不干净：20 个已跟踪文件被修改，86 个文件未跟踪。未跟踪内容约 400 KB，按
一级目录分为 38 个 `configs/`、28 个 `scripts/`、11 个 `plugins/`、7 个 `reports/`
和 2 个 `pasd_plugin/` 测试文件。没有 checkpoint、训练日志、大模型或异常大文件误入
仓库。主要风险不是垃圾文件，而是实验代码和配置尚未被 Git 提交，单靠 HEAD 无法复现。

## 3. 归档对象与完成门禁

| 归档 ID | 原目录 | 目标目录 | 完成门禁 |
|---|---|---|---|
| `stage_a_hybrid_20260821` | `experiments/stage_a/SALT-VI-pmt-mscm-hybrid-loss-20260821` | `experiments/archive/stage_a/SALT-VI-pmt-mscm-hybrid-loss-20260821` | scheduler 为 completed、无 running/failed、`results.json` 存在、H1-H4 均有至少一个最佳 `model_IR_*.pth` 和最终 latest checkpoint、目录静默 120 秒 |
| `stage_a_h3_qct_040_e35_20260822` | `experiments/stage_a/SALT-VI-pmt-mscm-hybrid-loss-h3-qct-040-e35-20260822` | `experiments/archive/stage_a/SALT-VI-pmt-mscm-hybrid-loss-h3-qct-040-e35-20260822` | `eval_epoch=34`、至少一个最佳 `model_IR_*.pth`、最终 latest checkpoint、resolved config 和 run manifest 均存在、目录静默 180 秒 |
| `qri_acceleration_20260821` | `experiments/qri_acceleration/qri-fast-search-gpu0-20260821` | `experiments/archive/qri/qri-fast-search-gpu0-20260821` | Phase A、Phase B dry-run、Qwen pilot 和质量审计 JSON 完整 |
| `qri_glasses_20260821` | `experiments/qri_glasses/qri-glasses-gpu0-20260821` | `experiments/archive/qri/qri-glasses-gpu0-20260821` | PASD、SD1.5 sweep、融合选择和 Qwen 视觉审计结果完整 |
| `qri_imagination_control_20260822` | `experiments/qri_imagination_control` | `experiments/archive/qri/qri-imagination-control-20260822` | Stage 8/9 与多样本验证的指标和身份审计完整；保留否定结果 |
| `qri_text_annotations_20260822` | `experiments/qri_text_annotations` | `experiments/archive/qri/qri-text-annotations-smoke-20260822` | v3 与 v3 compact smoke 的 worker summary 和 shard summary 完整；明确标注生产全量任务未启动 |
| `qri_text_imagination_20260822` | `experiments/qri_text_imagination/thinking_ablation_gpu0_20260822_v1` | `experiments/archive/qri/qri-text-imagination-thinking-ablation-gpu0-20260822` | thinking/no-thinking 六次记录、boards 和权威 `metrics.json` 完整 |

H3-35 当前仍由服务器 watcher 运行，按用户要求本轮不干预；完成后再执行下述门禁。其 `train.log`、`train_e35.log` 是两次启动失败诊断，成功训练日志是
`train_e35_retry.log`。三者都保留，并额外复制到 `provenance/launch_diagnostics/`；启动失败
不能混入最终科学指标。

训练默认 `max_save_model_num=1`，`model_IR_<epoch>.pth` 只在 Rank-1 创新高时写入，
因此最佳模型的文件名不一定等于最后 epoch。最终训练状态以 `checkpoint_latest.pth` 和
`eval_epoch`/scheduler 完成状态为准，最佳权重以保留的 `model_IR_*.pth` 为准。

## 4. 源码与配置固化方法

共享源码快照目标：

`/home/lab929/ybj/experiments/archive/_shared/source/SALT-VI-worktree-20260822-1604/`

快照包含：

- `source_tree/SALT-VI/`：`git ls-files -c -o --exclude-standard` 得到的全部 tracked 与
  untracked 源文件，内容来自当前工作树；
- `SALT-VI-source.tar.gz`：同一源码树的便携压缩包；
- `repository.bundle`：全部 Git refs；
- `worktree.patch` 和 `index.patch`：相对 HEAD 的二进制安全 patch；
- `source_inventory.sha256`：每个源码文件的 SHA-256，整体摘要写入
  `snapshot_manifest.json`；
- `repo_state.json`：分支、HEAD、upstream、ahead/behind 和完整 porcelain v2 状态；
- `environment.json`、`pip-freeze.txt`、GPU/驱动/Python/PyTorch/CUDA 信息；
- `external_assets.json`：外部数据、权重、模型目录和第三方源码的不可变指纹；
- `inventory.sha256`：共享快照自身的全量校验清单。

另建立强外部资产注册表：

`/home/lab929/ybj/experiments/archive/_shared/assets/SALT-VI-assets-20260822-strong-v1/`

它对实验实际读取的 SYSU-MM01 `cam1–cam6`、`exp/` 和 BLIP RGB/IR 字典、14 GB 的
预计算 SwinIR 树、实际 llama-server 二进制和无可读 Git 提交的 SwinIR 源码树计算
内容级 SHA-256，补足仅记录路径/大小无法证明字节一致性的缺口。与本批实验无关且
`lab929` 不可读的历史 Qwen 标注不进入逐项实验依赖列表，但由下述所有者侧完整树摘要
覆盖。注册表只保存摘要，不复制这些外部资产。

服务器还提供数据所有者账号 `cgv841`。使用该账号执行只读探针后，强资产注册表额外保存
`owner_asset_fingerprint.json` 与可重跑的 `owner_asset_probe.py`：完整 SYSU-MM01 根目录
共 135,402 文件、6,770,465,781 字节，树摘要为
`ff480fa7df2197f84d166b60802094a6713dfd23c6066c331c4b83073de95409`；SwinIR 提交为
`6545850fbf8df298df73d81f3e8cba638787c8bd`，其运行工作树摘要与 `lab929` 侧结果一致。
所有者账号只用于读取和计算摘要，不修改数据或源码。

每个实验归档内还会保存一份与该实验直接相关的 `provenance/source_files/`，例如混合损失
实验保存 pipeline、H1-H4/H3-35 配置、scheduler 和完整 `src/salt_vi/`；QRI 实验保存对应
`configs/`、`scripts/`、`reports/`、Qwen plugin 和需要的 PASD adapter。它们从共享快照
复制，不从归档时可能已变化的活跃工作区复制。

运行目录中的 resolved `configs.yaml` 与 `run_manifest.json` 原样保留，并在
`archive_manifest.json` 中逐项登记路径、大小和 SHA-256。因此归档同时具备“用户配置”、
“当时工作树源码”和“训练真正解析后的配置”三层证据。

## 5. 外部资产规范

以下外部资产不进入 Git，但必须进入 `external_assets.json`：

- SYSU-MM01 原始数据根目录和 PASD 派生 manifest；
- PMT-ViT 初始化权重；
- Qwen3.8-27B GGUF 与 mmproj；
- SwinIR、YOLOv8 pose、SCHP-LIP、SAM ViT-B 权重；
- SD1.5 Inpainting、PASD SD1.5 base、PASD checkpoint-100000 目录树；
- QRI ReID 身份 checkpoint；
- SwinIR、llama.cpp、SCHP 和 SAM 第三方源码树。

单文件记录 SHA-256；Diffusers/PASD 模型目录记录按“文件 SHA + 相对路径”计算的目录树
SHA-256；第三方 Git 源码记录提交与 dirty 状态；大数据根目录记录文件数、总字节数、
路径和 mtime，派生 manifest 另做 SHA-256。SYSU-MM01 的实际读取范围和预计算 SwinIR
树还在强资产注册表中补充内容级目录摘要。模型和数据原地保留，不复制到仓库。

## 6. 每个归档的固定结构

```text
<archive-root>/
├── <原实验目录的全部内容>
└── provenance/
    ├── archive_manifest.json
    ├── final_metrics.json
    ├── inventory.sha256
    ├── verification.json
    ├── source_files/
    ├── shared_snapshot/
    │   ├── snapshot_manifest.json
    │   ├── repo_state.json
    │   ├── environment.json
    │   ├── pip-freeze.txt
    │   ├── external_assets.json
    │   └── archive_spec.json
    └── launch_diagnostics/   # 仅有启动诊断时出现
```

`final_metrics.json` 从所有事件 JSONL 提取最后训练、最后评估和 Rank-1 最佳评估。QRI
实验的 `metrics.json`、`summary.json` 和 `selection.json` 作为权威原始结果保留并登记，
不会把 smoke/negative evidence 改写成完整训练结果。

## 7. 执行命令

归档清单：`configs/experiments/archive_20260822.json`
归档工具：`scripts/experiments/archive_research_runs.py`

```bash
cd /home/lab929/ybj/SALT-VI
PY=/home/lab929/ybj/.conda-envs/salt-vi-flash/bin/python3.9

# 1. 固化当前 dirty worktree 和外部资产指纹
$PY scripts/experiments/archive_research_runs.py snapshot

# 2. 查看全部门禁
$PY scripts/experiments/archive_research_runs.py status

# 3. 归档已满足门禁的对象
$PY scripts/experiments/archive_research_runs.py archive --id <archive-id>

# 4. 监视仍运行的实验，通过门禁后自动复制和校验
$PY scripts/experiments/archive_research_runs.py watch \
  --poll-seconds 300 --max-hours 36

# 5. 独立重算归档 SHA-256
$PY scripts/experiments/archive_research_runs.py verify --id <archive-id>

# 6. 全部对象完成后，执行原目录/归档载荷/源码配置/资产引用的最终审计
$PY scripts/experiments/archive_research_runs.py audit \
  --output /home/lab929/ybj/experiments/archive/final_audit_20260822.json
```

监视器只能执行清单中的固定目标，若门禁缺文件、scheduler 失败、最终 epoch 不足、目标
已存在但缺少 inventory，都会停止该对象并报告错误，不覆盖既有目录。

本轮状态：H4 归档已生成并通过 `verification.json`；五个 QRI 归档均已通过独立校验。H3-35
仍是外部运行中的未封存对象，不得因为代码、文档或 CSV 已推送而标记为完成。

## 8. 验收标准

每个对象只有同时满足以下条件才算归档完成：

1. 目标目录存在，原目录仍存在；
2. 完成门禁证据写入 `archive_manifest.json`；
3. 共享源码快照摘要、分支和 HEAD 已写入实验 manifest；
4. 直接相关源码和用户配置位于 `source_files/`；
5. resolved config、run manifest、日志、事件、最终模型和 latest checkpoint 均在归档中；
6. 外部资产清单无 required missing；
7. `inventory.sha256` 对 sealed payload 全量复算通过，`verification.json` 为 `ok=true`；
8. QRI smoke、否定结果和未启动的生产任务均按真实状态标注，不提升结论等级。

最终 `audit` 还必须证明：原实验目录仍存在；原目录与归档中非 `provenance/` 科学载荷的
文件集合及 SHA-256 完全一致；每个 `source_files/` 副本与共享源码快照一致；resolved
config/result 的登记哈希一致；共享源码快照和强资产注册表自身均通过 inventory 校验。

## 9. 回退与保留策略

本次归档不改变训练目录和仓库文件历史，因此不需要执行回退。若归档副本校验失败，只删除
工具本次创建且名称精确匹配 `.partial-<pid>` 的临时目录，然后从原目录重新复制；原实验、
当前分支、工作区修改和未跟踪源码始终保留。最终确认完成前不做 `git clean`、不删 checkpoint、
不移动原结果，也不提交或推送用户未批准的改动。

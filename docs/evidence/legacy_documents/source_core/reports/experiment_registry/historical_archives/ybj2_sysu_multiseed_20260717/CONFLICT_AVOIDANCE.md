# SALT-VI migrated evidence document

> Source document ID: `source_core:reports/experiment_registry/historical_archives/ybj2_sysu_multiseed_20260717/CONFLICT_AVOIDANCE.md`  
> Original SHA-256: `4cb474d5e36da2246352fa05538b5d6c0d1467c2ed397de033387324a4ada076`  
> This is read-only experiment evidence, not an active runtime instruction.

# ybj2 与 ybj 冲突规避方案

## 目标与边界

- 原改进项目保持在 `/home/cgv841/ybj`。
- 官方 SALT-VI 复现保持在 `/home/cgv841/ybj2/SALT-VI`。
- 本方案只防止路径、Git、环境、数据和输出相互污染，不增加与当前风险无关的权限或备份机制。

## 当前审计基线

- `/home/cgv841/ybj2/SALT-VI` 是独立 Git 仓库，不属于 `/home/cgv841/ybj`。
- `origin` 为 `https://github.com/WHU-HZY/SALT-VI.git`。
- 基线提交为 `682742130f2fb7bca26dabd92bc5a788225d7541`，分支为 `main`，工作树干净。
- 官方复现目录内没有 `/home/cgv841/ybj` 的硬编码引用。
- 官方复现目录内没有软链接，因此当前不存在通过链接写回 `ybj` 的风险。

## 必须遵守的隔离规则

1. 代码操作只能在 `/home/cgv841/ybj2/SALT-VI` 内执行，不把官方代码复制、同步或 checkout 到 `/home/cgv841/ybj/SALT-VI`。
2. 训练输出使用复现仓库自身的 `logs/raw/source_core/logs/`、`base_model/` 和 `debug_logs/`；配置中不得出现 `/home/cgv841/ybj/` 输出路径。
3. 后续安装依赖时使用独立 Conda 环境名 `tvilfm-ybj2`，不得向主项目当前使用的 `clipreid` 环境安装官方依赖。
4. 不把 `ybj` 内的数据集目录以可写软链接挂入 `ybj2`。如需共享原始数据，应使用 `/home/cgv841/datasets` 之类的中立数据源，并确保预处理生成文件写入 `ybj2` 自己的目录。
5. 在安装依赖、准备数据或启动训练前运行 `/home/cgv841/ybj2/verify-isolation.sh`。

## 本次实施的最小防护

- 将本文件放在 `ybj2` 根目录，避免修改官方仓库工作树。
- 创建只读式校验脚本 `verify-isolation.sh`，检查仓库真实路径、Git 根、官方远端、旧工作区硬编码、指向旧工作区的软链接以及工作树状态。
- 不创建 Conda 环境或数据链接：这些资源尚未进入复现步骤，提前创建没有必要。
- 不修改 `/home/cgv841/ybj` 的权限、Git 配置或文件。

## 校验通过标准

- 仓库真实路径和 Git 根均为 `/home/cgv841/ybj2/SALT-VI`。
- `origin` 必须为官方仓库 URL。
- 仓库文件中不得出现 `/home/cgv841/ybj/`。
- 不得存在解析后落入 `/home/cgv841/ybj/` 的软链接。
- Git 工作树允许以后出现明确的复现配置与结果，但校验会报告其状态，便于启动前人工确认。

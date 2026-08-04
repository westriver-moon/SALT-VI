# SALT-VI 代码释放迁移报告（2026-08-03）

## 结论

历史实现已释放为 SALT-VI 自己维护的可运行代码。运行时直接从 `src/salt_vi` 和 `scripts/train.py` 加载，不通过旧实现目录或中间映射层转发。

## 新实现

- canonical Python 包：`/home/cgv841/ybj/SALT-VI/src/salt_vi/`
- 统一训练入口：`/home/cgv841/ybj/SALT-VI/scripts/train.py`
- 视觉-文本基线：`src/salt_vi/baselines/vision_text/`
- canonical 配置：`configs/default.yaml`、`configs/stage_a/`、`configs/stage_b/`、`configs/metric_boost/`、`configs/experiments/`、`configs/super_resolution/`、`configs/vision_text/`

代码清单包含 142 个文件；canonical 配置共 52 个 YAML，继承链断裂数为 0。

## 总表与直接引用

- 唯一总表：`reports/experiment_registry/experiment_registry.csv`，1830 行。
- `code_root` 直接指向新包：1830/1830。
- 非空 `config_path`：291；可解析到 SALT-VI 的路径：153。
- 非空 `run_dir`：192；可解析到 SALT-VI 的目录：11。
- 原始日志 canonical 路径：4158 个，缺失 0 个。

活动脚本已直接调用 `salt_vi.*` 或 `scripts/train.py`；`semantic_config_migration` 仅保留为迁移输入和历史证据，不作为新运行入口。

## 边界

- checkpoint、pretrained 权重和数据集按此前指令没有移动、删除或改名；少量权重兼容文件名保留历史名称。
- 所有操作严格限制在 `/home/cgv841/ybj` 内。

## 验证

- `python3 -m compileall src scripts`：通过
- 关键模块导入：通过
- `scripts/train.py --help`：通过
- canonical 配置继承链：通过
- 非权重路径旧项目名：0
- 原始日志 canonical 路径缺失：0

审计文件：`runtime/code_release_manifest.json`、`runtime/code_release_final_audit_20260803.json`、`runtime/registry_canonicalization_20260803.json`。

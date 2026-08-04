# Checkpoint 路径核验与清理

核验时间：2026-08-03T13:17:56.512137+00:00

对象：/home/cgv841/ybj/SALT-VI/reports/experiment_registry/experiment_registry.csv

| 分类 | 数量 | 处理 |
|---|---:|---|
| 原有非 canonical checkpoint 路径 | 106 | 逐行核验 |
| 可解析旧软链接 | 4 | 改写为 SALT-VI canonical 路径 |
| 确认失效路径 | 102 | 清空 checkpoint_path，标记 checkpoint_path_status=无 |
| 原本未记录 checkpoint | 1537 | 保持空路径并标记为无 |

逐行审计记录：/home/cgv841/ybj/SALT-VI/runtime/invalid_checkpoint_paths_audit_20260803.json

失效路径不会继续出现在总表的 checkpoint_path 字段中；原始核验字符串仅保存在审计 JSON 中，不作为可用权重路径。

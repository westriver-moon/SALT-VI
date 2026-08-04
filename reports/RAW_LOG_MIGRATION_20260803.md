# 原始训练日志迁移记录（2026-08-03）

本次迁移仅在 `/home/cgv841/ybj` 内执行，采用同文件系统 `rename`，未复制出第二份日志，也未触碰 checkpoint/权重。

## 迁移摘要

| 项目 | 内容 |
|---|---|
| 迁移文件数 | 4,158 |
| 总大小 | 27,444,811 bytes（约 26.17 MiB） |
| 目标根目录 | `/home/cgv841/ybj/SALT-VI/logs/raw/` |
| 总表更新 | 24 行已改为新路径；126 行仍标记为未迁移或缺失；1680 行未记录 |
| 完整清单 | `/home/cgv841/ybj/SALT-VI/runtime/raw_log_migration_20260803.json` |
| 表格审计 | `/home/cgv841/ybj/SALT-VI/runtime/raw_log_table_update_20260803.json` |

## 按来源统计

| 来源 | 文件数 |
|---|---:|
| `PMT-SYSU/outputs` | 575 |
| `TVI-LFM/logs` | 94 |
| `TVI-LFM/reports` | 2,558 |
| `experiments` | 931 |

## 说明

- `log_source` 能在迁移清单中精确对应的记录已更新为 `SALT-VI/logs/raw/...`；旧值保存在新增的 `legacy_log_source` 列。
- 不在迁移清单中的绝对路径（包括 ybj 外部路径）没有被擅自删除或改写，只标记为“未迁移或缺失”。
- `TVI-LFM/reports/experiment_registry`、`caption_quality`、`archive` 等总表/证据归档子树未作为原始日志搬动对象，保留在原位置。
- 迁移前后文件哈希与 inode 已记录在清单中，可用于完整性核验。

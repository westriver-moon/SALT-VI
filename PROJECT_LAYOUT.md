# SALT-VI 项目布局

```text
SALT-VI/
├─ src/salt_vi/             # 自有模型、损失、数据适配和训练组件
├─ configs/                 # 可复现实验 YAML；不放数据实体
├─ scripts/                 # 训练、评估、归档和验证入口
├─ pasd_offline/            # 独立的 caption 驱动 PASD 离线数据生成模块
├─ semantic_imagination/    # VLM 观测、想象聚类、经验权重与 PASD records 插件
├─ data_sources/            # 公共数据目录和派生数据的路径/哈希清单
├─ experiments/             # 按 experiment_id 保存运行元数据
├─ checkpoints/             # 仅保存明确保留的项目权重
├─ pretrained/              # 外部初始化权重的登记和校验信息
├─ logs/                    # 训练与评估日志
├─ reports/                 # 结果总表、原始指标和复现证据
├─ paper/                   # 论文图表、表格、文字素材
├─ runtime/                 # 环境、迁移、机器和运行记录
└─ vendor/                  # 上游代码来源与许可证说明
```

# 运行入口

从项目根目录执行：

```bash
python scripts/train.py --config_select configs/default.yaml
```

脚本会将 `src/` 加入 Python 路径，并调用 `salt_vi.entrypoints.train`。

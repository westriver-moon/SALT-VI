# SALT-VI 代码释放迁移方案

## 目标

将历史实验所需的实现释放为 SALT-VI 自己维护的代码包。运行时只依赖 `src/salt_vi`，不建立旧实现到新实现的中间映射层。

## 阶段

1. **盘点与冻结**：记录旧入口、配置、代码差异和运行证据；旧源目录只读。
2. **代码释放**：将数据、训练引擎、模型、优化器、工具、配置加载器和基线组件整理进 `src/salt_vi`，并建立 `scripts/train.py`。
3. **引用重写**：将新项目内脚本、配置、总表和运行清单直接改为新项目路径。
4. **路径规范化**：将非权重资产的历史目录改为中性名称；不移动 checkpoint 或 pretrained 权重。
5. **验证与交付**：执行语法检查、文件完整性检查、日志路径检查和旧路径扫描。

## 最终运行关系

```text
scripts/train.py
    -> src/salt_vi/entrypoints/train.py
    -> src/salt_vi/{data,engine,models,optim,utils,config}
    -> configs/
    -> checkpoints/、logs/
```

## 不变约束

- 不修改或删除 `/home/cgv841/ybj` 之外的任何内容。
- 不移动 checkpoint 和 pretrained 权重。
- 历史日志作为证据保存，不作为新训练入口。

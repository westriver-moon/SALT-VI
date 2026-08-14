# Semantic Imagination v2 迁移清单

迁移日期：2026-08-14

## Git 迁移

- 源仓库：`/home/lab929/ybj/SALT-VI`
- 源提交：`5f58fc86949eb8edeba9a103908d441a493eb72c`
- 源状态：想象力组件修改位于 `main` 未提交工作区，不存在独立旧想象力分支；
- 新分支：`codex/semantic-imagination-v2`
- 新 worktree：`/home/lab929/ybj/autoresearch-v2/worktrees/semantic-imagination-v2-20260814/w1`

## 代码与报告

- `README.md`、`MATHEMATICAL_SPEC.md` 和问题报告已迁移并按v2语义更新；
- 原单文件插件拆为 taxonomy、schema、validation、sampling、clustering 和 PASD adapter；
- `plugin.py` 保留旧公共导入路径；
- InternVL实验入口迁移至 `experiments/run_internvl_sampling.py`；
- 旧前缀分析和manifest审计入口迁移至 `experiments/`。

## 历史实验产物

`reports/legacy_20260813/` 保存以下实验的 summary、records、manifest、PASD record、
日志和前缀比较：

- `stratified_n128_nested`；
- `stratified_n128_canonical`；
- `canonical_n512_nested`；
- `canonical_n512_audited`。

共32个文件，1.4 MiB；按排序文件SHA-256再次汇总的校验值为：

```text
2695310dc2a350db4e3239c547191204d14c1879cd5c63448619df6b45079c44
```

随机扰动JPEG和复制的输入`source.jpg`属于可由seed重新生成的缓存，没有进入Git
报告目录；原始缓存继续保留在服务器源实验目录，未被删除。v2真实模型smoke
test保存在`reports/20260814_v2_smoke_final/`。

## 保护项

`reports/experiment_registry/experiment_registry.csv`是源工作区中与本次任务无关的
既有修改，迁移和合并过程中必须原样保留，不得纳入本次提交。

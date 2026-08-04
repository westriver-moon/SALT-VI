# SALT-VI migrated evidence document

> Source document ID: `source_core:reports/experiment_registry/historical_archives/ybj2_sysu_multiseed_20260717/reproduction/sysu_multiseed/SCHEDULER_DESIGN.md`  
> Original SHA-256: `d52ca1e8f9d4d419dcfc3a1f54ad53d49249ca905ec42af7fa7ab79b4f3f53da`  
> This is read-only experiment evidence, not an active runtime instruction.

# SALT-VI SYSU 四种子后台调度器

## 目标

在不占用交互式终端或 tmux 等会话窗口的情况下，按每张 RTX 3090 的实际空闲状态调度，然后严格按既定顺序完成：

1. 全部启动门槛检查；
2. 官方 `SYSU_SALT-VI.pth` 的 MER 推理预飞；
3. seed 1 单 epoch smoke test；
4. seed 1–4 固定映射到物理 GPU 0–3 的正式训练；
5. 每个训练进程自行完成最佳 checkpoint 的 MER 测试。

## 运行模型

- 调度器使用 `nohup + setsid` 脱离 SSH 和控制终端运行；无需用户级 systemd linger。
- 等待阶段不创建 tmux 会话，也不保持 SSH 前台窗口。
- 正式种子分别使用独立的 `nohup + setsid` 进程。
- `/home/cgv841/ybj2/reproduction/sysu_multiseed/scheduler/scheduler.lock` 使用 `flock` 保证单实例。
- 默认每 60 秒轮询一次；没有最长等待时间。

## GPU 释放条件

单张卡满足以下条件时即视为可调度：

- `nvidia-smi` 未报告计算进程；
- 显存占用低于 500 MiB；
- GPU 利用率低于 10%。

两级预飞依次使用当时任意一张空闲卡。预飞全部通过后，seed 仍固定映射为 `1→GPU0、2→GPU1、3→GPU2、4→GPU3`：某张卡先空闲就立即启动对应 seed；同一轮有多张卡空闲就并行启动多个 seed。调度器不终止、不抢占其他用户的进程，也不自动换卡。

## 状态机

```text
waiting_for_preflight_gpu
  -> official_mer_preflight
  -> smoke_preflight
  -> scheduling_seeds
  -> running
  -> complete
```

任一预飞或正式种子失败时进入 `failed`，不会自动改变协议、batch size、seed 或 attempt。重试仍须使用新的 `attempt_02`。

## 可观察性与控制

- 原子状态：`scheduler/status.json`
- 调度日志：`scheduler/scheduler.log`
- 守护进程 PID：`scheduler/daemon.pid`
- 实时日志：`tail -f /home/cgv841/ybj2/reproduction/sysu_multiseed/scheduler/scheduler.log`
- 仅停止调度器：`kill -TERM $(cat scheduler/daemon.pid)`

停止调度器不会自动终止已经启动的正式训练种子，避免意外杀死实验。

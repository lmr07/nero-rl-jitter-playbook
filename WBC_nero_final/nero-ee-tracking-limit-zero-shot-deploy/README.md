# Nero 七关节末端跟踪限位版实物部署包

此目录为键盘控制版本。自动采样版本单独位于 `nero-ee-tracking-limit-auto-zero-shot-deploy/`；此目录的 `run.sh` 不会自动更换末端目标。

本包固定训练 `Oct27_23-25-59_limit_continuation_10000/model_10000.pt`，导出策略不再训练或修改权重。`assets/manifest.json` 记录 checkpoint、训练配置、URDF、辨识系数和导出文件的 SHA256。旧稳定版实物包保留在 `nero-ee-tracking-zero-shot-deploy/`。

## 控制一致性

训练和部署均使用 50 Hz 七关节绝对位置动作、16×39 观测历史、相同的工具正运动学、球坐标中心、初始 pitch 过渡与 20 ms 关节目标限速。实物请求的 Kp/Kd 用 `nero_identification/datasets/nero_eq20/run_20261023_103746/gain_delay_fit_with_frequency.json` 的 `command_only.train.coeff` 从训练 Kp/Kd 逐轴换算；每次启动先做 5 秒增益渐变和 5 秒实测姿态保持。

新训练的目标角限位余量为 J1–J6 各 `0.16 rad`、J7 `0.30 rad`。部署 actor 的候选目标和最终下发目标均执行同一余量，J7 目标上限为 `1.2707963 rad`。实测反馈继续使用 `q + 0.15*dq` 和距 URDF 极限 `0.08 rad` 的预测保护。安全保护触发会停止控制并执行原有关闭流程。

增益渐变后的 actor 启动检查使用实测关节角相对训练 home 的误差（每轴不超过 `0.10 rad`）、实测速度（每轴不超过 `0.08 rad/s`）、MIT 目标与实测角的偏差（每轴不超过反馈包络且不超过 `0.20 rad`），以及训练目标余量。增益换算会改变维持相同姿态所需的 MIT 目标，因此不再将该目标与训练初始 PD 指令比较。2026-10-28 的一次启动中，J4 目标差 `0.1512 rad` 刚好超过旧的固定 `0.15 rad` 检查，但实测 home 最大误差为 `0.055 rad`、速度为零、目标与实测角最大差为 `0.084 rad`；actor 当时没有启动，关闭验证通过。新检查已用该次日志离线回放验证。

## 已完成验证

- 精确 checkpoint 的 20 秒 CPU PhysX 回放：0 次重置、0 次预测限位事件、0 次目标余量违规；平均末端位置误差 `0.0716 m`，保持阶段 `0.0819 m`。
- TorchScript 导出与训练 actor 对 64 组观测的最大输出差为 0。
- 实物代码与训练仿真 400 步对照，含一次 5 cm 目标移动：观测最大差 `7.28e-4`，目标位置差 `1.08e-7 m`，关节下发目标差 `1.07e-7 rad`。
- 2026-10-28 只读检查：`can_nero` 1 Mbps、七轴反馈完整且连续；检查时关节角为 `[0.153, -0.040, -0.105, 0.103, 0.062, 0.083, -0.069] rad`，速度均为零。只读检查没有使能电机或发送目标。

这些离线结果不能代替新策略在实物上的实测。旧稳定版策略曾在 J7 达到 `q=1.321 rad`、`dq=1.205 rad/s` 时触发预测限位停机；本包是针对该问题重新训练的独立版本。

## 启动

在现场看护并可立即使用急停时，从仓库根目录先做不运行 actor 的增益保持测试：

```bash
env -u LD_LIBRARY_PATH bash nero-ee-tracking-limit-zero-shot-deploy/run.sh --act --gain-hold-only
```

确认该测试通过后，再单独启动 5 秒策略测试：

```bash
env -u LD_LIBRARY_PATH bash nero-ee-tracking-limit-zero-shot-deploy/run.sh --act --seconds 5
```

I/K 和方向键每次移动末端目标 5 cm；Q、Esc 或 Ctrl+C 停止。实物日志写入本包 `logs/<时间>/`。程序每次启动都会重新执行预定位和增益保持测试。

只读检查命令：

```bash
env -u LD_LIBRARY_PATH bash nero-ee-tracking-limit-zero-shot-deploy/run.sh --read-only --seconds 3
```

重新构建当前固定包的离线入口为 `bash nero-ee-tracking-limit-zero-shot-deploy/prepare_latest.sh`。

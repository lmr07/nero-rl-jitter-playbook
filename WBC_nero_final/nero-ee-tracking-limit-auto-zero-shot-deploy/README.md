# Nero 七关节末端跟踪自动采样实物部署包

本包独立于键盘控制版 `nero-ee-tracking-limit-zero-shot-deploy/`，使用相同的 `Oct27_23-25-59_limit_continuation_10000/model_10000.pt` 权重、逐轴辨识增益、关节目标余量和硬件保护。两个目录各自保存日志和部署文件。

actor 启动后先在训练初始末端位置过渡到水平姿态。初始运动时间从 `1–3 s` 均匀采样，保持时间从 `2–5 s` 均匀采样。此后每段从训练配置的球坐标范围独立均匀采样：`r=[0.4,0.7] m`、`pitch=[-π/5,2π/5]`、`yaw=[-3π/5,3π/5]`。候选目标沿球坐标路径做 10 点避碰检查，最多尝试 10 次；若全部被拒绝，则保持上一目标。每个新目标以 `1–3 s` 插值，随后保持 `2–5 s`，如此循环。末端姿态沿目标方向保持水平。Q、Esc 或 Ctrl+C 停止策略。

现场有人看护且急停可用时，从仓库根目录启动 60 秒自动采样策略：

```bash
env -u LD_LIBRARY_PATH bash nero-ee-tracking-limit-auto-zero-shot-deploy/run.sh --act --seconds 60
```

运行前仍会执行预定位、5 秒辨识增益渐变和 5 秒实测姿态保持测试，actor 在检查通过后才接管。策略期间继续执行逐轴目标限幅、限速、反馈包络及预测关节限位保护。控制日志与每次采样的终点写入本包 `logs/<时间>/`。

不打开 CAN 的自检：

```bash
env -u LD_LIBRARY_PATH bash nero-ee-tracking-limit-auto-zero-shot-deploy/run.sh --seconds 60
```

独立包的离线测试：

```bash
python3 -m unittest discover -s nero-ee-tracking-limit-auto-zero-shot-deploy/tests -p test_offline.py
```

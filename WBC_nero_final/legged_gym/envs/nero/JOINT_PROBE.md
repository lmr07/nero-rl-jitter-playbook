# Nero 逐关节固定 PD 策略

新增独立任务 `nero_joint`，原 `nero_fixed` 训练和部署目录不变。每个任务训练一个 192→1 actor，不能把原来的 624→7 checkpoint 接到这里，也不能把这个 actor 直接放入旧部署入口。

## 训练设定

- Isaac Gym PhysX 原生位置驱动：stiffness=48、damping=1.5，v_des=0。
- 物理步长 0.005 s、substeps=1、decimation=4，策略周期 0.02 s。
- 绝对位置动作：`q_des = lower + (clip(action,-1,1)+1)*(upper-lower)/2`。不累加，不使用 v4 的 0.012 rad 增量限幅；动作映射覆盖真实关节限位，给 70% 范围内的参考目标留出重力补偿余量。
- 参考端点在下表范围均匀采样。线性插值、保持 4–7 s、重采样；运动时间为 `ceil(abs(end-start)/(0.5*v_max*0.02))` 个策略周期，因此插值速度略低于或等于半最大速度。实际关节能否跟上由策略和动力学决定。
- 回合 24 s，活动关节初始角在同一范围内均匀采样。
- 其他六轴和夹爪转换为 URDF fixed joint，按 `R_origin * R_axis(q_locked)` 修改姿态，保留质量和惯量。每个 actor **只有一个活动自由度**。
- 默认随机采样 128 组锁定姿态，分配给 1024 个环境；可用 `--pose-bank-size 1024` 增加多样性。姿态在环境创建时采样，回合内和回合重置时保持不变。主动轴初始角和参考目标每回合重采样。姿态库保存在运行配置中。
- 不添加地面，不启用自碰撞；这是自由空间执行器对照实验。任意采样的仿真姿态不代表已经通过实物碰撞检查。
- 不随机化 PD、质量、摩擦、延迟，不进行课程/阶段切换。保留 URDF 惯量、重力、配置中的 armature；额外刚体线性/角阻尼设为零。

|关节|参考/锁定采样范围 rad|参考速度上限 rad/s|力矩上限 Nm|
|---|---:|---:|---:|
|J1|±1.893682|1.5|24|
|J2|±1.218|1.5|24|
|J3|±1.925|1.5|16|
|J4|±0.707|2|16|
|J5|±1.925|2|8|
|J6|±0.511|2|8|
|J7|±1.09955741|2|8|

范围统一为 `±0.7*min(-lower,upper)`，包含用户确认的 J4 对称截取；J6 同样处理。速度/力矩采用现有 Nero 配置中来自用户参考 URDF 的较低限制，不采用原始 nero-fixed.urdf 中偏大的 effort/velocity，也不将这些限制当作已测量的电机能力。

辨识来源：`analysis/nero_newpd_sysid/newpd_actuator_fit.json`。七轴延迟 `[9,9,12,12,9,10,11] ms`，均值 **10.285714 ms**。在命令端施加一次该集总延迟；5 ms 网格用 2/3 tick 的线性分数延迟近似，权重约 `[0.942857,0.057143]`。这是离散近似，带有少量高频衰减，不声称是精确连续纯延迟。不给反馈再叠加同一延迟，也不额外叠加一次 20 ms 传输延迟；辨识值并未独立分离 CAN、执行和反馈延迟。

Actor 每帧观测为 `[q,dq,q_ref,dq_ref,previous_clipped_action,locked_q1..q7]`，共12维、16帧。锁定姿态向量的活动轴槽恒为零；其真实角通过 q 输入。Critic 另见延迟命令和命令队列。奖励包含角度跟踪、速度跟踪、保持时速度、动作变化率、力矩和速度超限项。没有末端误差或多关节 actor 补偿。

## 启动

在工程根目录：

```bash
# 先训练 J5（可将 5 改成 1..7）
bash legged_gym/scripts/train_nero_joint.sh \
  --joint 5 --num_envs 1024 --max_iterations 5000

# 顺序训练七个独立策略；前一个完成后启动下一个
for joint in 1 2 3 4 5 6 7; do
  bash legged_gym/scripts/train_nero_joint.sh \
    --joint "$joint" --num_envs 1024 --max_iterations 5000 || break
done

# 本关节最新 checkpoint 可视化
bash legged_gym/scripts/play_nero_joint.sh \
  --joint 5 --num_envs 1 --checkpoint -1

# 未见过的锁定姿态上评估，并导出 actor（不连接 CAN）
bash legged_gym/scripts/play_nero_joint.sh \
  --joint 5 --headless --num_envs 128 --held-out-pose-seed 2026 \
  --steps 6000 --export
```

运行输出分别在 `logs/nero_joint_J1` 到 `logs/nero_joint_J7`。每个训练目录包含 checkpoint、`joint_config.json`、`pd_candidates_OFFLINE_ONLY.json`。恢复训练用 `--resume --load_run <运行目录名> --checkpoint -1`，同时保留原来的 `--joint`、姿态种子、姿态库大小；入口检查动作/PD/延迟 ABI。正式训练应从头启动，不继承原7轴策略。

## 后续实物 PD 对照

本次只提供训练、仿真回放、策略导出和统一候选表，**未接入 CAN 或执行实物扫描**。

候选以每类电机之前的 α、β 均值为中心，各自乘 `[0.8,1,1.25]`，形成相同的 3×3 网格；硬件命令为 `Kp=48/alpha, Kd=1.5/beta`。J1/J2 共用候选表；J3/J4 共用候选表；J5、J6、J7 各自独立。表中每一组候选必须在同型号所有关节上评估，最后选一个共同结果，不能给 J1 和 J2 分别选不同候选。

该表不是批准部署的 PD：之前 J2 在低增益下出现过保持下垂，且 SDK 电流—力矩尺度尚未绝对校准。实物测试需要独立单轴入口，其他轴采用固定目标保持并记录漂移，不能把 PD 保持声称为机械锁死。每个候选应使用同一组姿态、目标时间序列、初始状态和重复次数，比较保持抖动、跟踪误差、过冲、命令反转、饱和，以及 1–10 Hz 响应；同型号的候选应按两关节中的较差表现筛选。仅最平稳不足以确定唯一真实 α、β。

## 验证记录

2026-10-20：5项离线测试通过（对称范围、七种 fixed URDF 的变换/质量/单自由度、延迟来源、同型号候选一致性、排除导出文件的 checkpoint 选择）。J1–J7 各进行 4 环境、2 锁定姿态、160 策略周期的 GPU PhysX 检查，包含回合重置、有限观测/奖励、目标范围/速度、原生 PD 属性。J5 的 32 环境短程 PPO 更新、回放和 actor 导出通过。短程 checkpoint 仅验证程序链路，不代表收敛或可部署。

## Smooth-hold reward revision (2026-10-21)

Fresh training now writes `logs/nero_joint_smooth_J1` ... `J7`.
The action/observation ABI, 200 Hz simulation, 50 Hz policy, fixed PD 48/1.5,
mean identified delay, full joint ranges, 0.5*vmax reference speed and
0.99*vmax issued-command slew cap are unchanged.

Fixed reward weights in `NeroJointCfg.rewards`:
- holding velocity: 0.30 (previously 0.05; additional velocity tracking is 0.04);
- issued-command second difference divided by allowed command step, squared: 0.05;
- normalized requested-minus-issued position command, squared: 0.50.
All terms are negative and multiplied by policy dt. The gap is measured after
clipping the actor to [-1,1], before versus after the position slew limiter.
These are initial tuning values, not validated stability guarantees. There is
no new hard acceleration cap or deployment-side action transformation.

Episode metrics now include holding speed squared, normalized command second
difference squared, adapter gap squared and command reversal fraction (both
successive steps must exceed 1e-4 rad). Reset clears command-difference history.
Resume checks reward configuration; start fresh rather than resuming the old
oscillatory policies. Old play manifests use their original reward behavior.

Train all joints sequentially (stop on failure):
```bash
for j in 1 2 3 4 5 6 7; do
  bash legged_gym/scripts/train_nero_joint.sh --joint "$j" --num_envs 1024 --max_iterations 5000 || break
done
```
Replay a new run with `bash legged_gym/scripts/play_nero_joint.sh --joint 6 --smooth --checkpoint -1 --num_envs 1`.

Validation: J6 GPU smoke test (16 environments, 160 policy steps), including
reward numerical equality, command slew bound, reset history, observation
finiteness and PD properties; five existing joint-probe tests passed.
# A2nero Stage-1-aligned single-joint training (new independent runs)

`train_nero_joint.sh` now starts `nero_joint_a2stage1_J1` through `J7` from scratch.
New runs use Isaac Gym/PhysX native position PD at Kp=10, Kd=0.8 and a
5 ms physics / 20 ms policy tick.  The action ABI is different from every old
`nero_joint_*` checkpoint: raw action maps to `q_default + 0.25 * action`, then
the physical position target is clipped to the URDF joint limits and limited
to `0.99 * vmax * 0.02` radians per policy tick.  The observation contains
default-relative position, `0.05 * velocity`, the issued A2-style action,
reference position/velocity, and the six fixed joint coordinates.  Only
proprioception receives the same Stage-1 Gaussian noise as A2nero.

The transferable A2nero Stage-1 settings are: motor-strength multiplier
`[0.92, 1.08]` (applied to both native drive gains), `[0, 0.12] kg` payload
at `nero_link7`, no added command delay, 2-5 s target hold, 20 s episodes,
4096 environments, `512/256/128` actor and critic, 32-D latent, initial
action standard deviation 1.0, and the inherited A2nero PPO optimizer.
The default training budget is 5000 iterations per joint.
One-joint tracking uses the A2nero end-effector exponential reward's `4.0`
scale and `0.4` error scale, the A2nero arm action-rate/velocity/acceleration
regularizers, and its soft position-limit cost.  A dense one-joint position
error term remains necessary over full URDF travel.  Terrain, gait, body
orientation and multi-DOF end-effector rewards do not exist in this fixed-base
one-DOF task.  The earlier full-range goal sampler and 0.5*vmax reference
speed remain task requirements.  The earlier identified 10.3 ms delay is
intentionally removed in this A2nero Stage-1 comparison; it is not a claim
that real Nero has zero latency.

Start one fresh run with:

```bash
bash legged_gym/scripts/train_nero_joint.sh --joint 1
```

Replay one of these new runs with:

```bash
bash legged_gym/scripts/play_nero_joint.sh --joint 1 --a2stage1 --num_envs 1
```

For a simulation-only replay without observation noise, append
`--no-obs-noise`; the checkpoint and its saved training configuration are not
modified.

Older runs remain under their old experiment names and use their saved action
and observation ABI with `--legacy`, `--smooth`, or the default `--slew` path.
Earlier A2nero Stage-1 checkpoints also retain their saved Kp=48, Kd=1.5
configuration when replayed explicitly; new training runs are named
`J*_a2stage1_arm_physx_pd10_0p8`.

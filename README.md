# Nero 强化学习策略实物抖动：辨识、定位与修复

这份案例记录 2026 年 10 月在七关节 Nero 末端跟踪任务中，从数据采集到策略重训、限位保护和实物部署的排查过程。仓库同时收录了最终训练与实物部署代码、两份 checkpoint、导出策略、辨识结果和机器人模型，集中放在 [`WBC_nero_final/`](WBC_nero_final/README.md)。本文给出流程、关键参数和证据边界。

## 一页流程

```text
核对固件/关节顺序/坐标/附件/控制周期
  → actor 关闭：固定目标保持与小阶跃，记录原始 CAN 时间戳
  → 分轴采集静态、恒速、多频和留出轨迹
  → 辨识有效 Kp/Kd 乘数与命令—反馈对齐时延
  → actor 开启：分别记录原始请求、限速后目标、CAN 回显和关节反馈
  → 同姿态固定目标/冻结反馈/冻结动作历史做因果对照
  → 在训练中约束原始请求、目标限速、保持阶段关节运动和关节限位
  → 离线回放、导出一致性、增益渐变保持、短时策略、完整运行
```

关键原则：末端不动不代表关节不动。7 自由度机械臂跟踪 6 维末端位姿时，策略可能在相近末端位姿之间反复切换关节组合。要同时看末端误差、七轴角度/速度、原始关节请求、实际发送目标和 CAN 回显。

## 目录

- [01 数据采集与辨识](docs/01-identification.md)：实验设计、回归式、乘数与时延的解释。
- [02 抖动定位与策略修复](docs/02-policy-jitter.md)：如何区分 PD/时序与 actor 自主反向；最终训练改动。
- [03 验证与迁移检查](docs/03-validation.md)：逐层验收、限位事件与部署注意事项。
- [`data/gain-delay-20261023.csv`](data/gain-delay-20261023.csv)：最终部署使用的逐轴拟合摘要。
- [`WBC_nero_final/`](WBC_nero_final/README.md)：可迁移训练与键盘/自动采样部署包。

## 下载和核对

```bash
git clone https://github.com/lmr07/nero-rl-jitter-playbook.git
cd nero-rl-jitter-playbook/WBC_nero_final
bash verify_bundle.sh
```

迁移包约 143 MB，其中包含 checkpoint 和网格模型；GitHub 网页上的 **Download ZIP** 也可下载，但 Git 克隆便于后续更新。安装依赖、仿真可视化、离线部署测试与实物入口见[迁移包说明](WBC_nero_final/README.md)。克隆和校验不会连接 CAN 或使能电机。

## 此次结果及范围

最终成功运行的是 `nero_ee_tracking_limit_v3/Oct27_23-25-59_limit_continuation_10000/model_10000.pt`，它从稳定版 `Oct27_16-19-23_pitch_ramp_slew_hold_gravity/model_10000.pt` 继续训练 10000 轮。训练 PD 为 J1/J2 `40/1`、J3–J7 `20/1`；实物 MIT 增益按 2026-10-23 的有效乘数逐轴换算。训练采用 50 Hz 策略和 5 ms PhysX 子步，每轴目标限速，**未启用显式逐轴通信时延随机化**。因此辨识出的毫秒数没有被直接写成这版成功策略的训练时延参数。

2026-10-26 的较早策略在固定末端参考下出现约 12.5 Hz 的目标周期。离线固定目标时运动几乎消失；固定物理反馈后周期仍可出现，进一步冻结当前帧 previous-action 通道后该局部周期消失。最终版本通过惩罚限速前请求变化及请求与下发目标间的差距、限制目标速度、稳定保持姿态、处理初始姿态/重力平衡，并增加限位余量和预测惩罚，获得了用户报告的良好实物效果。现有证据支持这套组合在当时硬件和任务上的效果，不能把改善归因于单个参数，也不代表所有姿态、负载都已验证。

## 代码入口与溯源

本仓库 `WBC_nero_final/` 内：

| 用途 | 包内路径 |
|---|---|
| 稳定版训练配置及环境 | `legged_gym/envs/nero/nero_ee_tracking_stable_config.py`、`nero_ee_tracking_stable.py` |
| 限位版训练配置及环境 | `legged_gym/envs/nero/nero_ee_tracking_limit_config.py`、`nero_ee_tracking_limit.py` |
| 限位版训练入口 | `legged_gym/scripts/train_nero_ee_tracking_limit.sh` |
| 键盘部署 | `nero-ee-tracking-limit-zero-shot-deploy/` |
| 自动采样部署 | `nero-ee-tracking-limit-auto-zero-shot-deploy/` |
| 辨识摘要和 checkpoint | `identification/`、`logs/` |

辨识采集与完整拟合代码原位于本机 `nero_identification/`，原始 CAN 数据没有收录到本公开案例。确切训练配置请以该 checkpoint 旁的 `nero_ee_tracking_config.json` 为准；部署包 `assets/manifest.json` 记录了关键输入和导出的 SHA256。

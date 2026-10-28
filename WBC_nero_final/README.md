# WBC Nero final: seven-joint end-effector tracking

This folder is a movable snapshot of the training code and the two real-Nero deployment modes for `nero_ee_tracking_limit_v3`, run `Oct27_23-25-59_limit_continuation_10000`, checkpoint `model_10000.pt`. It includes the earlier stable `model_10000.pt` used only to initialize a fresh training continuation. The two deployment packages contain the same exported final policy and identified gain profile. Their `assets/manifest.json` files record exact SHA256 hashes. No training or CAN execution is started by copying this folder.

## Contents

- `legged_gym/` and `nero_joint_contract.py`: training environment, reward and limit configuration, PPO code, training and visualization scripts.
- `resources/robots/a2nero/`: Nero URDF and meshes needed by Isaac Gym.
- `logs/nero_ee_tracking_stable_v2/.../model_10000.pt`: initial network weights for a fresh limit-aware continuation.
- `logs/nero_ee_tracking_limit_v3/.../model_10000.pt`: final trained checkpoint and saved configuration.
- `identification/gain_delay_fit_with_frequency.json`: the exact seven-axis gain identification input used for the deployment export.
- `nero-ee-tracking-limit-zero-shot-deploy/`: manual I/K/arrow-key end-effector goals.
- `nero-ee-tracking-limit-auto-zero-shot-deploy/`: automatic spherical goal sampling, motion and hold cycles.

Historical absolute source paths inside saved checkpoint metadata and asset manifests are provenance only. The runnable scripts and re-export paths resolve files from this folder. Training and deployment can use separate Python environments.

## New-computer setup

Use Linux with SocketCAN for real hardware. Isaac Gym with GPU PhysX, its NVIDIA/CUDA driver stack, PyTorch, NumPy, SciPy and TensorBoard are required for training or visualization. Isaac Gym is a separately installed SDK and is not copied here. The original training environment used Python 3.8.20, PyTorch 2.3.1+cu121, NumPy 1.19.5 and SciPy 1.10.1. Set `NERO_PYTHON` or `NERO_TRAIN_PYTHON` to the interpreter with that environment.

For the real-Nero packages, install PyTorch, NumPy, SciPy and `python-can` into a Linux Python environment. The original deployment environment used Python 3.10.20, PyTorch 2.5.1+cu121, NumPy 2.2.6, SciPy 1.15.3 and python-can 4.6.1. The needed `pyAgxArm` source is bundled in each deployment package. Set `NERO_DEPLOY_PYTHON` to that interpreter. The `can_nero` interface must be configured for the physical controller; the packages never configure CAN automatically.

From this folder, verify copied files first:

```bash
bash verify_bundle.sh
```

Offline policy checks (no CAN opened):

```bash
NERO_DEPLOY_PYTHON=/path/to/deploy/python bash nero-ee-tracking-limit-zero-shot-deploy/run.sh --seconds 5
NERO_DEPLOY_PYTHON=/path/to/deploy/python bash nero-ee-tracking-limit-auto-zero-shot-deploy/run.sh --seconds 60
```

## Training and visualization

The final checkpoint is ready to visualize:

```bash
NERO_PYTHON=/path/to/train/python bash legged_gym/scripts/play_nero_ee_tracking.sh --limit --load_run Oct27_23-25-59_limit_continuation_10000 --checkpoint 10000 --no-obs-noise
```

To start a **new** 10000-iteration continuation from the bundled stable policy, use the bundled training entry point. It loads only network weights and starts a new optimizer at iteration zero:

```bash
NERO_PYTHON=/path/to/train/python bash legged_gym/scripts/train_nero_ee_tracking_limit.sh --num_envs 2048 --max_iterations 10000
```

The saved final checkpoint is in `logs/nero_ee_tracking_limit_v3/Oct27_23-25-59_limit_continuation_10000/model_10000.pt`. A new run gets a new log directory. See the environment configuration in `legged_gym/envs/nero/nero_ee_tracking_limit_config.py` for target margins and limit rewards.

## Real Nero

Both modes perform prepositioning, a five-second identified-gain ramp, a five-second measured-pose hold check, then three-second policy takeover. Existing joint target margins, feedback envelope, predictive joint-limit guard, stop keys and verified disable remain active. Start only with a local operator able to use emergency stop.

Manual goals:

```bash
NERO_DEPLOY_PYTHON=/path/to/deploy/python bash nero-ee-tracking-limit-zero-shot-deploy/run.sh --act --seconds 60
```

Automatic sampled goals:

```bash
NERO_DEPLOY_PYTHON=/path/to/deploy/python bash nero-ee-tracking-limit-auto-zero-shot-deploy/run.sh --act --seconds 60
```

The automatic mode samples `r=[0.4,0.7] m`, `pitch=[-pi/5,2pi/5]`, `yaw=[-3pi/5,3pi/5]`; it checks the spherical path at ten points, moves for 1-3 seconds and holds for 2-5 seconds. Q, Esc and Ctrl+C stop the policy. Each package writes its own runtime logs under `logs/<timestamp>/`. Before first actuation on a new computer, use `--read-only --seconds 3` to check all seven feedback streams and CAN state. Each package has its own README and offline tests.

To re-export a package from the included final checkpoint and identification file, run that package's `prepare_latest.sh` with `NERO_TRAIN_PYTHON` and `NERO_DEPLOY_PYTHON` set. This uses Isaac Gym for the CPU simulation gate.

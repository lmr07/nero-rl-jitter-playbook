# Nero seven-axis WBC-frame cycle

Run from the repository root:

```bash
env -u LD_LIBRARY_PATH bash legged_gym/scripts/train_nero_seven_cycle.sh \
  --num_envs 2048 --max_iterations 20000 --sim_device cuda:0 --rl_device cuda:0
```

Visualize the newest numeric checkpoint from the newest seven-cycle run:

```bash
env -u LD_LIBRARY_PATH bash legged_gym/scripts/play_nero_seven_cycle.sh \
  --load_run -1 --checkpoint -1 --num_envs 1
```

The viewer draws the interpolated goal in green, the segment endpoint in
yellow, and the measured tool in blue. Add `--no-obs-noise` for a clean
simulation replay, or `--steps 6000` to extend the default 3000 policy steps.

This `nero_seven_cycle` task does not reuse the 7-axis `nero_fixed` action ABI
or any earlier checkpoint. The seven actor outputs are absolute joint-position
offsets `q_des = q_WBC_home + 0.25 * clipped_action`. Each sent target is also
limited to `[1.5, 1.5, 1.5, 2, 2, 2, 2]` rad/s at the 20 ms policy tick. Isaac Gym PhysX
position drives use Kp=20 and Kd=1 for J1–J7, with the existing conservative
24/24/16/16/8/8/8 Nm and 3/3/3/4/4/4/4 rad/s asset metadata. Simulation is
200 Hz, policy 50 Hz. The WBC arm home is `[0, -1, 0, 1.8, 0, 0, 0.3]` rad.
The WBC 16-frame history and actor/critic architecture remain. The reward
combines end-effector tracking and acceleration terms with a dense position
error term and proximity-weighted hold velocity, sent-target rate and reversal
penalties. Only home holds have a joint-posture anchor. End-effector orientation
tracking and global joint deviation from zero are not rewarded. The A2 legs
and their observations/rewards are absent because the Nero base is fixed.

The spherical target `(radius, pitch, yaw)` uses the original WBC `sphere2cart`
function. Its origin is the original A2 goal center: the Nero mounting flange
minus 0.142 m in Z. World axes align with the original A2 base at zero yaw;
the Nero URDF retains its original 3.141 rad mounting rotation. The two fixed
position targets are `(0.5, 0, pi/4)` and `(0.6, pi/3, -pi/4)`. Their
mount-relative Cartesian coordinates are approximately `(0.3536, 0.3536,
-0.1420)` m and `(0.2121, -0.2121, 0.3776)` m. WBC home FK is approximately
`(0.0915, 0, 0.5744)` m from the mount; its spherical coordinates are
approximately `(0.7222, 1.4438, 0)`. Its position Jacobian is full rank,
unlike the previous all-zero home.

Each 27 s episode holds home for 3 s, then follows home → P1 → home → P2 →
home. Each leg interpolates **in spherical coordinates** for 3 s and holds its
endpoint for 3 s. P1 and P2 use the WBC zero-RPY-in-goal-heading orientation;
home uses its FK orientation as an observation target, not a reward. The two
fixed target coordinates and timing are unchanged. Training environments start
at one of four coherent phases: home-before-P1, P1-hold, home-before-P2,
P2-hold. P1/P2 starts use position-only URDF IK anchors, never hardware
commands. The P1 reset drive command uses J2=-0.80 rad to compensate the
Kp=20 native-drive gravity sag seen with a fixed target; the physical reset
joint angle remains J2=-1.49 rad. Phase assignment rotates on reset;
one-environment visualization
always starts at home. Do not offset a goal timer without its joint state.

The fixed provisional observation noise is 2.1 mrad position and 0.07 rad/s
velocity standard deviation, instead of WBC's roughly 1.05 rad/s velocity
noise. PPO uses 48 steps/environment/update, initial learning rate 2e-4
(adaptive bounds 1e-5 to 3e-4), four update epochs and entropy coefficient
0.003. The 20k default iterations preserve the old 40k x 24 rollout sample
budget. TensorBoard also records `Loss/kl_mean` to diagnose adaptive learning
rate reductions. Start a new run; old checkpoints used the zero-joint home and
old reward.

The 2026-10-23 identification supplied effective delays `[5.5, 7, 8, 7.5,
9, 9.5, 10]` ms for J1–J7. A single fractional delay is applied to each
position command at the 5 ms native-drive boundary. The source fit combines
CAN, drive and feedback alignment; it did not separately identify pure command
and observation delay, so the training model does not add a second feedback
delay. The immutable values and source SHA-256 are saved with each run.

Checks completed: a four-environment CPU PhysX smoke covers coherent phase
starts, sent-target slew and a full 1350-step cycle; a one-iteration PPO update
with 128 CPU environments and new-checkpoint playback also passed. Old
checkpoint replay passed through the saved v1 behavior path.
GPU training is not available in this container.

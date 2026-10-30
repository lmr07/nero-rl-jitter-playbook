"""Freeze the latest limit-aware EE checkpoint and identified MIT gains. Offline only."""
import hashlib
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shutil

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT.parent
ASSETS = ROOT / 'assets'
EXPERIMENT = TRAINING / 'logs/nero_ee_tracking_limit_v3'
ACTOR_SOURCE = TRAINING / 'legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py'
FIT_SOURCE = TRAINING / 'identification/gain_delay_fit_with_frequency.json'
URDF_SOURCE = TRAINING / 'resources/robots/a2nero/urdf/nero-fixed.urdf'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def latest_checkpoint(run_name=None):
    runs = []
    candidates = [EXPERIMENT / run_name] if run_name else EXPERIMENT.iterdir()
    for run in candidates:
        if not (run / 'nero_ee_tracking_config.json').is_file():
            continue
        models = sorted((p for p in run.glob('model_*.pt')
                         if re.fullmatch(r'model_\d+\.pt', p.name)),
                        key=lambda p: int(p.stem.split('_')[1]))
        if models:
            runs.append((run, models[-1]))
    if not runs:
        raise FileNotFoundError('No limit-aware EE training checkpoint found')
    return max(runs, key=lambda pair: (pair[0] / 'nero_ee_tracking_config.json').stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=int, help='Freeze a specific iteration from the selected run')
    parser.add_argument('--run', help='Training run directory name')
    args = parser.parse_args()
    run, checkpoint = latest_checkpoint(args.run)
    if args.checkpoint is not None:
        checkpoint = run / f'model_{args.checkpoint}.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    ASSETS.mkdir(exist_ok=True)
    cfg = json.loads((run / 'nero_ee_tracking_config.json').read_text())
    env = cfg['env']
    if env['control']['action_mode'] != 'absolute' or env['control']['pd_backend'] != 'physx_position':
        raise RuntimeError('Unexpected checkpoint control mode')
    if env['goal_ee']['ranges']['pos_p'] != [-np.pi / 5, 2 * np.pi / 5]:
        raise RuntimeError('Unexpected limit-aware EE pitch range')
    margins = np.asarray(env['control'].get('target_position_margin_rad'), dtype=float)
    if margins.shape != (7,) or not np.allclose(margins, [.16]*6+[.30], atol=1e-9):
        raise RuntimeError('Unexpected trained joint target margins')
    reward_cfg = env['rewards']
    if (reward_cfg['limit_prediction_s'] != .15 or reward_cfg['limit_stop_margin_rad'] != .08
        or reward_cfg['limit_stop_speed_rad_s'] != .15):
        raise RuntimeError('Training and hardware predictive limit settings differ')
    if env['env']['num_single_obs'] != 39 or env['env']['frame_stack'] != 16:
        raise RuntimeError('Unexpected stable EE observation shape')
    if not URDF_SOURCE.is_file():
        raise FileNotFoundError(URDF_SOURCE)
    for source, name in ((checkpoint, 'checkpoint.pt'),
                         (run / 'nero_ee_tracking_config.json', 'nero_config.json'),
                         (URDF_SOURCE, 'nero-fixed.urdf'),
                         (FIT_SOURCE, 'identification_fit.json')):
        shutil.copyfile(source, ASSETS / name)
    fit = json.loads((ASSETS / 'identification_fit.json').read_text())
    joints = sorted(fit['joints'], key=lambda row: row['joint'])
    if [row['joint'] for row in joints] != list(range(1, 8)):
        raise RuntimeError('Identification must cover J1..J7')
    alpha = np.array([row['command_only']['train']['coeff'][0] for row in joints])
    beta = np.array([row['command_only']['train']['coeff'][1] for row in joints])
    if not (np.isfinite(alpha).all() and np.isfinite(beta).all()
            and (alpha > 0).all() and (beta > 0).all()):
        raise RuntimeError('Invalid identification multipliers')
    names = [f'nero_joint{i}' for i in range(1, 8)]
    training_kp = np.array([env['control']['stiffness'][name] for name in names])
    training_kd = np.array([env['control']['damping'][name] for name in names])
    if not np.array_equal(training_kp, [40, 40, 20, 20, 20, 20, 20]) or not np.array_equal(training_kd, [1]*7):
        raise RuntimeError('Unexpected saved training PD gains')
    gains = {
        'kp': (training_kp / alpha).tolist(), 'kd': (training_kd / beta).tolist(),
        'alpha': alpha.tolist(), 'beta': beta.tolist(),
        'validation_r2': [row['command_only']['test_r2'] for row in joints],
        'ramp_seconds': 5.0, 'hold_seconds': 5.0,
        'max_tracking_error_rad': 0.10, 'max_speed_rad_s': 0.35,
        'source_sha256': digest(FIT_SOURCE),
    }
    (ASSETS / 'gain_profile.json').write_text(json.dumps(gains, indent=2))
    sim_gate_path = ASSETS / 'sim_gate.json'
    sim_gate = json.loads(sim_gate_path.read_text())
    if (sim_gate['checkpoint_sha256'] != digest(checkpoint) or sim_gate['resets'] != 0
        or sim_gate['predictive_limit_events'] != 0 or sim_gate['target_margin_violations'] != 0):
        raise RuntimeError('The latest checkpoint needs a matching, limit-safe CPU sim gate')
    if sim_gate['mean_error_m'] > .10 or sim_gate['hold_error_m'] > .10:
        raise RuntimeError('Checkpoint did not pass the 10 cm single-rollout tracking gate')
    gaps = np.asarray(sim_gate['sim_max_abs_target_minus_q_rad'], dtype=float)
    if gaps.shape != (7,) or not np.isfinite(gaps).all() or (gaps < 0).any():
        raise RuntimeError('Invalid simulation target-to-feedback gaps')
    floors = np.array([0.20, 0.30, 0.25, 0.36, 0.20, 0.20, 0.20])
    envelope = np.maximum(floors, np.ceil((gaps + 0.05) * 100) / 100)
    if (envelope >= 1.0).any():
        raise RuntimeError('Simulated feedback envelope is too large for this hardware profile')
    guards = {
        'feedback_envelope_rad': envelope.tolist(),
        'source': 'single 20 s noise-free CPU PhysX rollout of the exact checkpoint; no hardware validation',
        'sim_max_abs_target_minus_q_rad': gaps.tolist(),
        'limit_predict_horizon_s': reward_cfg['limit_prediction_s'],
        'limit_predict_margin_rad': reward_cfg['limit_stop_margin_rad'],
        'limit_predict_joints': [1, 2, 3, 4, 5, 6, 7],
        'target_position_margin_rad': margins.tolist(),
    }
    (ASSETS / 'guard_profile.json').write_text(json.dumps(guards, indent=2))
    spec = importlib.util.spec_from_file_location('training_actor', ACTOR_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    e, p = env['env'], cfg['train']['policy']
    ac = module.ActorCritic(e['num_observations'], e['num_privileged_obs'], e['num_pred_obs'],
                            e['num_single_obs'], e['num_actions'], **p)
    loaded = torch.load(ASSETS / 'checkpoint.pt', map_location='cpu', weights_only=False)
    iteration = int(checkpoint.stem.split('_')[1])
    if int(loaded['iter']) != iteration:
        raise RuntimeError('Checkpoint iteration mismatch')
    ac.load_state_dict(loaded['model_state_dict'], strict=True)
    ac.eval()

    class Actor(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            self.encoder = original.adaptation_encoder_module
            self.actor = original.actor_body

        def forward(self, obs):
            return self.actor(torch.cat((obs[:, -39:], self.encoder(obs)), -1))

    torch.manual_seed(iteration)
    sample = torch.randn(64, 624)
    with torch.no_grad():
        expected = ac.act_inference({'obs': sample})
        traced = torch.jit.trace(Actor(ac).eval(), sample)
        error = (traced(sample) - expected).abs().max().item()
        if error > 1.e-6:
            raise RuntimeError('Exported actor mismatch: ' + str(error))
    traced.save(str(ASSETS / 'policy.pt'))
    torch.save({'obs': sample, 'action': expected}, ASSETS / 'export_reference.pt')
    files = ['checkpoint.pt', 'nero_config.json', 'nero-fixed.urdf', 'identification_fit.json',
             'sim_gate.json',
             'gain_profile.json', 'guard_profile.json', 'policy.pt', 'export_reference.pt']
    manifest = {
        'source_run': run.name, 'checkpoint': iteration, 'source_checkpoint': str(checkpoint),
        'training_actor_sha256': digest(ACTOR_SOURCE), 'training_pd_backend': 'physx_position',
        'training_physics_dt_s': env['sim']['dt'], 'actor_shape': [624, 7],
        'frame_shape': [16, 39], 'policy_hz': 50, 'action_mode': 'absolute',
        'target_position_margin_rad': margins.tolist(),
        'control': 'NeroFW.V121 MIT, SocketCAN BCM 20 ms, v_des=0, t_ff=0',
        'startup_gains': {
            'kp': [1.769096842566818, 2.803365826343528, 4.292917438949484,
                   4.048447225123359, 35.77701114306786, 22.96755071762085,
                   29.73665768389295],
            'kd': [0.06199017502206735, 0.1075837623874799, 0.17605636609642372,
                   0.1533826550148886, 1.7768148889357307, 1.3480009723122108,
                   1.517175665895114],
        },
        'export_max_abs_error': error,
        'sha256': {name: digest(ASSETS / name) for name in files},
        'identification_kind': 'SDK current-derived torque multipliers; inverse mapping is experimental',
    }
    (ASSETS / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'run': run.name, 'checkpoint': checkpoint.name, 'hardware_kp': gains['kp'],
                      'hardware_kd': gains['kd'], 'export_error': error}, indent=2))


if __name__ == '__main__':
    main()

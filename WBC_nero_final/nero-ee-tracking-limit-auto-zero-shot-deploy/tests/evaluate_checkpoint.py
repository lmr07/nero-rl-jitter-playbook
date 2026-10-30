import argparse, hashlib, json, sys, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.scripts.play_nero_ee_tracking import restore

parser=argparse.ArgumentParser(add_help=False)
parser.add_argument('--checkpoint', type=int, required=True)
parser.add_argument('--run', default='Oct27_23-25-59_limit_continuation_10000')
parser.add_argument('--output', type=Path)
options, remaining=parser.parse_known_args()
sys.argv=[sys.argv[0]]+remaining
args=get_args(); args.task='nero_ee_tracking_limit'; args.num_envs=1; args.resume=False
run=ROOT/'logs/nero_ee_tracking_limit_v3'/options.run
ckpt=run/f'model_{options.checkpoint}.pt'
saved=json.loads((run/'nero_ee_tracking_config.json').read_text())
cfg,tcfg=task_registry.get_cfgs(args.task)
restore(cfg,saved['env']); restore(tcfg,saved['train'])
cfg.env.num_envs=1;cfg.noise.add_noise=False;cfg.domain_rand.randomize_gripper_mass=False
env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
runner,_=task_registry.make_alg_runner(env,args=args,train_cfg=tcfg,log_root=None)
runner.load(str(ckpt),load_optimizer=False)
actor=runner.alg.actor_critic.eval()
errs=[]; vel=[]; req=[]; q=[]; dq=[]; orn=[]; resets=[]; hold=[]; gaps=[]; sent=[]
with torch.no_grad():
 for tick in range(1000):
  raw=actor.act_inference(env.get_observations())
  _,reward,reset,_=env.step(raw)
  errs.append((env.ee_pos-env.curr_ee_goal_cart_world).norm().item())
  vel.append(env.dof_vel[0,:7].square().mean().sqrt().item())
  req.append(env.pre_slew_request[0].cpu().numpy().copy())
  q.append(env.dof_pos[0,:7].cpu().numpy().copy())
  dq.append(env.dof_vel[0,:7].cpu().numpy().copy())
  sent.append(env.drive_positions[0,:7].cpu().numpy().copy())
  gaps.append((env.drive_positions[0,:7]-env.dof_pos[0,:7]).abs().cpu().numpy().copy())
  orn.append((env.get_ee_orientation_rpy()[0]-env.curr_ee_goal_orn_rpy[0]).norm().item())
  resets.append(bool(reset[0]))
  hold.append(bool(env.goal_timer[0]>=env.traj_timesteps[0]))
for label,sl in [('first_3s',slice(0,150)),('all_20s',slice(0,1000))]:
 a=np.asarray(errs)[sl]; v=np.asarray(vel)[sl]; o=np.asarray(orn)[sl]; h=np.asarray(hold)[sl]
 print(label,{'mean_error_m':float(a.mean()),'max_error_m':float(a.max()),'orn_mean_rad':float(o.mean()),'joint_speed_rms_rad_s':float(np.sqrt(np.mean(v*v))),'hold_error_m':float(a[h].mean()) if h.any() else None})
print('resets',sum(resets),'q_ranges',np.ptp(q,axis=0).tolist(),'raw_step_p95_rad',np.percentile(np.abs(np.diff(req,axis=0)),95,axis=0).tolist())
lower=env.lower[:7].cpu().numpy(); upper=env.upper[:7].cpu().numpy()
margin=np.asarray(cfg.control.target_position_margin_rad)
predicted=np.asarray(q)+np.asarray(dq)*cfg.rewards.limit_prediction_s
approaching=((np.asarray(dq)<-cfg.rewards.limit_stop_speed_rad_s)&
             (predicted<lower+cfg.rewards.limit_stop_margin_rad))|((np.asarray(dq)>cfg.rewards.limit_stop_speed_rad_s)&
             (predicted>upper-cfg.rewards.limit_stop_margin_rad))
target_margin_violations=int(np.count_nonzero((np.asarray(sent)<lower+margin-1e-5)|
                                             (np.asarray(sent)>upper-margin+1e-5)))
summary={'checkpoint':str(ckpt),'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),
         'sim_max_abs_target_minus_q_rad':np.max(gaps,axis=0).tolist(),
         'mean_error_m':float(np.mean(errs)), 'hold_error_m':float(np.mean(np.asarray(errs)[np.asarray(hold)])),
         'resets':sum(resets), 'predictive_limit_events':int(np.count_nonzero(approaching)),
         'target_margin_violations':target_margin_violations,
         'max_abs_joint_velocity_rad_s':np.max(np.abs(dq),axis=0).tolist()}
print('SUMMARY_JSON',json.dumps(summary))
if options.output is not None:
 options.output.parent.mkdir(parents=True,exist_ok=True)
 options.output.write_text(json.dumps(summary,indent=2))
env.gym.destroy_sim(env.sim)

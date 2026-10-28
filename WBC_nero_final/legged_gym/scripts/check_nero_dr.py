"""GPU regression for fixed-target stability, real limits, sensor isolation and resets."""
import os,sys,json,time
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry

args=get_args();args.task='nero_fixed';args.headless=True;args.num_envs=64
cfg,_=task_registry.get_cfgs(args.task);cfg.noise.add_noise=False
# Same near-straight pose used for the real fixed-target diagnostic.
pose=[.006,.022,.078,.033,-.16,-.046,-.277]
cfg.init_state.default_joint_angles.update({f'nero_joint{i+1}':q for i,q in enumerate(pose)})
env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
try:
    a=env.actuator
    assert a is not None and not getattr(cfg.domain_rand,'curriculum',False)
    assert torch.allclose(env.torque_limits[:7],env.tensor([24,24,16,16,8,8,8]))
    assert list(env.dof_props['velocity'][:7])==[3,3,3,4,4,4,4]
    max_substep_speed=torch.zeros((),device=env.device)
    original_feedback=a.feedback
    def capture_feedback(q,dq):
        if tick>=200:
            max_substep_speed.copy_(torch.maximum(max_substep_speed,dq.abs().max()))
        original_feedback(q,dq)
    a.feedback=capture_feedback
    p0=env.gym.get_actor_rigid_body_properties(env.envs[0],env.actors[0])
    p1=env.gym.get_actor_rigid_body_properties(env.envs[1],env.actors[1])
    for x,y in zip(p0,p1):
        trace_x=x.inertia.x.x+x.inertia.y.y+x.inertia.z.z
        trace_y=y.inertia.x.x+y.inertia.y.y+y.inertia.z.z
        if trace_x>0 and trace_y>0:
            assert abs(trace_x/trace_y-x.mass/y.mass)<1.e-4,'Mass/inertia scaling mismatch'
    fixed_gain=a.kp_scale.clone();tail_speed=0.;tail_positions=[];begun=time.monotonic()
    for tick in range(300):
        with torch.no_grad():env.step(torch.zeros(64,7,device=env.device))
        assert torch.isfinite(env.obs_buf).all()
        assert (env.torques[:,:7].abs()<=a.torque_cap+1.e-6).all()
        assert torch.equal(a.kp_scale,fixed_gain)
        torch.testing.assert_close(env.obs_history[:,-1,:7],a.measured_q-env.default_dof_pos[:7])
        torch.testing.assert_close(env.obs_history[:,-1,14:21],(a.host_target-env.default_dof_pos[:7])/.25)
        if tick>=200:
            tail_speed=max(tail_speed,env.dof_vel[:,:7].abs().max().item());tail_positions.append(env.dof_pos[:,:7].clone())
    # Changing only the unobserved true state must not change actor sensor channels.
    old=env.dof_pos.clone();actor_before=env.obs_history[:,-1].clone();env.dof_pos[:,0]+=.1
    env.compute_observations()
    torch.testing.assert_close(env.obs_history[:,-1,:14],actor_before[:,:14])
    torch.testing.assert_close(env.obs_history[:,-1,21:24],actor_before[:,21:24])
    assert (env.critic_history[:,-1,0]-actor_before[:,0]).abs().min()>.09
    env.dof_pos.copy_(old)
    ids=torch.tensor([0,3,7],device=env.device);other=a.host_target[1].clone()
    env.reset_idx(ids);env.compute_observations()
    assert (env.obs_history[ids,:-1]==0).all()
    assert (env.last_candidate_delta[ids]==0).all()
    assert (env.hold_samples[ids]==0).all()
    torch.testing.assert_close(a.host_target[1],other)
    torch.testing.assert_close(a.command_history[:,ids],env.default_dof_pos[:7].expand(a.ring_size,len(ids),7))
    report={'seconds':time.monotonic()-begun,'max_tail_joint_speed_rad_s':tail_speed,
            'max_tail_substep_speed_rad_s':max_substep_speed.item(),
            'max_tail_joint_ptp_rad':(torch.stack(tail_positions).amax(0)-torch.stack(tail_positions).amin(0)).max().item(),
            'num_envs':64,'policy_ticks':300,'reference_torque_limits':cfg.actuator.torque_limits}
    print(json.dumps(report,indent=2),flush=True)
    assert max_substep_speed.item()<.05,'Unexpected substep velocity during fixed-target hold'
    print('PASS: fixed-target randomized dynamics, caps, sensor isolation, partial reset')
finally:env.gym.destroy_sim(env.sim)

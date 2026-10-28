"""CPU tests do not import IsaacGym or initialize hardware."""
import ast
import importlib.util
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('nero_actuator', ROOT/'legged_gym/envs/nero/nero_actuator.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

def config():
    tree = ast.parse((ROOT/'legged_gym/envs/nero/nero_fixed_config.py').read_text())
    outer = next(c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == 'NeroFixedCfg')
    cls = next(c for c in outer.body if isinstance(c, ast.ClassDef) and c.name == 'actuator')
    namespace = {}; exec(compile(ast.Module(body=[cls], type_ignores=[]), '<actuator config>', 'exec'), namespace)
    return namespace['actuator']()

def model(cfg=None, randomize=False):
    return module.NeroActuator(2, .000125, .02, cfg or config(), randomize, torch.zeros(7), -torch.ones(7)*3, torch.ones(7)*3)

class ActuatorTests(unittest.TestCase):
    def test_incremental_takeover_and_no_reset_windup(self):
        a = model()
        a.host_target.fill_(.1)
        a.host_elapsed.fill_(1.5)  # half takeover
        sent = a.submit(a.host_target + .012, incremental=True).clone()
        torch.testing.assert_close(sent, torch.full((2,7), .106))
        # Zero increment holds current command; does not pull back to initial_target.
        held = a.submit(a.host_target.clone(), incremental=True).clone()
        torch.testing.assert_close(held, sent)
        a.reset(torch.tensor([0]), torch.zeros(1,7), torch.zeros(1,7))
        torch.testing.assert_close(a.host_target[0], torch.zeros(7))
        torch.testing.assert_close(a.host_target[1], sent[1])
        # Feedback envelope overrides incremental accumulation and becomes next origin.
        a.host_elapsed.fill_(3.)
        a.host_target.fill_(.19)
        a.submit(a.host_target + .012, incremental=True)
        torch.testing.assert_close(a.host_target, torch.full((2,7), .2))
        a.submit(a.host_target + .012, incremental=True)
        torch.testing.assert_close(a.host_target, torch.full((2,7), .2))

    def test_limits_and_constant_distributions(self):
        a = model(randomize=True)
        for _ in range(8):
            a.reset(torch.arange(2), torch.zeros(2,7), torch.zeros(2,7))
            self.assertTrue((a.torque_cap <= torch.tensor([24,24,16,16,8,8,8])).all())
            self.assertTrue((a.torque_cap >= .8*torch.tensor([24,24,16,16,8,8,8])).all())
            self.assertTrue((a.speed_cap <= torch.tensor([3,3,3,4,4,4,4])).all())
            before = a.kp_scale.clone()
            a.submit(torch.ones(2,7)); a.feedback(torch.zeros(2,7),torch.zeros(2,7))
            torch.testing.assert_close(a.kp_scale, before)

    def test_host_target_and_can_wait(self):
        a = model(); a.host_elapsed[:] = 3
        a.next_tx[:] = 4
        sent = a.submit(torch.ones(2,7)).clone()
        torch.testing.assert_close(sent, torch.full((2,7),.02))
        q=torch.zeros(2,7);kp=torch.ones(7)*48;kd=torch.ones(7)*1.5
        for _ in range(4):
            torch.testing.assert_close(a.torque(q,q,kp,kd),q)
            a.feedback(q,q)
        self.assertTrue((a.torque(q,q,kp,kd)>0).all())
        torch.testing.assert_close(a.drive_target,sent)
        torch.testing.assert_close(a.host_target,sent)

    def test_feedback_delay_quantization_and_no_state_leak(self):
        a=model();a.feedback_delay[:]=2;a.next_rx[:]=1;a.rx_period[:]=1
        zero=torch.zeros(2,7);q=torch.full((2,7),.1234)
        a.feedback(q,zero);torch.testing.assert_close(a.measured_q,zero)
        a.feedback(q,zero);torch.testing.assert_close(a.measured_q,zero)
        a.feedback(q,zero);torch.testing.assert_close(a.measured_q,torch.full((2,7),.123))

    def test_cap_and_braking_not_disabled_at_speed_limit(self):
        a=model();q=torch.zeros(2,7);dq=torch.full((2,7),4.)
        a.host_target[:]=3;a.next_tx[:]=0
        torque=a.torque(q,dq,torch.ones(7)*48,torch.ones(7)*1.5)
        self.assertTrue((torque==0).all())
        a.host_target[:]=-3;a.next_tx[:]=0
        torque=a.torque(q,dq,torch.ones(7)*48,torch.ones(7)*1.5)
        self.assertTrue((torque<0).all());self.assertTrue((torque.abs()<=a.torque_cap).all())

    def test_partial_reset_has_no_old_episode_samples(self):
        a=model();q=torch.ones(2,7);z=torch.zeros(2,7)
        for _ in range(20):a.feedback(q,z)
        old_clock=a.clock[1].clone();old_history=a.state_history[:,1].clone()
        a.reset(torch.tensor([0]),torch.full((1,7),.5),torch.zeros(1,7))
        self.assertTrue((a.state_history[:,0,:,0]==.5).all())
        self.assertTrue((a.command_history[:,0]==.5).all())
        torch.testing.assert_close(a.state_history[:,1],old_history)
        torch.testing.assert_close(a.clock[1],old_clock)

    def test_takeover_starts_at_reset_anchor(self):
        a=model();q=torch.full((2,7),.4)
        a.reset(torch.arange(2),q,torch.zeros(2,7))
        torch.testing.assert_close(a.submit(torch.full((2,7),2.)),q)

    def test_invalid_timestep_and_reference_limits_rejected(self):
        c=config();c.torque_limits=[100]*6
        with self.assertRaises(ValueError):model(c)
        with self.assertRaises(ValueError):module.NeroActuator(2,.005,.02,config(),False,torch.zeros(7),-torch.ones(7),torch.ones(7))
        with self.assertRaises(ValueError):module.NeroActuator(2,.0005,.02,config(),False,torch.zeros(7),-torch.ones(7),torch.ones(7))

    def test_validated_larger_timestep_keeps_bus_timing(self):
        a=module.NeroActuator(2,.0002,.02,config(),False,torch.zeros(7),-torch.ones(7),torch.ones(7))
        self.assertTrue((a.tx_period==100).all())
        self.assertTrue((a.rx_period==25).all())
        intended=torch.arange(7)*.00025
        self.assertTrue(((a.next_tx[0]*a.dt-intended).abs()<=.0001001).all())

if __name__=='__main__':unittest.main()

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nero_only.core import FixedPolicy, ROOT
from nero_only.gain_trial import profile, equilibrium_target
from nero_only.hardware import NeroHardware, predictive_limit_risk
from nero_only.run import check_post_gain_home


class OfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.policy = FixedPolicy()

    def setUp(self):
        self.policy.reset()

    def test_export_and_identified_gains(self):
        p = self.policy
        reference = torch.load(ROOT/'assets/export_reference.pt', map_location='cpu', weights_only=True)
        with torch.no_grad():
            self.assertLess((p.actor(reference['obs'])-reference['action']).abs().max().item(), 1.e-6)
        gains = profile()
        np.testing.assert_allclose(p.training_kp/np.asarray(gains['alpha']), gains['kp'], rtol=1.e-12)
        np.testing.assert_allclose(p.training_kd/np.asarray(gains['beta']), gains['kd'], rtol=1.e-12)

    def test_initial_orientation_and_goal_bounds(self):
        p = self.policy
        np.testing.assert_allclose(p.target_cart(), p.fk(p.default)[0], atol=1.e-3)
        self.assertAlmostEqual(p.orn[1], -.4707963, places=4)
        for _ in range(100):
            p.advance()
        self.assertAlmostEqual(p.orn[1], 0., places=6)
        self.assertTrue(p.move('i'))
        self.assertTrue(np.all(p.endpoint >= p.bounds[:, 0]))
        self.assertTrue(np.all(p.endpoint <= p.bounds[:, 1]))

    def test_automatic_goals_follow_training_ranges_and_timing(self):
        p = self.policy
        p.enable_auto_goals()
        self.assertGreaterEqual(p.travel_ticks*p.dt, 1.)
        self.assertLessEqual(p.travel_ticks*p.dt, 3.)
        self.assertGreaterEqual((p.total_ticks-p.travel_ticks)*p.dt, 2.)
        self.assertLessEqual((p.total_ticks-p.travel_ticks)*p.dt, 5.)
        changes = 0
        for _ in range(round(60/p.dt)):
            previous = p.segment
            p.advance()
            if p.segment != previous:
                changes += 1
                self.assertTrue(np.all(p.endpoint >= p.bounds[:, 0]))
                self.assertTrue(np.all(p.endpoint <= p.bounds[:, 1]))
                self.assertFalse(p.path_collides(p.start, p.endpoint))
                self.assertGreaterEqual(p.travel_ticks*p.dt, 1.)
                self.assertLessEqual(p.travel_ticks*p.dt, 3.)
                self.assertGreaterEqual((p.total_ticks-p.travel_ticks)*p.dt, 2.)
                self.assertLessEqual((p.total_ticks-p.travel_ticks)*p.dt, 5.)
                np.testing.assert_allclose(p.orn_end, np.zeros(3))
        self.assertGreaterEqual(changes, 4)

    def test_rate_limit_and_sent_action_feedback(self):
        p = self.policy
        for n in range(50):
            previous = p.last_sent.copy()
            target = p.candidate_from_raw(np.full(7, 100 if n%2 else -100))
            self.assertTrue(np.all(np.abs(target-previous) <= p.max_target_step+1.e-12))
            self.assertTrue(np.all(target >= p.target_lower-1.e-12))
            self.assertTrue(np.all(target <= p.target_upper+1.e-12))
            p.record_sent(target)
            np.testing.assert_allclose(p.previous_action, (target-p.default)/p.scale)

    def test_gain_ramp_conserves_static_p_effort(self):
        p = self.policy
        gains = profile()
        anchor = p.default + np.array([0., -.02, 0., .03, 0., 0., 0.])
        old_target = p.default.copy()
        old_kp = p.kp.copy()
        new_kp = np.asarray(gains['kp'])
        target = equilibrium_target(anchor, old_target, old_kp, new_kp)
        np.testing.assert_allclose(new_kp*(target-anchor), old_kp*(old_target-anchor), atol=1.e-12)

    def test_predictive_limit_guard(self):
        p = self.policy
        q, dq = p.default.copy(), np.zeros(7)
        q[1], dq[1] = p.lower[1]+.12, -.9
        risk = predictive_limit_risk(q, dq, p.lower, p.upper, .15, .08)
        self.assertIn(1, risk)

    def test_feedback_envelope_and_slew_intersection(self):
        p = self.policy
        hw = NeroHardware('unused', p)
        hw.last_sent = p.default.copy()
        q = p.default.copy()
        q[1] += .28
        target = hw.guarded_target(p.default + .5, q, max_step=.03, envelope=.30)
        self.assertTrue(np.all(np.abs(target-p.default) <= .03+1.e-12))
        self.assertTrue(np.all(np.abs(target-q) <= .30+1.e-12))
        q[1] += .10
        with self.assertRaisesRegex(RuntimeError, 'No safe target'):
            hw.guarded_target(p.default + .5, q, max_step=.03, envelope=.30)

    def test_hardware_target_preserves_j7_training_margin(self):
        p = self.policy
        hw = NeroHardware('unused', p)
        q = p.default.copy()
        q[6] = p.target_upper[6] - .01
        hw.last_sent = q.copy()
        candidate = q.copy()
        candidate[6] = p.upper[6]
        target = hw.guarded_target(candidate, q, max_step=p.max_target_step,
                                   envelope=p.feedback_envelope)
        self.assertLessEqual(target[6], p.target_upper[6] + 1.e-12)
        self.assertAlmostEqual(p.target_upper[6], 1.2707963, places=6)

    def test_post_gain_home_check_uses_measured_state(self):
        p = self.policy
        q = p.default + np.array([.008, -.055, -.009, .047, .001, 0., .006])
        target = p.initial_target.copy()
        target[3] = 1.78073814
        self.assertGreater(abs(target[3] - p.initial_target[3]), .15)
        check_post_gain_home(p, q, np.zeros(7), target)
        bad_q = q.copy()
        bad_q[6] = p.default[6] + .11
        with self.assertRaisesRegex(RuntimeError, 'home error'):
            check_post_gain_home(p, bad_q, np.zeros(7), target)
        bad_dq = np.zeros(7)
        bad_dq[6] = .09
        with self.assertRaisesRegex(RuntimeError, 'speed'):
            check_post_gain_home(p, q, bad_dq, target)
        bad_target = target.copy()
        bad_target[6] = p.target_upper[6] + .01
        with self.assertRaisesRegex(RuntimeError, 'target margin violations'):
            check_post_gain_home(p, q, np.zeros(7), bad_target)


if __name__ == '__main__':
    unittest.main()

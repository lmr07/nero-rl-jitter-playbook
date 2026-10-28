import importlib.util
import json
from pathlib import Path
import unittest

import torch

from legged_gym.b2_gym_learn.ppo_cse_pf.actor_critic import ActorCritic
from legged_gym.b2_gym_learn.ppo_cse_pf.ppo import PPO


WBC_ROOT = Path(__file__).resolve().parents[2]
ACTUATOR_PATH = (
    WBC_ROOT / "legged_gym/envs/a2/a2nero_stage3_actuator.py"
)
SENSOR_PATH = WBC_ROOT / "legged_gym/envs/a2/a2nero_stage3_sensors.py"
PROFILE_PATH = (
    WBC_ROOT / "training_profiles/model39000_woimu_training_identification.json"
)


def _load_actuator_module():
    spec = importlib.util.spec_from_file_location(
        "model39000_identified_actuator", ACTUATOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sensor_module():
    spec = importlib.util.spec_from_file_location(
        "model39000_identified_sensors", SENSOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Model39000IdentifiedActuatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_actuator_module()
        cls.sensor_module = _load_sensor_module()

    def _make(self, *, delay=0, tau=0.0):
        return self.module.A2NeroStage3Actuator(
            1,
            torch.zeros(19),
            torch.full((19,), -10.0),
            torch.full((19,), 10.0),
            identified_enabled=True,
            action_scale=0.25,
            a2_delay_options=[0],
            a2_delay_probabilities=[1.0],
            nero_delay_options=[0],
            nero_delay_probabilities=[1.0],
            joint_delay_ranges_ticks=[[delay, delay] for _ in range(19)],
            a2_tau_range_s=(tau, tau),
            nero_tau_range_s=(tau, tau),
            policy_dt=0.02,
        )

    def test_delay_changes_plant_not_sent_history(self):
        actuator = self._make(delay=2)
        zero = torch.zeros(1, 19)
        actuator.reset(zero, a2_delay_ticks=0, nero_delay_ticks=0)
        plant = []
        for _ in range(3):
            tick = actuator.advance(torch.ones(1, 19), zero, False)
            plant.append(float(tick.plant_target[0, 0]))
        self.assertEqual(plant, [0.0, 0.0, 0.25])
        torch.testing.assert_close(tick.previous_action, torch.ones(1, 19))

    def test_time_constant_is_plant_only(self):
        actuator = self._make(tau=0.02)
        zero = torch.zeros(1, 19)
        actuator.reset(zero, a2_delay_ticks=0, nero_delay_ticks=0)
        tick = actuator.advance(torch.ones(1, 19), zero, False)
        expected = 0.25 * (1.0 - torch.exp(torch.tensor(-1.0)))
        torch.testing.assert_close(
            tick.plant_target, torch.full_like(tick.plant_target, expected)
        )
        torch.testing.assert_close(tick.previous_action, torch.ones(1, 19))

    def test_profile_preserves_model39000_abi(self):
        profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(profile["source"]["iteration"], 38000)
        self.assertEqual(profile["abi"]["num_single_obs"], 85)
        self.assertEqual(profile["abi"]["frame_stack"], 16)
        self.assertEqual(profile["abi"]["num_actions"], 19)
        self.assertEqual(profile["abi"]["end_effector_imu_channels"], 0)
        self.assertFalse(profile["stage1"]["ppo"]["load_optimizer_state"])
        self.assertFalse(
            profile["stage1"]["ppo"]["nominal_evaluation_enabled"]
        )
        self.assertLessEqual(
            profile["stage1"]["ppo"]["actor_learning_rate_cap"], 2.0e-7
        )
        self.assertTrue(profile["stage1"]["ppo"]["freeze_action_std"])
        self.assertEqual(
            profile["stage1"]["ppo"]["actor_warmup_fraction"], 0.2
        )
        self.assertEqual(
            profile["stage1"]["ppo"]["adaptation_warmup_fraction"], 0.4
        )
        self.assertGreater(
            profile["stage1"]["ppo"]["source_action_anchor_coef"], 0.0
        )
        self.assertEqual(
            profile["stage1"]["curriculum"]["reward_ramp_end_fraction"],
            0.2,
        )
        self.assertGreater(
            profile["stage1"]["curriculum"]["randomization_start_fraction"],
            0.1,
        )
        self.assertEqual(
            profile["stage2"]["actuator"]["delay_ms"]["nero_outer"],
            [60.0, 100.0],
        )

    def test_unobserved_flex_state_is_finite_and_not_an_observation(self):
        profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        model = self.sensor_module.A2NeroUnobservedWristFlexModel(
            2,
            "cpu",
            0.02,
            profile["stage2"]["unobserved_wrist_flex"],
        )
        for tick in range(200):
            excitation = torch.zeros(2, 3)
            if tick == 0:
                excitation[:, 2] = 10.0
            output = model.advance(excitation)
            self.assertEqual(tuple(output.shape), (2, 3))
            self.assertTrue(torch.all(torch.isfinite(output)))
        # The class intentionally exposes no method that appends channels to
        # the 85-D student observation.
        self.assertFalse(hasattr(model, "observation"))


class ContinuationOptimizationSafetyTest(unittest.TestCase):
    def _make_algorithm(self):
        actor_critic = ActorCritic(
            num_obs=8,
            num_privileged_obs=6,
            num_obs_pred=9,
            num_single_obs=4,
            num_actions=3,
            actor_hidden_dims=[8],
            critic_hidden_dims=[8],
            adaptation_num_single_obs=4,
            latent_dim=2,
        )
        return PPO(
            actor_critic,
            actor_warmup_iterations=2,
            freeze_action_std=True,
            adaptation_optimizer_modules_only=True,
            source_action_anchor_coef=1.0,
            adaptation_action_anchor_coef=1.0,
        )

    def test_source_policy_snapshot_is_immutable(self):
        algorithm = self._make_algorithm()
        observations = torch.randn(5, 8)
        algorithm.capture_source_policy()
        source_before = algorithm._source_action_mean(observations)
        with torch.no_grad():
            algorithm.actor_critic.actor_body[-1].bias.add_(1.0)
        source_after = algorithm._source_action_mean(observations)
        current = algorithm.actor_critic.act_student(observations)
        torch.testing.assert_close(source_before, source_after)
        self.assertFalse(torch.allclose(current, source_after))

    def test_adaptation_optimizer_cannot_update_actor_or_critic(self):
        algorithm = self._make_algorithm()
        optimized_ids = {
            id(parameter)
            for group in algorithm.adaptation_module_optimizer.param_groups
            for parameter in group["params"]
        }
        expected_ids = {
            id(parameter)
            for module in (
                algorithm.actor_critic.adaptation_encoder_module,
                algorithm.actor_critic.adaptation_decoder_module,
            )
            for parameter in module.parameters()
        }
        self.assertEqual(optimized_ids, expected_ids)
        self.assertTrue(all(
            id(parameter) not in optimized_ids
            for parameter in algorithm.actor_critic.actor_body.parameters()
        ))
        self.assertTrue(all(
            id(parameter) not in optimized_ids
            for parameter in algorithm.actor_critic.critic_body.parameters()
        ))

    def test_actor_gradients_are_suppressed_during_warmup(self):
        algorithm = self._make_algorithm()
        algorithm.set_continuation_iteration(0)
        observations = torch.randn(5, 8)
        critic_observations = torch.randn(5, 6)
        algorithm.actor_critic.update_distribution(observations)
        loss = (
            algorithm.actor_critic.action_mean.sum()
            + algorithm.actor_critic.evaluate(
                critic_observations, observations=observations
            ).sum()
        )
        loss.backward()
        algorithm._freeze_warmup_actor_gradients()
        self.assertTrue(all(
            parameter.grad is None
            for parameter in algorithm.actor_critic.actor_body.parameters()
        ))
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in algorithm.actor_critic.critic_body.parameters()
        ))


if __name__ == "__main__":
    unittest.main()

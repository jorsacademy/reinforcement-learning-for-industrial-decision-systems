import importlib.util
import unittest

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class MaintenancePOMDPCoreTests(unittest.TestCase):
    def test_bayes_update_normalizes_and_alarm_increases_severe_belief(self):
        from industrial_rl.pomdp_maintenance import (
            ALARM,
            OPERATE,
            MaintenancePOMDPEnv,
        )

        env = MaintenancePOMDPEnv()
        prior = np.asarray([0.70, 0.25, 0.05, 0.0])
        predicted = env.predict_belief(prior, OPERATE)
        posterior = env.bayes_update(predicted, ALARM)

        self.assertAlmostEqual(float(posterior.sum()), 1.0)
        self.assertTrue(np.all(posterior >= 0.0))
        self.assertGreater(
            posterior[2] + posterior[3],
            predicted[2] + predicted[3],
        )

    def test_replacement_prediction_returns_asset_near_healthy(self):
        from industrial_rl.pomdp_maintenance import (
            REPLACE,
            MaintenancePOMDPEnv,
        )

        env = MaintenancePOMDPEnv()
        belief = np.asarray([0.05, 0.15, 0.50, 0.30])
        predicted = env.predict_belief(belief, REPLACE)

        self.assertGreater(predicted[0], 0.99)
        self.assertLess(predicted[1], 0.01)
        self.assertAlmostEqual(float(predicted.sum()), 1.0)

    def test_belief_grid_is_valid_simplex(self):
        from industrial_rl.pomdp_maintenance import belief_grid

        grid = belief_grid(resolution=4)
        self.assertEqual(grid.shape[1], 4)
        np.testing.assert_allclose(grid.sum(axis=1), 1.0)
        self.assertTrue(np.all(grid >= 0.0))
        self.assertEqual(len(grid), 35)

    def test_discretized_belief_dp_returns_valid_actions(self):
        from industrial_rl.pomdp_maintenance import (
            MaintenancePOMDPConfig,
            discretized_belief_dp,
        )

        config = MaintenancePOMDPConfig(horizon=4)
        grid, values, policy = discretized_belief_dp(
            config,
            resolution=4,
        )

        self.assertEqual(values.shape, (5, len(grid)))
        self.assertEqual(policy.shape, (4, len(grid)))
        self.assertTrue(np.all((policy >= 0) & (policy < 3)))
        self.assertTrue(np.isfinite(values).all())

    def test_policy_state_does_not_expose_hidden_health(self):
        from industrial_rl.pomdp_maintenance import MaintenancePOMDPEnv

        env = MaintenancePOMDPEnv()
        state = env.reset(seed=4)
        self.assertEqual(state.shape, (5,))
        self.assertAlmostEqual(float(state[1:].sum()), 1.0)


@unittest.skipUnless(
    TORCH_AVAILABLE,
    "PyTorch optional dependency is not installed",
)
class MaintenancePOMDPDQNTests(unittest.TestCase):
    def test_small_belief_dqn_is_executable(self):
        from industrial_rl.pomdp_maintenance import (
            BeliefDQNAgent,
            BeliefDQNConfig,
            MaintenancePOMDPConfig,
            MaintenancePOMDPEnv,
        )

        config = MaintenancePOMDPConfig(horizon=5)
        env = MaintenancePOMDPEnv(config)
        agent = BeliefDQNAgent(
            env,
            BeliefDQNConfig(
                episodes=120,
                warmup=32,
                batch_size=32,
                replay_size=1500,
                target_update=50,
                seed=31,
            ),
        ).fit()

        env.reset(seed=100)
        action = agent.policy()(
            env.t,
            env.belief.copy(),
            env.observation,
        )
        self.assertTrue(0 <= action < env.n_actions)


if __name__ == "__main__":
    unittest.main()

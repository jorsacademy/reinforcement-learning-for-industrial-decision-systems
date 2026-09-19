import importlib.util
import unittest

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch optional dependency is not installed")
class DQNInventoryTests(unittest.TestCase):
    def test_exact_dp_and_dqn_actions_are_feasible(self):
        from industrial_rl.dqn_inventory import (
            DQNConfig,
            DQNInventoryAgent,
            RegimeInventoryConfig,
            RegimeInventoryEnv,
            exact_regime_inventory_dp,
        )

        config = RegimeInventoryConfig(horizon=5, max_inventory=20, max_order=6)
        values, table = exact_regime_inventory_dp(config)
        self.assertEqual(values.shape, (6, 21, 2))
        self.assertEqual(table.shape, (5, 21, 2))

        env = RegimeInventoryEnv(config)
        agent = DQNInventoryAgent(
            env,
            DQNConfig(episodes=120, warmup=40, batch_size=32, replay_size=2000, seed=9),
        ).fit()
        policy = agent.policy()
        for t in range(config.horizon):
            for inv in (0, 5, 10, 20):
                for regime in (0, 1):
                    self.assertIn(policy(t, inv, regime), env.feasible_actions(inv))

    def test_observation_is_finite(self):
        from industrial_rl.dqn_inventory import RegimeInventoryEnv

        env = RegimeInventoryEnv()
        obs = env.reset(seed=1)
        self.assertEqual(obs.shape, (4,))
        self.assertTrue(np.isfinite(obs).all())


if __name__ == "__main__":
    unittest.main()

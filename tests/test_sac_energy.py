import importlib.util
import unittest

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch optional dependency is not installed")
class SACEnergyTests(unittest.TestCase):
    def test_environment_action_mapping_and_transition(self):
        from industrial_rl.sac_energy import EnergyProductionConfig, EnergyProductionEnv

        config = EnergyProductionConfig(horizon=4, max_rate=8.0)
        env = EnergyProductionEnv(config)
        env.reset(seed=1)

        self.assertAlmostEqual(env._map_action(-1.0), 0.0)
        self.assertAlmostEqual(env._map_action(1.0), config.max_rate)

        next_inv, reward, info = env.transition(
            inventory=2.0,
            previous_rate=3.0,
            production_rate=4.0,
            demand=5.0,
            price=1.2,
        )
        self.assertAlmostEqual(next_inv, 1.0)
        self.assertLess(reward, 0.0)
        self.assertGreater(info["energy_use"], 0.0)

    def test_mpc_policy_returns_bounded_action(self):
        from industrial_rl.sac_energy import (
            EnergyProductionConfig,
            EnergyProductionEnv,
            mpc_grid_policy,
        )

        config = EnergyProductionConfig(horizon=5)
        env = EnergyProductionEnv(config)
        obs = env.reset(seed=2)
        action = mpc_grid_policy(config, lookahead=2, grid_points=5)(obs)
        self.assertTrue(-1.0 <= action <= 1.0)

    def test_small_sac_is_executable_and_bounded(self):
        from industrial_rl.sac_energy import (
            EnergyProductionConfig,
            EnergyProductionEnv,
            SACConfig,
            SACProductionAgent,
        )

        config = EnergyProductionConfig(horizon=5)
        env = EnergyProductionEnv(config)
        agent = SACProductionAgent(
            env,
            SACConfig(
                total_steps=260,
                warmup_steps=64,
                batch_size=32,
                replay_size=1000,
                seed=11,
            ),
        ).fit()

        obs = env.reset(seed=10)
        action = agent.policy()(obs)
        self.assertTrue(np.isfinite(action))
        self.assertTrue(-1.0 <= action <= 1.0)


if __name__ == "__main__":
    unittest.main()

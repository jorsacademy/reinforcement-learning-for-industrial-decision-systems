import importlib.util
import unittest

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch optional dependency is not installed")
class SACEnergyTests(unittest.TestCase):
    def test_energy_and_transition_accounting(self):
        from industrial_rl.sac_energy import (
            EnergyProductionConfig,
            EnergyProductionEnv,
        )

        config = EnergyProductionConfig(horizon=4)
        env = EnergyProductionEnv(config)

        low_energy = env.energy_consumption(2.0)
        high_energy = env.energy_consumption(8.0)
        self.assertGreater(high_energy, low_energy)

        next_inventory, reward, info = env.transition(
            net_inventory=2.0,
            energy_price=1.2,
            production=5.0,
            demand=6.0,
        )
        self.assertAlmostEqual(next_inventory, 1.0)
        self.assertGreater(-reward, 0.0)
        self.assertGreater(info["energy_cost"], 0.0)

    def test_mpc_action_is_bounded(self):
        from industrial_rl.sac_energy import (
            EnergyProductionConfig,
            rolling_horizon_mpc_policy,
        )

        config = EnergyProductionConfig(horizon=5, max_production=8.0)
        policy = rolling_horizon_mpc_policy(config, lookahead=2, grid_points=5)
        action = policy(0, 0.0, 1.0)
        self.assertGreaterEqual(action, 0.0)
        self.assertLessEqual(action, config.max_production)

    def test_small_sac_fit_returns_finite_continuous_action(self):
        from industrial_rl.sac_energy import (
            EnergyProductionConfig,
            EnergyProductionEnv,
            SACConfig,
            SACEnergyAgent,
        )

        config = EnergyProductionConfig(horizon=5, max_production=8.0)
        env = EnergyProductionEnv(config)
        agent = SACEnergyAgent(
            env,
            SACConfig(
                episodes=35,
                replay_size=1500,
                batch_size=32,
                warmup_steps=40,
                updates_per_step=1,
                seed=13,
            ),
        ).fit()

        action = agent.policy()(0, config.initial_inventory, config.initial_price)
        self.assertTrue(np.isfinite(action))
        self.assertGreaterEqual(action, 0.0)
        self.assertLessEqual(action, config.max_production)


if __name__ == "__main__":
    unittest.main()

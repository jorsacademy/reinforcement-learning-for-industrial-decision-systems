import importlib.util
import unittest

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class RiskInventoryCoreTests(unittest.TestCase):
    def test_empirical_cvar_hand_check(self):
        from industrial_rl.risk_inventory import empirical_cvar

        costs = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(
            empirical_cvar(costs, alpha=0.60),
            4.5,
        )

    def test_demand_pmf_is_valid(self):
        from industrial_rl.risk_inventory import RiskInventoryEnv

        env = RiskInventoryEnv()
        values, probs = env.demand_pmf()
        self.assertEqual(len(values), len(probs))
        self.assertAlmostEqual(float(probs.sum()), 1.0)
        self.assertTrue(np.all(probs >= 0.0))

    def test_exact_dp_actions_are_feasible(self):
        from industrial_rl.risk_inventory import (
            RiskInventoryConfig,
            RiskInventoryEnv,
            exact_risk_neutral_dp,
        )

        config = RiskInventoryConfig(
            horizon=4,
            max_inventory=14,
            max_order=6,
            initial_inventory=4,
        )
        values, policy = exact_risk_neutral_dp(config)
        env = RiskInventoryEnv(config)

        self.assertEqual(values.shape, (5, 15))
        self.assertEqual(policy.shape, (4, 15))
        for t in range(config.horizon):
            for inventory in range(config.max_inventory + 1):
                self.assertIn(
                    int(policy[t, inventory]),
                    env.feasible_actions(inventory),
                )

    def test_base_stock_search_returns_valid_target(self):
        from industrial_rl.risk_inventory import (
            RiskInventoryConfig,
            optimize_base_stock_target,
        )

        config = RiskInventoryConfig(
            horizon=4,
            max_inventory=10,
            max_order=5,
            initial_inventory=3,
        )
        target, metrics = optimize_base_stock_target(
            config,
            objective="cvar",
            episodes=40,
            seed=99,
        )
        self.assertTrue(0 <= target <= config.max_inventory)
        self.assertTrue(np.isfinite(metrics["cvar90_cost"]))


@unittest.skipUnless(
    TORCH_AVAILABLE,
    "PyTorch optional dependency is not installed",
)
class RiskInventoryPPOTests(unittest.TestCase):
    def test_small_tail_weighted_ppo_is_executable(self):
        from industrial_rl.risk_inventory import (
            RiskInventoryConfig,
            RiskInventoryEnv,
            RiskPPOConfig,
            TailWeightedPPOAgent,
        )

        config = RiskInventoryConfig(
            horizon=5,
            max_inventory=14,
            max_order=6,
            initial_inventory=4,
        )
        env = RiskInventoryEnv(config)
        agent = TailWeightedPPOAgent(
            env,
            RiskPPOConfig(
                updates=3,
                rollout_episodes=4,
                ppo_epochs=2,
                minibatch_size=32,
                tail_weight=3.0,
                seed=21,
            ),
        ).fit()

        policy = agent.policy()
        for t in range(config.horizon):
            for inventory in (0, 3, 7, 14):
                self.assertIn(
                    policy(t, inventory),
                    env.feasible_actions(inventory),
                )


if __name__ == "__main__":
    unittest.main()

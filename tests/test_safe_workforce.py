import importlib.util
import unittest

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(
    TORCH_AVAILABLE,
    "PyTorch optional dependency is not installed",
)
class SafeWorkforceTests(unittest.TestCase):
    def test_constraint_signal_is_separate_from_economic_reward(self):
        from industrial_rl.safe_workforce import (
            SafeWorkforceConfig,
            SafeWorkforceEnv,
        )

        config = SafeWorkforceConfig(service_backlog_limit=1)
        env = SafeWorkforceEnv(config)

        next_backlog, reward, constraint_cost, info = env.transition(
            backlog=np.asarray([0, 0, 0]),
            action_index=0,
            workload=np.asarray([8, 8, 8]),
        )

        self.assertTrue(np.any(next_backlog > 1))
        self.assertEqual(constraint_cost, 1.0)
        self.assertAlmostEqual(-reward, info["economic_cost"])
        self.assertNotEqual(info["economic_cost"], constraint_cost)

    def test_shield_repairs_expected_unsafe_action(self):
        from industrial_rl.safe_workforce import (
            SafeWorkforceConfig,
            SafeWorkforceEnv,
            shield_action,
        )

        config = SafeWorkforceConfig(service_backlog_limit=2)
        env = SafeWorkforceEnv(config)

        backlog = np.asarray([3, 3, 3])
        proposed = 0
        repaired = shield_action(
            proposed,
            t=0,
            backlog=backlog,
            config=config,
        )

        self.assertTrue(0 <= repaired < env.n_actions)
        self.assertNotEqual(repaired, proposed)

    def test_constrained_baseline_is_executable(self):
        from industrial_rl.safe_workforce import (
            SafeWorkforceConfig,
            constrained_expected_workload_policy,
            evaluate_safe_workforce_policy,
        )

        config = SafeWorkforceConfig()
        metrics = evaluate_safe_workforce_policy(
            constrained_expected_workload_policy(config),
            config,
            episodes=50,
            seed=2000,
        )

        self.assertTrue(np.isfinite(metrics["mean_economic_cost"]))
        self.assertTrue(
            np.isfinite(metrics["mean_violation_periods"])
        )

    def test_small_primal_dual_ppo_is_executable(self):
        from industrial_rl.safe_workforce import (
            PrimalDualPPOAgent,
            PrimalDualPPOConfig,
            SafeWorkforceConfig,
            SafeWorkforceEnv,
        )

        config = SafeWorkforceConfig(
            workforce=SafeWorkforceConfig().workforce,
            expected_violation_budget=2.0,
        )
        env = SafeWorkforceEnv(config)
        agent = PrimalDualPPOAgent(
            env,
            PrimalDualPPOConfig(
                updates=3,
                rollout_episodes=4,
                ppo_epochs=2,
                minibatch_size=32,
                seed=12,
            ),
        ).fit()

        action = agent.policy()(
            0,
            np.zeros(3, dtype=int),
        )
        self.assertTrue(0 <= action < env.n_actions)
        self.assertTrue(np.isfinite(agent.lagrange_multiplier))
        self.assertGreaterEqual(agent.lagrange_multiplier, 0.0)


if __name__ == "__main__":
    unittest.main()

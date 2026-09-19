import unittest

from industrial_rl import (
    CapacityConfig,
    CapacityEnv,
    LagrangianCapacityQLearner,
    LagrangianQLearningConfig,
    capacity_table_policy,
    evaluate_capacity_policy,
    exact_capacity_dp,
)


class CapacityTests(unittest.TestCase):
    def test_overtime_budget_is_hard_constraint(self):
        config = CapacityConfig(horizon=2, overtime_budget=1, regular_capacity=4, max_production=6)
        env = CapacityEnv(config)
        env.reset(seed=1)
        with self.assertRaises(ValueError):
            env.transition(0, overtime_used=1, action=5, demand=4)

    def test_exact_dp_policy_has_zero_budget_violations(self):
        config = CapacityConfig(horizon=5, overtime_budget=4)
        _, policy = exact_capacity_dp(config)
        metrics = evaluate_capacity_policy(
            capacity_table_policy(policy, config),
            config,
            episodes=500,
            seed=500,
        )
        self.assertEqual(metrics["budget_violation_rate"], 0.0)
        self.assertLessEqual(metrics["mean_overtime"], config.overtime_budget)

    def test_lagrangian_q_learning_produces_feasible_policy(self):
        config = CapacityConfig(horizon=5, overtime_budget=4)
        learner = LagrangianCapacityQLearner(
            CapacityEnv(config),
            LagrangianQLearningConfig(episodes=7000, seed=17),
        ).fit()
        metrics = evaluate_capacity_policy(learner.policy(), config, episodes=500, seed=600)
        self.assertEqual(metrics["budget_violation_rate"], 0.0)
        self.assertGreater(metrics["mean_cost"], 0.0)


if __name__ == "__main__":
    unittest.main()

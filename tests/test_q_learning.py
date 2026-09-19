import unittest

from industrial_rl import (
    InventoryConfig,
    InventoryEnv,
    QLearningConfig,
    TabularQLearner,
    evaluate_inventory_policy,
)


class QLearningTests(unittest.TestCase):
    def test_q_learning_improves_over_never_order(self):
        config = InventoryConfig(horizon=6, max_inventory=14, max_order=7, initial_inventory=4)
        learner = TabularQLearner(
            InventoryEnv(config),
            QLearningConfig(episodes=9000, alpha=0.15, seed=11),
        ).fit()

        learned = evaluate_inventory_policy(learner.policy(), config, episodes=800, seed=9000)
        never = evaluate_inventory_policy(lambda t, inv: 0, config, episodes=800, seed=9000)
        self.assertLess(learned["mean_cost"], never["mean_cost"])

    def test_learned_actions_are_feasible(self):
        config = InventoryConfig(horizon=4, max_inventory=10, max_order=5)
        env = InventoryEnv(config)
        learner = TabularQLearner(
            env,
            QLearningConfig(episodes=2000, seed=3),
        ).fit()
        policy = learner.policy()
        for t in range(config.horizon):
            for inv in range(config.max_inventory + 1):
                self.assertIn(policy(t, inv), env.feasible_actions(inv))


if __name__ == "__main__":
    unittest.main()

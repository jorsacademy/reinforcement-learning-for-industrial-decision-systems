import unittest

import numpy as np

from industrial_rl import (
    InventoryConfig,
    InventoryEnv,
    base_stock_policy,
    evaluate_inventory_policy,
    exact_dynamic_programming,
    table_policy,
)


class InventoryTests(unittest.TestCase):
    def test_transition_accounting(self):
        config = InventoryConfig(horizon=2, max_inventory=10, max_order=5)
        env = InventoryEnv(config)
        next_inv, reward, info = env.transition(inventory=3, action=2, demand=4)
        self.assertEqual(next_inv, 1)
        self.assertEqual(info["sales"], 4)
        self.assertEqual(info["lost_sales"], 0)
        expected_cost = 2 * config.order_cost + config.fixed_order_cost + config.holding_cost
        self.assertAlmostEqual(-reward, expected_cost)

    def test_exact_dp_policy_feasible_and_matches_first_backup(self):
        config = InventoryConfig(horizon=1, max_inventory=8, max_order=4, initial_inventory=2)
        values, policy = exact_dynamic_programming(config)
        env = InventoryEnv(config)
        action = int(policy[0, config.initial_inventory])
        self.assertIn(action, env.feasible_actions(config.initial_inventory))

        expected = 0.0
        for demand, prob in zip(config.demand_values, config.demand_probabilities):
            _, reward, _ = env.transition(config.initial_inventory, action, demand)
            expected += prob * (-reward)
        self.assertAlmostEqual(values[0, config.initial_inventory], expected)

    def test_dp_not_worse_than_simple_base_stock_in_monte_carlo(self):
        config = InventoryConfig(horizon=8)
        _, policy = exact_dynamic_programming(config)
        dp_metrics = evaluate_inventory_policy(table_policy(policy), config, episodes=1200, seed=1000)
        base_metrics = evaluate_inventory_policy(base_stock_policy(config, 7), config, episodes=1200, seed=1000)
        self.assertLessEqual(dp_metrics["mean_cost"], base_metrics["mean_cost"] + 0.5)
        self.assertTrue(0 <= dp_metrics["mean_fill_rate"] <= 1)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from industrial_rl import (
    BehaviorCloningTabular,
    InventoryConfig,
    InventoryEnv,
    PessimisticFQIConfig,
    PessimisticTabularFQI,
    base_stock_policy,
    evaluate_inventory_policy,
    generate_inventory_dataset,
)


class OfflineRLTests(unittest.TestCase):
    def setUp(self):
        self.config = InventoryConfig(horizon=6, max_inventory=14, max_order=7)
        self.env = InventoryEnv(self.config)
        self.behavior = base_stock_policy(self.config, target=7)
        self.data = generate_inventory_dataset(
            self.behavior,
            self.config,
            episodes=900,
            epsilon=0.12,
            seed=100,
        )

    def test_dataset_is_reproducible(self):
        other = generate_inventory_dataset(
            self.behavior,
            self.config,
            episodes=900,
            epsilon=0.12,
            seed=100,
        )
        self.assertEqual(self.data, other)

    def test_behavior_cloning_uses_only_supported_actions(self):
        bc = BehaviorCloningTabular(self.env).fit(self.data)
        policy = bc.policy()
        for t in range(self.config.horizon):
            for inv in range(self.config.max_inventory + 1):
                action = policy(t, inv)
                self.assertIn(action, self.env.feasible_actions(inv))

    def test_pessimistic_fqi_is_finite_and_executable(self):
        model = PessimisticTabularFQI(
            self.env,
            PessimisticFQIConfig(iterations=60, pessimism=2.0),
        ).fit(self.data)
        self.assertTrue(np.isfinite(model.q).all())
        metrics = evaluate_inventory_policy(model.policy(), self.config, episodes=500, seed=1500)
        self.assertTrue(np.isfinite(metrics["mean_cost"]))
        self.assertTrue(0 <= metrics["mean_fill_rate"] <= 1)


if __name__ == "__main__":
    unittest.main()

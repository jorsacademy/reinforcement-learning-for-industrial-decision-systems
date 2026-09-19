import unittest

from industrial_rl import InventoryConfig, InventoryEnv, exact_dynamic_programming


class SmokeTests(unittest.TestCase):
    def test_package_import_and_small_dp(self):
        config = InventoryConfig(horizon=2, max_inventory=6, max_order=3, initial_inventory=2)
        values, policy = exact_dynamic_programming(config)
        env = InventoryEnv(config)
        self.assertEqual(values.shape, (3, 7))
        self.assertEqual(policy.shape, (2, 7))
        self.assertIn(int(policy[0, 2]), env.feasible_actions(2))


if __name__ == "__main__":
    unittest.main()

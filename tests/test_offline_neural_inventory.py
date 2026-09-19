import importlib.util
import unittest

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class OfflineNeuralInventoryCoreTests(unittest.TestCase):
    def setUp(self):
        from industrial_rl.dqn_inventory import (
            RegimeInventoryConfig,
            regime_base_stock_policy,
        )
        from industrial_rl.offline_neural_inventory import (
            generate_regime_inventory_logs,
        )

        self.config = RegimeInventoryConfig(
            horizon=5,
            max_inventory=18,
            max_order=6,
            initial_inventory=5,
        )
        self.behavior = regime_base_stock_policy(
            self.config,
            low_target=7,
            high_target=10,
        )
        self.dataset = generate_regime_inventory_logs(
            self.behavior,
            self.config,
            episodes=70,
            epsilon=0.15,
            seed=222,
        )

    def test_dataset_is_reproducible(self):
        from industrial_rl.offline_neural_inventory import (
            generate_regime_inventory_logs,
        )

        other = generate_regime_inventory_logs(
            self.behavior,
            self.config,
            episodes=70,
            epsilon=0.15,
            seed=222,
        )
        self.assertEqual(self.dataset, other)

    def test_coverage_metrics_are_bounded(self):
        from industrial_rl.offline_neural_inventory import dataset_coverage

        m = dataset_coverage(self.dataset, self.config)
        self.assertGreater(m["transitions"], 0)
        self.assertTrue(0.0 < m["state_coverage"] <= 1.0)
        self.assertTrue(0.0 < m["state_action_coverage"] <= 1.0)
        self.assertGreaterEqual(m["mean_behavior_entropy"], 0.0)

    def test_fqe_behavior_policy_is_finite(self):
        from industrial_rl.offline_neural_inventory import tabular_fqe

        value = tabular_fqe(
            self.dataset,
            self.behavior,
            self.config,
            iterations=30,
        )
        self.assertTrue(np.isfinite(value))


@unittest.skipUnless(
    TORCH_AVAILABLE,
    "PyTorch optional dependency is not installed",
)
class OfflineNeuralInventoryLearnerTests(unittest.TestCase):
    def setUp(self):
        from industrial_rl.dqn_inventory import (
            RegimeInventoryConfig,
            regime_base_stock_policy,
        )
        from industrial_rl.offline_neural_inventory import (
            generate_regime_inventory_logs,
        )

        self.config = RegimeInventoryConfig(
            horizon=5,
            max_inventory=18,
            max_order=6,
            initial_inventory=5,
        )
        behavior = regime_base_stock_policy(
            self.config,
            low_target=7,
            high_target=10,
        )
        self.dataset = generate_regime_inventory_logs(
            behavior,
            self.config,
            episodes=90,
            epsilon=0.20,
            seed=333,
        )

    def _assert_policy_feasible(self, policy):
        from industrial_rl.dqn_inventory import RegimeInventoryEnv

        env = RegimeInventoryEnv(self.config)
        for t in range(self.config.horizon):
            for inventory in (0, 4, 9, 18):
                for regime in (0, 1):
                    self.assertIn(
                        policy(t, inventory, regime),
                        env.feasible_actions(inventory),
                    )

    def test_neural_bc_is_executable(self):
        from industrial_rl.offline_neural_inventory import (
            NeuralBehaviorCloning,
            OfflineBCConfig,
        )

        model = NeuralBehaviorCloning(
            self.config,
            OfflineBCConfig(
                gradient_steps=40,
                batch_size=32,
                seed=4,
            ),
        ).fit(self.dataset)
        self._assert_policy_feasible(model.policy())

    def test_discrete_cql_is_executable(self):
        from industrial_rl.offline_neural_inventory import (
            CQLConfig,
            DiscreteCQL,
        )

        model = DiscreteCQL(
            self.config,
            CQLConfig(
                gradient_steps=50,
                batch_size=32,
                target_update=20,
                seed=5,
            ),
        ).fit(self.dataset)
        self._assert_policy_feasible(model.policy())

    def test_discrete_iql_is_executable(self):
        from industrial_rl.offline_neural_inventory import (
            DiscreteIQL,
            IQLConfig,
        )

        model = DiscreteIQL(
            self.config,
            IQLConfig(
                gradient_steps=50,
                batch_size=32,
                target_update=20,
                seed=6,
            ),
        ).fit(self.dataset)
        self._assert_policy_feasible(model.policy())


if __name__ == "__main__":
    unittest.main()

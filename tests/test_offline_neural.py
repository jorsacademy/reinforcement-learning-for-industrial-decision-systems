import importlib.util
import unittest

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(
    TORCH_AVAILABLE,
    "PyTorch optional dependency is not installed",
)
class OfflineNeuralTests(unittest.TestCase):
    def setUp(self):
        from industrial_rl import (
            InventoryConfig,
            base_stock_policy,
            generate_inventory_dataset,
        )

        self.config = InventoryConfig(
            horizon=5,
            max_inventory=14,
            max_order=6,
            initial_inventory=4,
        )
        self.behavior = base_stock_policy(
            self.config,
            target=7,
        )
        self.dataset = generate_inventory_dataset(
            self.behavior,
            self.config,
            episodes=180,
            epsilon=0.18,
            seed=500,
        )

    def test_diagnostics_are_bounded(self):
        from industrial_rl import InventoryEnv
        from industrial_rl.offline_neural import (
            offline_dataset_diagnostics,
        )

        metrics = offline_dataset_diagnostics(
            self.dataset,
            InventoryEnv(self.config),
            target_policy=self.behavior,
        )

        for key, value in metrics.items():
            self.assertTrue(
                0.0 <= value <= 1.0,
                key,
            )

    def test_offline_cql_fit_does_not_call_env_step(self):
        from industrial_rl import InventoryEnv
        from industrial_rl.offline_neural import (
            ConservativeOfflineConfig,
            ConservativeOfflineDQN,
        )

        env = InventoryEnv(self.config)

        def forbidden_step(action):
            raise AssertionError(
                "offline learner accessed simulator step"
            )

        env.step = forbidden_step

        model = ConservativeOfflineDQN(
            env,
            ConservativeOfflineConfig(
                gradient_steps=80,
                batch_size=32,
                target_update=20,
                seed=9,
            ),
        ).fit(self.dataset)

        self.assertGreater(
            len(model.loss_history),
            0,
        )
        self.assertTrue(
            np.isfinite(model.loss_history).all()
        )

    def test_fqe_fit_does_not_call_env_step(self):
        from industrial_rl import InventoryEnv
        from industrial_rl.offline_neural import (
            FQEConfig,
            TabularFQE,
        )

        env = InventoryEnv(self.config)

        def forbidden_step(action):
            raise AssertionError(
                "FQE accessed simulator step"
            )

        env.step = forbidden_step

        fqe = TabularFQE(
            env,
            FQEConfig(iterations=30),
        ).fit(
            self.dataset,
            self.behavior,
        )

        estimate = fqe.initial_cost(
            self.behavior
        )
        self.assertTrue(
            np.isfinite(estimate)
        )

    def test_conservative_policy_actions_are_feasible(self):
        from industrial_rl import InventoryEnv
        from industrial_rl.offline_neural import (
            ConservativeOfflineConfig,
            ConservativeOfflineDQN,
        )

        env = InventoryEnv(self.config)
        model = ConservativeOfflineDQN(
            env,
            ConservativeOfflineConfig(
                gradient_steps=100,
                batch_size=32,
                target_update=25,
                seed=10,
            ),
        ).fit(self.dataset)

        policy = model.policy()
        for t in range(self.config.horizon):
            for inventory in range(
                self.config.max_inventory + 1
            ):
                self.assertIn(
                    policy(t, inventory),
                    env.feasible_actions(
                        inventory
                    ),
                )


if __name__ == "__main__":
    unittest.main()

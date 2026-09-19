import importlib.util
import unittest

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class OfflineSchedulingCoreTests(unittest.TestCase):
    def test_dynamic_schedule_completes_all_jobs(self):
        from industrial_rl.offline_scheduling import (
            ATC,
            DynamicDispatchEnv,
            SchedulingConfig,
        )

        config = SchedulingConfig(n_jobs=12)
        env = DynamicDispatchEnv(config)
        env.reset(seed=1)
        done = False
        while not done:
            _, _, done, _ = env.step(ATC)

        self.assertEqual(len(env.completed), config.n_jobs)
        self.assertGreaterEqual(env.total_weighted_tardiness, 0.0)
        self.assertGreaterEqual(env.total_setup, 0.0)

    def test_state_vector_is_finite(self):
        from industrial_rl.offline_scheduling import DynamicDispatchEnv

        env = DynamicDispatchEnv()
        state = env.reset(seed=2)
        self.assertEqual(state.shape, (9,))
        self.assertTrue(np.isfinite(state).all())

    def test_lookahead_action_is_valid(self):
        from industrial_rl.offline_scheduling import DynamicDispatchEnv

        env = DynamicDispatchEnv()
        env.reset(seed=3)
        action = env.two_step_lookahead_action()
        self.assertTrue(0 <= action < env.n_actions)

    def test_logged_dataset_is_reproducible(self):
        from industrial_rl.offline_scheduling import (
            SchedulingConfig,
            generate_scheduling_dataset,
        )

        config = SchedulingConfig(n_jobs=10)
        a = generate_scheduling_dataset(
            config,
            episodes=12,
            seed=77,
        )
        b = generate_scheduling_dataset(
            config,
            episodes=12,
            seed=77,
        )
        self.assertEqual(a, b)


@unittest.skipUnless(
    TORCH_AVAILABLE,
    "PyTorch optional dependency is not installed",
)
class OfflineSchedulingNeuralTests(unittest.TestCase):
    def test_small_offline_cql_is_executable(self):
        from industrial_rl.offline_scheduling import (
            OfflineSchedulingCQL,
            SchedulingCQLConfig,
            SchedulingConfig,
            generate_scheduling_dataset,
        )

        dataset = generate_scheduling_dataset(
            SchedulingConfig(n_jobs=10),
            episodes=30,
            seed=88,
        )
        model = OfflineSchedulingCQL(
            SchedulingCQLConfig(
                gradient_steps=60,
                batch_size=32,
                target_update=20,
                seed=12,
            )
        ).fit(dataset)

        self.assertTrue(
            all(
                np.isfinite(p.detach().cpu().numpy()).all()
                for p in model.online.parameters()
            )
        )

    def test_behavior_cloner_is_executable(self):
        from industrial_rl.offline_scheduling import (
            DynamicDispatchEnv,
            SchedulingConfig,
            fit_behavior_cloner,
            generate_scheduling_dataset,
        )

        config = SchedulingConfig(n_jobs=10)
        dataset = generate_scheduling_dataset(
            config,
            episodes=25,
            seed=90,
        )
        _, policy = fit_behavior_cloner(
            dataset,
            epochs=2,
            batch_size=32,
            seed=13,
        )
        env = DynamicDispatchEnv(config)
        env.reset(seed=91)
        action = policy(env)
        self.assertTrue(0 <= action < 4)


if __name__ == "__main__":
    unittest.main()

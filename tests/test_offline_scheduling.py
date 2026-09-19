import importlib.util
import unittest

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class OfflineSchedulingCoreTests(unittest.TestCase):
    def test_event_driven_episode_completes_all_jobs(self):
        from industrial_rl.offline_scheduling import (
            FIFO,
            DynamicDispatchEnv,
            SchedulingConfig,
        )

        config = SchedulingConfig(n_jobs=12)
        env = DynamicDispatchEnv(config)
        state = env.reset(seed=5)

        self.assertEqual(state.shape, (10,))
        steps = 0
        done = False

        while not done:
            state, reward, done, info = env.step(FIFO)
            self.assertTrue(np.isfinite(reward))
            self.assertIn("job_id", info)
            steps += 1

        self.assertEqual(steps, config.n_jobs)
        self.assertEqual(len(env.completed), config.n_jobs)

    def test_fixed_rules_return_valid_actions(self):
        from industrial_rl.offline_scheduling import (
            ATC,
            EDD,
            FIFO,
            SPT,
            fixed_rule_policy,
        )

        state = np.zeros(10, dtype=np.float32)
        for action in (FIFO, EDD, SPT, ATC):
            self.assertEqual(
                fixed_rule_policy(action)(state),
                action,
            )

    def test_logged_dataset_is_reproducible(self):
        from industrial_rl.offline_scheduling import (
            SchedulingConfig,
            generate_dispatch_log,
        )

        config = SchedulingConfig(n_jobs=10)
        a = generate_dispatch_log(
            config,
            episodes=8,
            seed=100,
        )
        b = generate_dispatch_log(
            config,
            episodes=8,
            seed=100,
        )
        self.assertEqual(a, b)

    def test_rolling_horizon_returns_finite_metrics(self):
        from industrial_rl.offline_scheduling import (
            SchedulingConfig,
            evaluate_rolling_horizon,
        )

        metrics = evaluate_rolling_horizon(
            SchedulingConfig(n_jobs=10),
            episodes=5,
            seed=300,
            depth=3,
            candidate_cap=5,
        )
        for value in metrics.values():
            self.assertTrue(np.isfinite(value))


@unittest.skipUnless(
    TORCH_AVAILABLE,
    "PyTorch optional dependency is not installed",
)
class OfflineSchedulingNeuralTests(unittest.TestCase):
    def setUp(self):
        from industrial_rl.offline_scheduling import (
            SchedulingConfig,
            generate_dispatch_log,
        )

        self.config = SchedulingConfig(n_jobs=10)
        self.data = generate_dispatch_log(
            self.config,
            episodes=30,
            epsilon=0.18,
            seed=400,
        )

    def test_bc_and_cql_fit_without_simulator(self):
        from industrial_rl.offline_scheduling import (
            BCConfig,
            BehaviorCloningSelector,
            ConservativeDispatchDQN,
            OfflineDispatchConfig,
        )

        bc = BehaviorCloningSelector(
            config=BCConfig(
                gradient_steps=40,
                batch_size=32,
                seed=1,
            )
        ).fit(self.data)

        cql = ConservativeDispatchDQN(
            config=OfflineDispatchConfig(
                gradient_steps=60,
                batch_size=32,
                target_update=20,
                seed=2,
            )
        ).fit(self.data)

        state = np.asarray(
            self.data[0].state,
            dtype=np.float32,
        )
        self.assertTrue(
            0 <= bc.policy()(state) < 4
        )
        self.assertTrue(
            0 <= cql.policy()(state) < 4
        )

    def test_fqe_is_finite(self):
        from industrial_rl.offline_scheduling import (
            ATC,
            DispatchFQEConfig,
            NeuralDispatchFQE,
            fixed_rule_policy,
        )

        policy = fixed_rule_policy(ATC)
        fqe = NeuralDispatchFQE(
            config=DispatchFQEConfig(
                gradient_steps=50,
                batch_size=32,
                target_update=20,
                seed=3,
            )
        ).fit(
            self.data,
            policy,
        )
        self.assertTrue(
            np.isfinite(
                fqe.initial_cost(
                    self.data,
                    policy,
                )
            )
        )

    def test_probability_support_guard_intervenes(self):
        from industrial_rl.offline_scheduling import (
            BCConfig,
            BehaviorCloningSelector,
            ProbabilitySupportGuard,
        )

        bc = BehaviorCloningSelector(
            config=BCConfig(
                gradient_steps=40,
                batch_size=32,
                seed=4,
            )
        ).fit(self.data)

        def rare_policy(state):
            probs = bc.probabilities(state)
            return int(np.argmin(probs))

        guard = ProbabilitySupportGuard(
            rare_policy,
            bc,
            threshold=0.99,
        )
        state = np.asarray(
            self.data[0].state,
            dtype=np.float32,
        )
        action = guard(state)

        self.assertTrue(0 <= action < 4)
        self.assertEqual(guard.interventions, 1)


if __name__ == "__main__":
    unittest.main()

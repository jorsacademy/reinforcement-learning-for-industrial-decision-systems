import importlib.util
import unittest

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch optional dependency is not installed")
class PPOWorkforceTests(unittest.TestCase):
    def test_action_space_respects_flex_pool(self):
        from industrial_rl.ppo_workforce import WorkforceConfig, WorkforceEnv

        config = WorkforceConfig(flex_pool=3)
        env = WorkforceEnv(config)
        self.assertGreater(env.n_actions, 1)
        self.assertTrue(all(sum(a) <= config.flex_pool for a in env.actions))

    def test_myopic_policy_and_small_ppo_are_executable(self):
        from industrial_rl.ppo_workforce import (
            PPOConfig,
            PPOWorkforceAgent,
            WorkforceConfig,
            WorkforceEnv,
            evaluate_workforce_policy,
            myopic_expected_cost_policy,
        )

        config = WorkforceConfig(horizon=6)
        baseline = evaluate_workforce_policy(
            myopic_expected_cost_policy(config),
            config,
            episodes=60,
            seed=1200,
        )
        self.assertTrue(np.isfinite(baseline["mean_cost"]))

        env = WorkforceEnv(config)
        agent = PPOWorkforceAgent(
            env,
            PPOConfig(
                updates=3,
                rollout_episodes=4,
                ppo_epochs=2,
                minibatch_size=32,
                seed=7,
            ),
        ).fit()
        policy = agent.policy()
        action = policy(0, np.zeros(3, dtype=int))
        self.assertTrue(0 <= action < env.n_actions)


if __name__ == "__main__":
    unittest.main()

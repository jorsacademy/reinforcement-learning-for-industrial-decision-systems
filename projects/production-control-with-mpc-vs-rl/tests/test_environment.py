import numpy as np

from production_control.environment import ProductionControlEnv


def test_reset_is_seed_reproducible():
    env1 = ProductionControlEnv()
    env2 = ProductionControlEnv()
    obs1, _ = env1.reset(seed=7)
    obs2, _ = env2.reset(seed=7)
    assert np.allclose(obs1, obs2)


def test_step_respects_spaces_and_horizon():
    env = ProductionControlEnv()
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    done = False
    steps = 0
    while not done:
        obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert info["wip"] >= 0.0
        assert info["backlog"] >= 0.0
        steps += 1
        done = terminated or truncated
    assert steps == env.config.horizon


def test_action_is_clipped():
    env = ProductionControlEnv()
    env.reset(seed=2)
    env.step(np.array([5.0], dtype=np.float32))
    assert env.prev_control == 1.0

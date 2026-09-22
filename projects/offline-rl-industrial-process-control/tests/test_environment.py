import numpy as np

from offline_process_control.environment import IndustrialProcessEnv


def test_reset_and_step_contract():
    env = IndustrialProcessEnv(horizon=5)
    obs, info = env.reset(seed=7)
    assert obs.shape == (5,)
    assert env.observation_space.contains(obs)
    next_obs, reward, terminated, truncated, info = env.step(np.array([0.2], dtype=np.float32))
    assert env.observation_space.contains(next_obs)
    assert isinstance(reward, float)
    assert not truncated


def test_seed_is_deterministic():
    a = IndustrialProcessEnv(); b = IndustrialProcessEnv()
    obs_a, _ = a.reset(seed=42); obs_b, _ = b.reset(seed=42)
    np.testing.assert_allclose(obs_a, obs_b)
    step_a = a.step(np.array([0.1], dtype=np.float32))
    step_b = b.step(np.array([0.1], dtype=np.float32))
    np.testing.assert_allclose(step_a[0], step_b[0])
    assert step_a[1] == step_b[1]

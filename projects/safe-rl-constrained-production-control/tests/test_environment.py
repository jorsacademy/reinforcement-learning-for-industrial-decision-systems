import numpy as np

from safe_production_control.environment import SafeProductionControlEnv


def test_reset_is_seeded_and_valid():
    env = SafeProductionControlEnv(horizon=5)
    obs1, _ = env.reset(seed=7)
    obs2, _ = env.reset(seed=7)
    assert env.observation_space.contains(obs1)
    np.testing.assert_allclose(obs1, obs2)


def test_episode_truncates_at_horizon():
    env = SafeProductionControlEnv(horizon=4)
    env.reset(seed=1)
    truncated = False
    for _ in range(4):
        _, _, _, truncated, _ = env.step(np.array([0.4], dtype=np.float32))
    assert truncated


def test_constraint_cost_detects_temperature_excess():
    env = SafeProductionControlEnv()
    env.reset(seed=2)
    env.temperature = 0.95
    _, _, _, _, info = env.step(np.array([0.2], dtype=np.float32))
    assert info["temperature_violation"] > 0.0
    assert info["constraint_violated"]


def test_action_is_clipped_to_physical_range():
    env = SafeProductionControlEnv()
    env.reset(seed=3)
    obs, _, _, _, _ = env.step(np.array([4.0], dtype=np.float32))
    assert 0.0 <= obs[-1] <= 1.0

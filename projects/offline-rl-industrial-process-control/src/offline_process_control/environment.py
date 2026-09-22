"""Continuous industrial process-control environment.

The process is intentionally compact but captures the core offline-RL issue:
a controller must regulate a quality variable around a target while limiting
energy use and aggressive actuator movement. Historical data can be collected
from a safe behavior controller and reused without online exploration.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class IndustrialProcessEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, horizon: int = 120):
        super().__init__()
        self.horizon = int(horizon)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-3.0, -3.0, -3.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([3.0, 3.0, 3.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.state = np.zeros(3, dtype=np.float32)
        self.prev_action = 0.0
        self.t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.state = self.np_random.normal(0.0, 0.15, size=3).astype(np.float32)
        self.prev_action = 0.0
        self.t = 0
        return self._obs(), self._info(0.0, 0.0, 0.0)

    def step(self, action):
        u = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        x1, x2, x3 = map(float, self.state)

        disturbance = float(self.np_random.normal(0.0, 0.035))
        next_x1 = 0.82 * x1 + 0.18 * x2 + 0.20 * u + disturbance
        next_x2 = 0.10 * x1 + 0.86 * x2 + 0.08 * u + 0.5 * disturbance
        next_x3 = 0.90 * x3 + 0.06 * x2 + 0.04 * u
        self.state = np.clip([next_x1, next_x2, next_x3], -3.0, 3.0).astype(np.float32)

        quality_cost = float(self.state[0] ** 2 + 0.35 * self.state[1] ** 2)
        energy_cost = float(0.08 * u * u)
        smoothness_cost = float(0.04 * (u - self.prev_action) ** 2)
        total_cost = quality_cost + energy_cost + smoothness_cost
        reward = -total_cost

        self.prev_action = u
        self.t += 1
        terminated = self.t >= self.horizon
        truncated = False
        return self._obs(), reward, terminated, truncated, self._info(
            quality_cost, energy_cost, smoothness_cost
        )

    def _obs(self) -> np.ndarray:
        return np.array(
            [self.state[0], self.state[1], self.state[2], self.prev_action, self.t / self.horizon],
            dtype=np.float32,
        )

    def _info(self, quality_cost: float, energy_cost: float, smoothness_cost: float) -> dict:
        return {
            "quality_cost": float(quality_cost),
            "energy_cost": float(energy_cost),
            "smoothness_cost": float(smoothness_cost),
            "total_cost": float(quality_cost + energy_cost + smoothness_cost),
            "quality_deviation": float(abs(self.state[0])),
            "action": float(self.prev_action),
        }

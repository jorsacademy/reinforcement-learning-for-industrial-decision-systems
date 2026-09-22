"""Continuous industrial production-control environment.

The controller chooses a normalized production-rate command in [0, 1]. The
plant converts that command into completed production subject to finite WIP,
stochastic demand and energy costs. The same dynamics are used by heuristic,
MPC and reinforcement-learning controllers so comparisons are meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class PlantConfig:
    horizon: int = 168
    max_rate: float = 12.0
    max_wip: float = 45.0
    target_wip: float = 18.0
    demand_mean: float = 8.0
    demand_amplitude: float = 2.5
    demand_noise: float = 0.8
    energy_base: float = 0.08
    energy_peak: float = 0.22
    holding_cost: float = 0.08
    backlog_cost: float = 0.55
    energy_weight: float = 1.0
    smoothness_weight: float = 0.35


class ProductionControlEnv(gym.Env):
    """Single-line aggregate production-control benchmark.

    Observation: [WIP, backlog, demand, energy price, previous control, time].
    Action: normalized production-rate command in [0, 1].
    """

    metadata = {"render_modes": []}

    def __init__(self, config: PlantConfig | None = None):
        super().__init__()
        self.config = config or PlantConfig()
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )
        self.t = 0
        self.wip = 0.0
        self.backlog = 0.0
        self.demand = 0.0
        self.energy_price = 0.0
        self.prev_control = 0.0
        self.total_throughput = 0.0
        self.total_energy_cost = 0.0
        self.total_holding_cost = 0.0
        self.total_backlog_cost = 0.0
        self.total_smoothness_cost = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        cfg = self.config
        self.t = 0
        self.wip = cfg.target_wip
        self.backlog = 0.0
        self.prev_control = cfg.demand_mean / cfg.max_rate
        self.total_throughput = 0.0
        self.total_energy_cost = 0.0
        self.total_holding_cost = 0.0
        self.total_backlog_cost = 0.0
        self.total_smoothness_cost = 0.0
        self._sample_exogenous()
        return self._observation(), self._info()

    def step(self, action):
        cfg = self.config
        u = float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], 0.0, 1.0))
        requested = u * cfg.max_rate

        # New demand becomes backlog before fulfillment.
        self.backlog += self.demand

        # Production replenishes WIP; shipment is constrained by available stock.
        produced = min(requested, max(0.0, cfg.max_wip - self.wip))
        self.wip += produced
        shipped = min(self.wip, self.backlog)
        self.wip -= shipped
        self.backlog -= shipped

        energy_cost = self.energy_price * (0.25 * requested + 0.75 * requested**2 / cfg.max_rate)
        holding_cost = cfg.holding_cost * self.wip
        backlog_cost = cfg.backlog_cost * self.backlog
        smoothness_cost = cfg.smoothness_weight * abs(u - self.prev_control)

        step_cost = (
            holding_cost
            + backlog_cost
            + cfg.energy_weight * energy_cost
            + smoothness_cost
        )
        reward = -float(step_cost)

        self.total_throughput += shipped
        self.total_energy_cost += energy_cost
        self.total_holding_cost += holding_cost
        self.total_backlog_cost += backlog_cost
        self.total_smoothness_cost += smoothness_cost
        self.prev_control = u
        self.t += 1
        terminated = False
        truncated = self.t >= cfg.horizon
        if not truncated:
            self._sample_exogenous()

        return self._observation(), reward, terminated, truncated, self._info()

    def _sample_exogenous(self) -> None:
        cfg = self.config
        phase = 2.0 * np.pi * (self.t % 24) / 24.0
        mean = cfg.demand_mean + cfg.demand_amplitude * np.sin(phase - 0.6)
        self.demand = max(0.0, float(mean + self.np_random.normal(0.0, cfg.demand_noise)))
        hour = self.t % 24
        peak = 9 <= hour < 18
        self.energy_price = cfg.energy_peak if peak else cfg.energy_base

    def _observation(self) -> np.ndarray:
        cfg = self.config
        obs = np.array(
            [
                self.wip / cfg.max_wip,
                min(self.backlog / (2.0 * cfg.max_wip), 1.0),
                min(self.demand / (1.5 * cfg.max_rate), 1.0),
                self.energy_price / cfg.energy_peak,
                self.prev_control,
                self.t / cfg.horizon,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, 0.0, 1.0)

    def _info(self) -> dict:
        cfg = self.config
        utilization_proxy = self.total_throughput / max(1.0, self.t * cfg.max_rate)
        total_cost = (
            self.total_energy_cost
            + self.total_holding_cost
            + self.total_backlog_cost
            + self.total_smoothness_cost
        )
        return {
            "time": self.t,
            "wip": self.wip,
            "backlog": self.backlog,
            "demand": self.demand,
            "energy_price": self.energy_price,
            "throughput": self.total_throughput,
            "utilization_proxy": utilization_proxy,
            "energy_cost": self.total_energy_cost,
            "holding_cost": self.total_holding_cost,
            "backlog_cost": self.total_backlog_cost,
            "smoothness_cost": self.total_smoothness_cost,
            "total_cost": total_cost,
        }

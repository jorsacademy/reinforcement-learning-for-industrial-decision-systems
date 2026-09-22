from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class SafetyLimits:
    max_temperature: float = 0.85
    max_wip: float = 0.90
    max_utilization: float = 0.95


class SafeProductionControlEnv(gym.Env):
    """Continuous production control with explicit safety constraints.

    Observation: [wip, backlog, temperature, demand, energy_price, previous_action].
    Action: normalized production command in [0, 1].
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, horizon: int = 168, limits: SafetyLimits | None = None):
        super().__init__()
        self.horizon = int(horizon)
        self.limits = limits or SafetyLimits()
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(0.0, 1.5, shape=(6,), dtype=np.float32)
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.wip = float(self.np_random.uniform(0.25, 0.45))
        self.backlog = float(self.np_random.uniform(0.15, 0.30))
        self.temperature = float(self.np_random.uniform(0.35, 0.50))
        self.previous_action = 0.5
        self.total_cost = 0.0
        self.constraint_violations = 0
        self.violation_severity = 0.0
        self.throughput = 0.0
        self.energy_cost = 0.0
        self._sample_exogenous()
        return self._obs(), self._info()

    def _sample_exogenous(self):
        hour = self.t % 24
        day_peak = np.exp(-0.5 * ((hour - 14.0) / 4.0) ** 2)
        self.demand = float(np.clip(0.45 + 0.28 * day_peak + self.np_random.normal(0, 0.05), 0.15, 1.0))
        self.energy_price = float(np.clip(0.25 + 0.45 * day_peak + self.np_random.normal(0, 0.03), 0.10, 1.0))

    def _obs(self):
        return np.asarray([
            self.wip,
            self.backlog,
            self.temperature,
            self.demand,
            self.energy_price,
            self.previous_action,
        ], dtype=np.float32)

    def _constraint_cost(self, utilization: float) -> tuple[float, dict]:
        temp_excess = max(0.0, self.temperature - self.limits.max_temperature)
        wip_excess = max(0.0, self.wip - self.limits.max_wip)
        util_excess = max(0.0, utilization - self.limits.max_utilization)
        severity = temp_excess + wip_excess + util_excess
        violated = severity > 0.0
        return severity, {
            "temperature_violation": temp_excess,
            "wip_violation": wip_excess,
            "utilization_violation": util_excess,
            "constraint_violated": violated,
        }

    def step(self, action):
        u = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        available = min(self.wip + 0.18, 1.2)
        produced = min(0.24 * u, available)
        utilization = float(np.clip(produced / 0.24, 0.0, 1.0))

        arrivals = float(np.clip(0.16 + self.np_random.normal(0, 0.02), 0.08, 0.24))
        self.wip = float(np.clip(self.wip + arrivals - produced, 0.0, 1.4))
        served = min(self.backlog + self.demand * 0.16, produced)
        self.backlog = float(np.clip(self.backlog + self.demand * 0.16 - served, 0.0, 1.4))
        self.throughput += produced

        cooling = 0.055
        thermal_load = 0.095 * (u ** 2)
        self.temperature = float(np.clip(self.temperature + thermal_load - cooling + self.np_random.normal(0, 0.008), 0.0, 1.4))

        holding_cost = 1.2 * self.wip
        backlog_cost = 4.5 * self.backlog
        energy = self.energy_price * (0.35 + 1.9 * u * u)
        smoothness = 0.5 * abs(u - self.previous_action)
        constraint_cost, violation_info = self._constraint_cost(utilization)
        safety_penalty = 35.0 * constraint_cost
        period_cost = holding_cost + backlog_cost + energy + smoothness + safety_penalty

        self.energy_cost += energy
        self.total_cost += period_cost
        if violation_info["constraint_violated"]:
            self.constraint_violations += 1
            self.violation_severity += constraint_cost

        self.previous_action = u
        self.t += 1
        truncated = self.t >= self.horizon
        self._sample_exogenous()
        info = self._info() | violation_info | {
            "constraint_cost": constraint_cost,
            "utilization": utilization,
        }
        return self._obs(), -float(period_cost), False, truncated, info

    def _info(self):
        return {
            "wip": self.wip,
            "backlog": self.backlog,
            "temperature": self.temperature,
            "throughput": self.throughput,
            "energy_cost": self.energy_cost,
            "total_cost": self.total_cost,
            "constraint_violations": self.constraint_violations,
            "violation_severity": self.violation_severity,
        }

    def render(self):
        return f"t={self.t} wip={self.wip:.3f} backlog={self.backlog:.3f} temp={self.temperature:.3f}"

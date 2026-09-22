from __future__ import annotations

import numpy as np


class SafetyRuleController:
    """Conservative heuristic that throttles production near safety limits."""

    def act(self, obs):
        wip, backlog, temperature, demand, energy_price, previous_action = map(float, obs)
        target = 0.45 + 0.45 * backlog + 0.20 * demand
        if temperature > 0.78:
            target = min(target, 0.35)
        if wip > 0.82:
            target = min(target, 0.45)
        if energy_price > 0.75 and backlog < 0.45:
            target -= 0.12
        target = 0.65 * previous_action + 0.35 * target
        return np.asarray([np.clip(target, 0.0, 0.93)], dtype=np.float32)


class AggressiveController:
    """Throughput-seeking reference policy useful for exposing safety trade-offs."""

    def act(self, obs):
        backlog = float(obs[1])
        demand = float(obs[3])
        return np.asarray([np.clip(0.72 + 0.25 * backlog + 0.08 * demand, 0.0, 1.0)], dtype=np.float32)

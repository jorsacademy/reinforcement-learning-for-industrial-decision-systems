from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InventoryConfig:
    horizon: int = 60
    demand_rate: float = 5.0
    initial_inventory: int = 12
    holding_cost: float = 1.0
    lost_sales_cost: float = 8.0
    order_cost: float = 2.0

    def validate(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.demand_rate <= 0:
            raise ValueError("demand_rate must be positive")
        if self.initial_inventory < 0:
            raise ValueError("initial_inventory must be non-negative")
        if min(self.holding_cost, self.lost_sales_cost, self.order_cost) < 0:
            raise ValueError("costs must be non-negative")


def evaluate_policy(
    reorder_point: int,
    order_quantity: int,
    *,
    config: InventoryConfig | None = None,
    seeds: tuple[int, ...] = (101, 202, 303, 404),
) -> float:
    """Returns mean total cost for an integer (r, Q) inventory policy."""
    cfg = config or InventoryConfig()
    cfg.validate()
    if reorder_point < 0:
        raise ValueError("reorder_point must be non-negative")
    if order_quantity <= 0:
        raise ValueError("order_quantity must be positive")
    if not seeds:
        raise ValueError("at least one seed is required")

    costs: list[float] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        inventory = cfg.initial_inventory
        total = 0.0
        for _ in range(cfg.horizon):
            if inventory <= reorder_point:
                inventory += order_quantity
                total += cfg.order_cost * order_quantity
            demand = int(rng.poisson(cfg.demand_rate))
            sales = min(inventory, demand)
            lost = demand - sales
            inventory -= sales
            total += cfg.holding_cost * inventory + cfg.lost_sales_cost * lost
        costs.append(total)
    return float(np.mean(costs))

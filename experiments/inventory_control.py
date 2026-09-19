from __future__ import annotations

from industrial_rl import (
    InventoryConfig,
    InventoryEnv,
    QLearningConfig,
    TabularQLearner,
    base_stock_policy,
    evaluate_inventory_policy,
    exact_dynamic_programming,
    table_policy,
)


def main():
    config = InventoryConfig()
    values, dp_table = exact_dynamic_programming(config)
    dp = table_policy(dp_table)
    base_stock = base_stock_policy(config, target=9)

    learner = TabularQLearner(
        InventoryEnv(config),
        QLearningConfig(episodes=30_000, seed=42),
    ).fit()

    methods = {
        "Exact DP": dp,
        "Base-stock(9)": base_stock,
        "Tabular Q-learning": learner.policy(),
    }

    print("Inventory control benchmark")
    print(f"Exact expected cost from initial state: {values[0, config.initial_inventory]:.3f}")
    print()
    print(f"{'method':<22}{'mean cost':>12}{'p90 cost':>12}{'fill rate':>12}{'stockout':>12}")
    for name, policy in methods.items():
        metrics = evaluate_inventory_policy(policy, config, episodes=2500, seed=20_000)
        print(
            f"{name:<22}{metrics['mean_cost']:>12.3f}{metrics['p90_cost']:>12.3f}"
            f"{metrics['mean_fill_rate']:>12.4f}{metrics['mean_stockout_period_rate']:>12.4f}"
        )


if __name__ == "__main__":
    main()

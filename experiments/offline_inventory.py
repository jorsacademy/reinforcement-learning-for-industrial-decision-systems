from __future__ import annotations

from industrial_rl import (
    BehaviorCloningTabular,
    InventoryConfig,
    InventoryEnv,
    PessimisticFQIConfig,
    PessimisticTabularFQI,
    base_stock_policy,
    evaluate_inventory_policy,
    exact_dynamic_programming,
    generate_inventory_dataset,
    table_policy,
)


def main():
    config = InventoryConfig()
    env = InventoryEnv(config)
    behavior = base_stock_policy(config, target=7)
    dataset = generate_inventory_dataset(
        behavior,
        config,
        episodes=1200,
        epsilon=0.12,
        seed=7000,
    )

    bc = BehaviorCloningTabular(env).fit(dataset)
    offline_q = PessimisticTabularFQI(
        env,
        PessimisticFQIConfig(iterations=45, pessimism=2.0),
    ).fit(dataset)
    _, dp_table = exact_dynamic_programming(config)

    methods = {
        "Behavior policy": behavior,
        "Behavior cloning": bc.policy(),
        "Pessimistic offline FQI": offline_q.policy(),
        "Exact DP reference": table_policy(dp_table),
    }

    print("Offline inventory-learning benchmark")
    print(f"Logged transitions: {len(dataset)}")
    print("Offline learners see only the logged transition table during fitting.")
    print()
    print(f"{'method':<26}{'mean cost':>12}{'p90 cost':>12}{'fill rate':>12}{'stockout':>12}")
    for name, policy in methods.items():
        metrics = evaluate_inventory_policy(policy, config, episodes=1200, seed=40_000)
        print(
            f"{name:<26}{metrics['mean_cost']:>12.3f}{metrics['p90_cost']:>12.3f}"
            f"{metrics['mean_fill_rate']:>12.4f}{metrics['mean_stockout_period_rate']:>12.4f}"
        )


if __name__ == "__main__":
    main()

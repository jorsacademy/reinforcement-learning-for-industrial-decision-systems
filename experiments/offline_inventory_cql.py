from __future__ import annotations

import argparse

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
from industrial_rl.offline_neural import (
    ActionSupportGuard,
    ConservativeOfflineConfig,
    ConservativeOfflineDQN,
    FQEConfig,
    TabularFQE,
    offline_dataset_diagnostics,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    config = InventoryConfig()
    env = InventoryEnv(config)

    behavior = base_stock_policy(config, target=7)
    dataset = generate_inventory_dataset(
        behavior,
        config,
        episodes=700 if args.smoke else 3000,
        epsilon=0.15,
        seed=170_000,
    )

    bc = BehaviorCloningTabular(
        InventoryEnv(config)
    ).fit(dataset)

    tabular = PessimisticTabularFQI(
        InventoryEnv(config),
        PessimisticFQIConfig(
            iterations=70 if args.smoke else 120,
            pessimism=2.0,
        ),
    ).fit(dataset)

    cql = ConservativeOfflineDQN(
        InventoryEnv(config),
        ConservativeOfflineConfig(
            gradient_steps=1000 if args.smoke else 5000,
            batch_size=128,
            conservative_alpha=1.0,
            seed=707,
        ),
    ).fit(dataset)

    _, exact_table = exact_dynamic_programming(config)

    cql_policy = cql.policy()
    guarded_cql = ActionSupportGuard(
        cql_policy,
        behavior,
        dataset,
        InventoryEnv(config),
        min_count=1,
    )

    methods = [
        ("Behavior policy", behavior, None),
        ("Behavior cloning", bc.policy(), None),
        ("Pessimistic tabular FQI", tabular.policy(), None),
        ("Conservative offline DQN", cql_policy, None),
        ("CQL + support guard", guarded_cql, guarded_cql),
        ("Exact DP reference", table_policy(exact_table), None),
    ]

    eval_episodes = 500 if args.smoke else 1800

    print("Offline inventory CQL-style benchmark")
    print(f"Logged transitions: {len(dataset)}")
    print()
    print(
        f"{'method':<27}{'support':>10}{'FQE cost':>12}"
        f"{'sim cost':>12}{'p90':>11}{'fill':>10}{'stockout':>10}"
        f"{'guard':>9}"
    )

    for name, policy, guard in methods:
        diagnostics = offline_dataset_diagnostics(
            dataset,
            env,
            target_policy=policy,
        )
        fqe = TabularFQE(
            InventoryEnv(config),
            FQEConfig(iterations=90),
        ).fit(
            dataset,
            policy,
        )
        fqe_cost = fqe.initial_cost(policy)

        if guard is not None:
            guard.reset_stats()

        realized = evaluate_inventory_policy(
            policy,
            config,
            episodes=eval_episodes,
            seed=175_000,
        )
        guard_rate = (
            guard.intervention_rate
            if guard is not None
            else 0.0
        )

        print(
            f"{name:<27}"
            f"{diagnostics['target_action_support_rate']:>10.3f}"
            f"{fqe_cost:>12.3f}"
            f"{realized['mean_cost']:>12.3f}"
            f"{realized['p90_cost']:>11.3f}"
            f"{realized['mean_fill_rate']:>10.4f}"
            f"{realized['mean_stockout_period_rate']:>10.4f}"
            f"{guard_rate:>9.3f}"
        )

    coverage = offline_dataset_diagnostics(
        dataset,
        env,
    )
    print()
    print(
        "Dataset coverage: "
        f"states={coverage['state_coverage']:.3f}, "
        f"state-actions={coverage['state_action_coverage']:.3f}"
    )
    print(
        "FQE is fitted only on logged transitions. "
        "Simulator evaluation is performed only after policies are frozen."
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

from industrial_rl.dqn_inventory import (
    RegimeInventoryConfig,
    evaluate_regime_inventory_policy,
    exact_regime_inventory_dp,
    regime_base_stock_policy,
    regime_table_policy,
)
from industrial_rl.offline_neural_inventory import (
    CQLConfig,
    DiscreteCQL,
    DiscreteIQL,
    IQLConfig,
    NeuralBehaviorCloning,
    OfflineBCConfig,
    dataset_coverage,
    generate_regime_inventory_logs,
    tabular_fqe,
    unsupported_policy_action_rate,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    config = RegimeInventoryConfig()
    behavior = regime_base_stock_policy(
        config,
        low_target=9,
        high_target=14,
    )

    log_episodes = 350 if args.smoke else 1200
    eval_episodes = 300 if args.smoke else 1200

    dataset = generate_regime_inventory_logs(
        behavior,
        config,
        episodes=log_episodes,
        epsilon=0.15,
        seed=170_000,
    )
    coverage = dataset_coverage(dataset, config)

    bc = NeuralBehaviorCloning(
        config,
        OfflineBCConfig(
            gradient_steps=250 if args.smoke else 1200,
            batch_size=128,
            seed=707,
        ),
    ).fit(dataset)

    cql = DiscreteCQL(
        config,
        CQLConfig(
            gradient_steps=450 if args.smoke else 1800,
            batch_size=128,
            conservative_weight=1.0,
            target_update=75 if args.smoke else 100,
            seed=708,
        ),
    ).fit(dataset)

    iql = DiscreteIQL(
        config,
        IQLConfig(
            gradient_steps=500 if args.smoke else 2000,
            batch_size=128,
            expectile=0.70,
            temperature=2.0,
            target_update=75 if args.smoke else 100,
            seed=709,
        ),
    ).fit(dataset)

    _, dp_table = exact_regime_inventory_dp(config)
    methods = {
        "Behavior policy": behavior,
        "Neural BC": bc.policy(),
        "Discrete CQL": cql.policy(),
        "Discrete IQL": iql.policy(),
        "Exact DP reference": regime_table_policy(dp_table),
    }

    print("Offline neural inventory benchmark")
    print(
        f"Logged transitions: {int(coverage['transitions'])}; "
        f"state coverage={coverage['state_coverage']:.4f}; "
        f"state-action coverage={coverage['state_action_coverage']:.4f}; "
        f"behavior entropy={coverage['mean_behavior_entropy']:.4f}"
    )
    print(
        "BC/CQL/IQL fit only from the fixed logged transition table. "
        "The simulator is used only after policies are frozen."
    )
    print()
    print(
        f"{'method':<21}{'mean cost':>11}{'p90':>10}{'fill':>9}"
        f"{'stockout':>10}{'FQE cost':>11}{'unsupported':>13}"
    )

    for name, policy in methods.items():
        m = evaluate_regime_inventory_policy(
            policy,
            config,
            episodes=eval_episodes,
            seed=175_000,
        )
        unsupported = unsupported_policy_action_rate(
            dataset,
            policy,
        )
        fqe = tabular_fqe(
            dataset,
            policy,
            config,
            iterations=60,
        )
        print(
            f"{name:<21}"
            f"{m['mean_cost']:>11.3f}"
            f"{m['p90_cost']:>10.3f}"
            f"{m['mean_fill_rate']:>9.4f}"
            f"{m['mean_stockout_period_rate']:>10.4f}"
            f"{fqe:>11.3f}"
            f"{unsupported:>13.4f}"
        )


if __name__ == "__main__":
    main()

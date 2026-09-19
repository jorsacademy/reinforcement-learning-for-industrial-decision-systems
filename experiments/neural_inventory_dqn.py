from __future__ import annotations

import argparse

from industrial_rl.dqn_inventory import (
    DQNConfig,
    DQNInventoryAgent,
    RegimeInventoryConfig,
    RegimeInventoryEnv,
    evaluate_regime_inventory_policy,
    exact_regime_inventory_dp,
    regime_base_stock_policy,
    regime_table_policy,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    config = RegimeInventoryConfig()
    values, table = exact_regime_inventory_dp(config)
    dp = regime_table_policy(table)
    baseline = regime_base_stock_policy(config)

    episodes = 700 if args.smoke else 3500
    eval_episodes = 300 if args.smoke else 1500
    agent = DQNInventoryAgent(
        RegimeInventoryEnv(config),
        DQNConfig(episodes=episodes, warmup=200 if args.smoke else 500, seed=101),
    ).fit()

    methods = {
        "Exact regime-DP": dp,
        "Regime base-stock": baseline,
        "DQN": agent.policy(),
    }

    print("Regime-switching inventory DQN benchmark")
    print(
        "Exact expected cost from initial state: "
        f"{values[0, config.initial_inventory, config.initial_regime]:.3f}"
    )
    print()
    print(f"{'method':<22}{'mean cost':>12}{'p90 cost':>12}{'fill rate':>12}{'stockout':>12}")
    for name, policy in methods.items():
        m = evaluate_regime_inventory_policy(
            policy, config, episodes=eval_episodes, seed=70_000
        )
        print(
            f"{name:<22}{m['mean_cost']:>12.3f}{m['p90_cost']:>12.3f}"
            f"{m['mean_fill_rate']:>12.4f}{m['mean_stockout_period_rate']:>12.4f}"
        )


if __name__ == "__main__":
    main()

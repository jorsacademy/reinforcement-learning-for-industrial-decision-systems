from __future__ import annotations

import argparse

from industrial_rl.risk_inventory import (
    RiskInventoryConfig,
    RiskInventoryEnv,
    RiskPPOConfig,
    TailWeightedPPOAgent,
    base_stock_policy,
    evaluate_risk_inventory_policy,
    exact_risk_neutral_dp,
    optimize_base_stock_target,
    table_policy,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    config = RiskInventoryConfig()

    search_episodes = 250 if args.smoke else 1200
    eval_episodes = 300 if args.smoke else 1500
    updates = 14 if args.smoke else 140
    rollout_episodes = 12 if args.smoke else 32
    ppo_epochs = 3 if args.smoke else 4

    values, dp_table = exact_risk_neutral_dp(config)
    mean_target, _ = optimize_base_stock_target(
        config,
        objective="mean",
        episodes=search_episodes,
        seed=121_000,
    )
    cvar_target, _ = optimize_base_stock_target(
        config,
        objective="cvar",
        episodes=search_episodes,
        seed=121_000,
    )

    mean_ppo = TailWeightedPPOAgent(
        RiskInventoryEnv(config),
        RiskPPOConfig(
            updates=updates,
            rollout_episodes=rollout_episodes,
            ppo_epochs=ppo_epochs,
            minibatch_size=128 if args.smoke else 256,
            tail_weight=0.0,
            seed=505,
        ),
    ).fit()

    cvar_ppo = TailWeightedPPOAgent(
        RiskInventoryEnv(config),
        RiskPPOConfig(
            updates=updates,
            rollout_episodes=rollout_episodes,
            ppo_epochs=ppo_epochs,
            minibatch_size=128 if args.smoke else 256,
            tail_weight=4.0,
            seed=506,
        ),
    ).fit()

    methods = {
        "Risk-neutral DP": table_policy(dp_table),
        f"Mean base-stock({mean_target})": base_stock_policy(
            config,
            mean_target,
        ),
        f"CVaR base-stock({cvar_target})": base_stock_policy(
            config,
            cvar_target,
        ),
        "Mean PPO": mean_ppo.policy(),
        "Tail-weighted PPO": cvar_ppo.policy(),
    }

    print("Tail-risk / CVaR-aware inventory benchmark")
    print(
        "Exact risk-neutral expected cost from initial state: "
        f"{values[0, config.initial_inventory]:.3f}"
    )
    print(
        f"Selected base-stock targets: mean={mean_target}, "
        f"CVaR90={cvar_target}"
    )
    print()
    print(
        f"{'method':<24}{'mean':>10}{'p90':>10}{'p95':>10}"
        f"{'CVaR90':>11}{'fill':>10}{'stockout':>10}"
    )

    for name, policy in methods.items():
        m = evaluate_risk_inventory_policy(
            policy,
            config,
            episodes=eval_episodes,
            seed=135_000,
            cvar_alpha=0.90,
        )
        print(
            f"{name:<24}"
            f"{m['mean_cost']:>10.3f}"
            f"{m['p90_cost']:>10.3f}"
            f"{m['p95_cost']:>10.3f}"
            f"{m['cvar90_cost']:>11.3f}"
            f"{m['mean_fill_rate']:>10.4f}"
            f"{m['mean_stockout_period_rate']:>10.4f}"
        )


if __name__ == "__main__":
    main()

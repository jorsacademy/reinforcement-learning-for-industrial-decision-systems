from __future__ import annotations

import argparse

from industrial_rl.safe_workforce import (
    PrimalDualPPOAgent,
    PrimalDualPPOConfig,
    SafeWorkforceConfig,
    SafeWorkforceEnv,
    constrained_expected_workload_policy,
    evaluate_safe_workforce_policy,
)


def no_flex_policy(t, backlog):
    del t, backlog
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    config = SafeWorkforceConfig(
        service_backlog_limit=4,
        expected_violation_budget=4.0,
        soft_backlog_cost=0.20,
    )

    updates = 24 if args.smoke else 180
    rollout_episodes = 10 if args.smoke else 24
    eval_episodes = 250 if args.smoke else 1200

    agent = PrimalDualPPOAgent(
        SafeWorkforceEnv(config),
        PrimalDualPPOConfig(
            updates=updates,
            rollout_episodes=rollout_episodes,
            ppo_epochs=3 if args.smoke else 4,
            minibatch_size=128 if args.smoke else 256,
            dual_learning_rate=0.10,
            seed=404,
        ),
    ).fit()

    learned = agent.policy()
    methods = [
        ("No flex", no_flex_policy, False),
        (
            "Constrained expected",
            constrained_expected_workload_policy(config),
            False,
        ),
        ("Primal-dual PPO", learned, False),
        ("PPO + safety shield", learned, True),
    ]

    print("Service-constrained workforce CMDP benchmark")
    print(
        "Expected violation budget: "
        f"{config.expected_violation_budget:.3f} periods/episode"
    )
    print(
        "Final learned Lagrange multiplier: "
        f"{agent.lagrange_multiplier:.3f}"
    )
    print()
    print(
        f"{'method':<24}{'econ cost':>11}{'p90 cost':>11}"
        f"{'viol periods':>13}{'gap':>9}{'any viol':>10}"
        f"{'mean flex':>11}{'shield':>9}"
    )

    for name, policy, use_shield in methods:
        m = evaluate_safe_workforce_policy(
            policy,
            config,
            episodes=eval_episodes,
            seed=115_000,
            use_shield=use_shield,
        )
        print(
            f"{name:<24}"
            f"{m['mean_economic_cost']:>11.3f}"
            f"{m['p90_economic_cost']:>11.3f}"
            f"{m['mean_violation_periods']:>13.3f}"
            f"{m['constraint_gap']:>9.3f}"
            f"{m['episode_violation_probability']:>10.3f}"
            f"{m['mean_flex_workers']:>11.3f}"
            f"{m['shield_intervention_rate']:>9.3f}"
        )


if __name__ == "__main__":
    main()

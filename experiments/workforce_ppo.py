from __future__ import annotations

import argparse

from industrial_rl.ppo_workforce import (
    PPOConfig,
    PPOWorkforceAgent,
    WorkforceConfig,
    WorkforceEnv,
    evaluate_workforce_policy,
    myopic_expected_cost_policy,
)


def no_flex_policy(t, backlog):
    del t, backlog
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    config = WorkforceConfig()
    env = WorkforceEnv(config)
    updates = 20 if args.smoke else 140
    rollout_episodes = 10 if args.smoke else 24
    eval_episodes = 250 if args.smoke else 1200

    agent = PPOWorkforceAgent(
        env,
        PPOConfig(
            updates=updates,
            rollout_episodes=rollout_episodes,
            ppo_epochs=3 if args.smoke else 4,
            seed=202,
        ),
    ).fit()

    methods = {
        "No flex workers": no_flex_policy,
        "Myopic expected-cost": myopic_expected_cost_policy(config),
        "PPO": agent.policy(),
    }

    print("Dynamic workforce allocation PPO benchmark")
    print()
    print(
        f"{'method':<24}{'mean cost':>12}{'p90 cost':>12}"
        f"{'mean flex':>12}{'final backlog':>15}"
    )
    for name, policy in methods.items():
        m = evaluate_workforce_policy(
            policy, config, episodes=eval_episodes, seed=80_000
        )
        print(
            f"{name:<24}{m['mean_cost']:>12.3f}{m['p90_cost']:>12.3f}"
            f"{m['mean_flex_workers']:>12.3f}{m['mean_final_backlog']:>15.3f}"
        )


if __name__ == "__main__":
    main()

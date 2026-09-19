from __future__ import annotations

import argparse

from industrial_rl.pomdp_maintenance import (
    BeliefDQNAgent,
    BeliefDQNConfig,
    MaintenancePOMDPConfig,
    MaintenancePOMDPEnv,
    belief_threshold_policy,
    discretized_belief_dp,
    discretized_belief_policy,
    evaluate_maintenance_policy,
    reactive_sensor_policy,
)


def always_operate(t, belief, observation):
    del t, belief, observation
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    config = MaintenancePOMDPConfig()

    resolution = 6 if args.smoke else 10
    dqn_episodes = 650 if args.smoke else 3500
    eval_episodes = 300 if args.smoke else 1500

    grid, values, dp_table = discretized_belief_dp(
        config,
        resolution=resolution,
    )
    dqn = BeliefDQNAgent(
        MaintenancePOMDPEnv(config),
        BeliefDQNConfig(
            episodes=dqn_episodes,
            warmup=160 if args.smoke else 500,
            batch_size=64,
            replay_size=8000 if args.smoke else 25_000,
            seed=606,
        ),
    ).fit()

    methods = {
        "Always operate": always_operate,
        "Reactive sensor": reactive_sensor_policy,
        "Belief threshold": belief_threshold_policy(),
        f"Belief-DP grid({resolution})": discretized_belief_policy(
            grid,
            dp_table,
        ),
        "Belief DQN": dqn.policy(),
    }

    print("Partially observable predictive-maintenance benchmark")
    print(
        f"Belief grid points: {len(grid)}; "
        f"approximate DP horizon: {config.horizon}"
    )
    print()
    print(
        f"{'method':<24}{'mean cost':>12}{'p90 cost':>12}"
        f"{'fail rate':>11}{'replace':>10}{'minor':>10}{'entropy':>10}"
    )

    for name, policy in methods.items():
        m = evaluate_maintenance_policy(
            policy,
            config,
            episodes=eval_episodes,
            seed=155_000,
        )
        print(
            f"{name:<24}"
            f"{m['mean_cost']:>12.3f}"
            f"{m['p90_cost']:>12.3f}"
            f"{m['mean_failure_period_rate']:>11.4f}"
            f"{m['mean_replacement_rate']:>10.4f}"
            f"{m['mean_minor_maintenance_rate']:>10.4f}"
            f"{m['mean_belief_entropy']:>10.4f}"
        )

    print()
    print(
        "The belief-DP value table is an approximation on a discretized "
        "belief simplex; it is not an exact POMDP solution."
    )
    print(
        f"Value-table range at t=0: "
        f"[{values[0].min():.3f}, {values[0].max():.3f}]"
    )


if __name__ == "__main__":
    main()

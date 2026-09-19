from __future__ import annotations

import argparse

from industrial_rl.offline_scheduling import (
    ATC,
    EDD,
    SETUP_AWARE,
    SPT,
    DynamicDispatchEnv,
    OfflineSchedulingCQL,
    SchedulingCQLConfig,
    SchedulingConfig,
    contextual_dispatch_policy,
    evaluate_scheduling_policy,
    fit_behavior_cloner,
    fixed_rule_policy,
    generate_scheduling_dataset,
    lookahead_policy,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    config = SchedulingConfig()
    dataset = generate_scheduling_dataset(
        config,
        episodes=220 if args.smoke else 900,
        seed=185_000,
    )

    _, bc_policy = fit_behavior_cloner(
        dataset,
        epochs=8 if args.smoke else 35,
        batch_size=128,
        seed=809,
    )

    cql = OfflineSchedulingCQL(
        SchedulingCQLConfig(
            gradient_steps=700 if args.smoke else 3500,
            batch_size=128,
            conservative_alpha=0.8,
            seed=808,
        )
    ).fit(dataset)

    methods = {
        "EDD": fixed_rule_policy(EDD),
        "SPT": fixed_rule_policy(SPT),
        "Setup-aware": fixed_rule_policy(SETUP_AWARE),
        "ATC": fixed_rule_policy(ATC),
        "Contextual planner": contextual_dispatch_policy,
        "Behavior cloning": bc_policy,
        "Offline CQL selector": cql.policy(),
        "2-step lookahead": lookahead_policy,
    }

    eval_episodes = 180 if args.smoke else 800

    print("Offline dynamic production scheduling benchmark")
    print(f"Logged transitions: {len(dataset)}")
    print()
    print(
        f"{'method':<22}{'objective':>12}{'p90':>11}"
        f"{'w.tardiness':>14}{'setup':>10}{'flow':>10}{'on-time':>10}"
    )

    for name, policy in methods.items():
        m = evaluate_scheduling_policy(
            policy,
            config,
            episodes=eval_episodes,
            seed=195_000,
        )
        print(
            f"{name:<22}"
            f"{m['mean_objective']:>12.3f}"
            f"{m['p90_objective']:>11.3f}"
            f"{m['mean_weighted_tardiness']:>14.3f}"
            f"{m['mean_setup_time']:>10.3f}"
            f"{m['mean_flow_time']:>10.3f}"
            f"{m['mean_on_time_rate']:>10.4f}"
        )

    counts = [0, 0, 0, 0]
    for row in dataset:
        counts[row.action] += 1
    total = len(dataset)
    labels = ["EDD", "SPT", "Setup-aware", "ATC"]
    print()
    print("Historical action mix:")
    for label, count in zip(labels, counts):
        print(f"{label:<12}{count / total:>8.3f}")


if __name__ == "__main__":
    main()

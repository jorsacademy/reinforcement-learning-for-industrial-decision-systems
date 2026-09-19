from __future__ import annotations

import argparse

from industrial_rl.offline_scheduling import (
    ATC,
    EDD,
    FIFO,
    SPT,
    BCConfig,
    BehaviorCloningSelector,
    ConservativeDispatchDQN,
    DispatchFQEConfig,
    NeuralDispatchFQE,
    OfflineDispatchConfig,
    ProbabilitySupportGuard,
    RULE_NAMES,
    SchedulingConfig,
    evaluate_dispatch_policy,
    evaluate_rolling_horizon,
    fixed_rule_policy,
    generate_dispatch_log,
    legacy_rule_selector,
    target_behavior_support,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    config = SchedulingConfig()

    dataset = generate_dispatch_log(
        config,
        episodes=220 if args.smoke else 1000,
        epsilon=0.12,
        seed=180_000,
    )

    bc = BehaviorCloningSelector(
        config=BCConfig(
            gradient_steps=350 if args.smoke else 1600,
            batch_size=128,
            seed=808,
        )
    ).fit(dataset)

    cql = ConservativeDispatchDQN(
        config=OfflineDispatchConfig(
            gradient_steps=700 if args.smoke else 4200,
            batch_size=128,
            conservative_alpha=0.8,
            target_update=150,
            seed=809,
        )
    ).fit(dataset)

    cql_policy = cql.policy()
    guarded = ProbabilitySupportGuard(
        cql_policy,
        bc,
        threshold=0.12,
    )

    policies = [
        ("FIFO", fixed_rule_policy(FIFO), None),
        ("EDD", fixed_rule_policy(EDD), None),
        ("SPT", fixed_rule_policy(SPT), None),
        ("ATC", fixed_rule_policy(ATC), None),
        ("Legacy selector", legacy_rule_selector, None),
        ("Behavior cloning", bc.policy(), None),
        ("Offline CQL-style", cql_policy, None),
        ("CQL + support guard", guarded, guarded),
    ]

    eval_episodes = 120 if args.smoke else 600

    print("Offline dynamic shop-floor dispatching benchmark")
    print(f"Logged transitions: {len(dataset)}")
    print()
    print(
        f"{'method':<24}{'support':>10}{'FQE cost':>12}"
        f"{'sim cost':>12}{'p90':>11}{'tardiness':>12}"
        f"{'setup':>9}{'flow':>9}{'on-time':>10}{'guard':>9}"
    )

    for i, (name, policy, guard) in enumerate(policies):
        support = target_behavior_support(
            dataset,
            policy,
            bc,
        )

        fqe = NeuralDispatchFQE(
            config=DispatchFQEConfig(
                gradient_steps=450 if args.smoke else 2200,
                batch_size=128,
                seed=900 + i,
            )
        ).fit(
            dataset,
            policy,
        )
        fqe_cost = fqe.initial_cost(
            dataset,
            policy,
        )

        if guard is not None:
            guard.reset_stats()

        metrics = evaluate_dispatch_policy(
            policy,
            config,
            episodes=eval_episodes,
            seed=190_000,
        )
        guard_rate = (
            guard.intervention_rate
            if guard is not None
            else 0.0
        )

        print(
            f"{name:<24}"
            f"{support:>10.3f}"
            f"{fqe_cost:>12.3f}"
            f"{metrics['mean_cost']:>12.3f}"
            f"{metrics['p90_cost']:>11.3f}"
            f"{metrics['mean_weighted_tardiness']:>12.3f}"
            f"{metrics['mean_setup_time']:>9.3f}"
            f"{metrics['mean_flow_time']:>9.3f}"
            f"{metrics['mean_on_time_rate']:>10.4f}"
            f"{guard_rate:>9.3f}"
        )

    rh = evaluate_rolling_horizon(
        config,
        episodes=60 if args.smoke else 250,
        seed=190_000,
        depth=4,
        candidate_cap=7,
    )

    print()
    print("Rolling-horizon local permutation search")
    print(
        f"mean_cost={rh['mean_cost']:.3f}, "
        f"p90={rh['p90_cost']:.3f}, "
        f"weighted_tardiness={rh['mean_weighted_tardiness']:.3f}, "
        f"setup={rh['mean_setup_time']:.3f}, "
        f"flow={rh['mean_flow_time']:.3f}, "
        f"on_time={rh['mean_on_time_rate']:.4f}"
    )
    print(
        "The rolling-horizon reference is model-based and not fitted from logs; "
        "FQE is therefore not reported for it."
    )
    print("Dispatch rules:", ", ".join(RULE_NAMES))


if __name__ == "__main__":
    main()

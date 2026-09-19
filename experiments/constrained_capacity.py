from __future__ import annotations

from industrial_rl import (
    CapacityConfig,
    CapacityEnv,
    LagrangianCapacityQLearner,
    LagrangianQLearningConfig,
    capacity_table_policy,
    evaluate_capacity_policy,
    exact_capacity_dp,
)


def myopic_policy(config: CapacityConfig):
    def choose(t: int, net_inventory: int, overtime_used: int) -> int:
        del t
        target = 4 - net_inventory
        remaining_ot = config.overtime_budget - overtime_used
        upper = min(config.max_production, config.regular_capacity + remaining_ot)
        return max(0, min(int(target), upper))

    return choose


def main():
    config = CapacityConfig()
    values, dp_table = exact_capacity_dp(config)
    dp = capacity_table_policy(dp_table, config)

    learner = LagrangianCapacityQLearner(
        CapacityEnv(config),
        LagrangianQLearningConfig(episodes=45_000, lagrange_multiplier=0.8, seed=123),
    ).fit()

    methods = {
        "Exact budget-DP": dp,
        "Myopic production": myopic_policy(config),
        "Lagrangian Q-learning": learner.policy(),
    }

    env = CapacityEnv(config)
    init_idx = env._inv_index(config.initial_net_inventory)
    print("Constrained capacity-allocation benchmark")
    print(f"Exact expected cost from initial state: {values[0, init_idx, 0]:.3f}")
    print()
    print(
        f"{'method':<24}{'mean cost':>12}{'p90 cost':>12}"
        f"{'overtime':>12}{'violations':>12}{'final backlog':>15}"
    )
    for name, policy in methods.items():
        metrics = evaluate_capacity_policy(policy, config, episodes=2500, seed=30_000)
        print(
            f"{name:<24}{metrics['mean_cost']:>12.3f}{metrics['p90_cost']:>12.3f}"
            f"{metrics['mean_overtime']:>12.3f}{metrics['budget_violation_rate']:>12.4f}"
            f"{metrics['mean_final_backlog']:>15.3f}"
        )


if __name__ == "__main__":
    main()

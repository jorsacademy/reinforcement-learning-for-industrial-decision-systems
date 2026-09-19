from __future__ import annotations

import argparse

from industrial_rl.sac_energy import (
    EnergyProductionConfig,
    EnergyProductionEnv,
    SACConfig,
    SACProductionAgent,
    evaluate_energy_policy,
    mpc_grid_policy,
)


def constant_rate_policy(config: EnergyProductionConfig):
    normalized = 2.0 * config.base_demand / config.max_rate - 1.0

    def choose(obs):
        del obs
        return float(normalized)

    return choose


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    config = EnergyProductionConfig()

    total_steps = 4500 if args.smoke else 30_000
    warmup = 400 if args.smoke else 1_000
    eval_episodes = 220 if args.smoke else 1000

    agent = SACProductionAgent(
        EnergyProductionEnv(config),
        SACConfig(
            total_steps=total_steps,
            warmup_steps=warmup,
            batch_size=64 if args.smoke else 128,
            replay_size=12_000 if args.smoke else 50_000,
            seed=303,
        ),
    ).fit()

    methods = {
        "Constant-rate": constant_rate_policy(config),
        "3-step MPC grid": mpc_grid_policy(config, lookahead=3, grid_points=9),
        "SAC": agent.policy(),
    }

    print("Energy-aware continuous production SAC benchmark")
    print()
    print(
        f"{'method':<20}{'mean cost':>12}{'p90 cost':>12}"
        f"{'energy use':>12}{'final backlog':>15}{'total ramp':>12}"
    )
    for name, policy in methods.items():
        metrics = evaluate_energy_policy(
            policy,
            config,
            episodes=eval_episodes,
            seed=95_000,
        )
        print(
            f"{name:<20}{metrics['mean_cost']:>12.3f}{metrics['p90_cost']:>12.3f}"
            f"{metrics['mean_energy_use']:>12.3f}"
            f"{metrics['mean_final_backlog']:>15.3f}"
            f"{metrics['mean_total_ramp']:>12.3f}"
        )


if __name__ == "__main__":
    main()

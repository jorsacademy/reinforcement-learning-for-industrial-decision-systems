from __future__ import annotations

import argparse

from industrial_rl.sac_energy import (
    EnergyProductionConfig,
    EnergyProductionEnv,
    SACConfig,
    SACEnergyAgent,
    evaluate_energy_policy,
    myopic_energy_policy,
    rolling_horizon_mpc_policy,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = EnergyProductionConfig()

    if args.smoke:
        sac_config = SACConfig(
            episodes=250,
            replay_size=10_000,
            batch_size=64,
            warmup_steps=180,
            updates_per_step=1,
            seed=303,
        )
        eval_episodes = 250
    else:
        sac_config = SACConfig()
        eval_episodes = 1000

    agent = SACEnergyAgent(EnergyProductionEnv(config), sac_config).fit()

    methods = {
        "Myopic economic dispatch": myopic_energy_policy(config),
        "Rolling-horizon MPC": rolling_horizon_mpc_policy(
            config,
            lookahead=2,
            grid_points=7,
        ),
        "SAC": agent.policy(),
    }

    print("Energy-aware continuous production SAC benchmark")
    print()
    print(
        f"{'method':<27}{'mean cost':>12}{'p90 cost':>12}"
        f"{'energy cost':>14}{'final backlog':>15}{'production':>12}"
    )
    for name, policy in methods.items():
        metrics = evaluate_energy_policy(
            policy,
            config,
            episodes=eval_episodes,
            seed=90_000,
        )
        print(
            f"{name:<27}{metrics['mean_cost']:>12.3f}"
            f"{metrics['p90_cost']:>12.3f}"
            f"{metrics['mean_energy_cost']:>14.3f}"
            f"{metrics['mean_final_backlog']:>15.3f}"
            f"{metrics['mean_total_production']:>12.3f}"
        )


if __name__ == "__main__":
    main()

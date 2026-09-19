import argparse

from industrial_rl.model_based_control import CEMConfig, DynamicsTrainingConfig, evaluate_model_based_controller, train_dynamics_ensemble
from industrial_rl.sac_energy import EnergyProductionConfig, evaluate_energy_policy, mpc_grid_policy


def main(smoke: bool = False) -> None:
    env_config = EnergyProductionConfig()
    training = (
        DynamicsTrainingConfig(transitions=120, ensemble_size=2, hidden_dim=16, epochs=2, batch_size=32)
        if smoke
        else DynamicsTrainingConfig()
    )
    cem = CEMConfig(horizon=2, candidates=8, iterations=1, elite_fraction=0.25) if smoke else CEMConfig()
    episodes = 2 if smoke else 100
    learned = train_dynamics_ensemble(env_config, training)
    mb = evaluate_model_based_controller(learned, env_config, cem, episodes=episodes)
    oracle_mpc = evaluate_energy_policy(
        mpc_grid_policy(env_config, lookahead=2 if smoke else 3, grid_points=5 if smoke else 9),
        env_config,
        episodes=episodes,
        seed=95_000,
    )
    print("learned-model CEM-MPC:", mb)
    print("known-model grid MPC:  ", oracle_mpc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(smoke=args.smoke)

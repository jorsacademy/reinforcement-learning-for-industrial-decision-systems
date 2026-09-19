from industrial_rl.model_based_control import CEMConfig, DynamicsTrainingConfig, evaluate_model_based_controller, train_dynamics_ensemble
from industrial_rl.sac_energy import EnergyProductionConfig, evaluate_energy_policy, mpc_grid_policy


def main() -> None:
    env_config = EnergyProductionConfig()
    learned = train_dynamics_ensemble(env_config, DynamicsTrainingConfig())
    mb = evaluate_model_based_controller(learned, env_config, CEMConfig(), episodes=100)
    oracle_mpc = evaluate_energy_policy(mpc_grid_policy(env_config), env_config, episodes=100, seed=95_000)
    print("learned-model CEM-MPC:", mb)
    print("known-model grid MPC:  ", oracle_mpc)


if __name__ == "__main__":
    main()

import argparse

from industrial_rl.latent_state_control import LatentControlConfig, evaluate_latent_control, train_latent_control


def main(smoke: bool = False) -> None:
    config = (
        LatentControlConfig(
            high_dim=16,
            latent_dim=4,
            samples=40,
            autoencoder_epochs=2,
            policy_epochs=2,
            action_levels=5,
        )
        if smoke
        else LatentControlConfig()
    )
    models = train_latent_control(config=config)
    results = evaluate_latent_control(models, episodes=2 if smoke else 150)
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(smoke=args.smoke)

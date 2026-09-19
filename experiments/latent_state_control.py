from industrial_rl.latent_state_control import LatentControlConfig, evaluate_latent_control, train_latent_control


def main() -> None:
    models = train_latent_control(config=LatentControlConfig())
    results = evaluate_latent_control(models)
    print(results)


if __name__ == "__main__":
    main()

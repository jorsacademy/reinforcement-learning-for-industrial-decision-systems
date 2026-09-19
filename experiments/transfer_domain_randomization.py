import argparse

from industrial_rl.transfer_control import TransferDQNConfig, transfer_benchmark


def main(smoke: bool = False) -> None:
    config = (
        TransferDQNConfig(
            episodes=4,
            action_levels=5,
            batch_size=8,
            replay_size=200,
            warmup=8,
            target_update=5,
        )
        if smoke
        else TransferDQNConfig()
    )
    results = transfer_benchmark(
        config,
        fine_tune_episodes=2 if smoke else 300,
        evaluation_episodes=2 if smoke else 200,
    )
    for name, metrics in results.items():
        print(name)
        for key, value in metrics.items():
            print(f"  {key:24s}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(smoke=args.smoke)

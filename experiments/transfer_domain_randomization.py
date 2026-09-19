from industrial_rl.transfer_control import TransferDQNConfig, transfer_benchmark


def main() -> None:
    results = transfer_benchmark(TransferDQNConfig())
    for name, metrics in results.items():
        print(name)
        for key, value in metrics.items():
            print(f"  {key:24s}: {value:.4f}")


if __name__ == "__main__":
    main()

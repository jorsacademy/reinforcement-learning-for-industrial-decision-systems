from .tuning import run_tune_search


def main() -> None:
    result = run_tune_search(num_samples=6)
    print(f"reorder_point: {result.reorder_point}")
    print(f"order_quantity: {result.order_quantity}")
    print(f"validation_cost: {result.cost:.2f}")
    print(f"trials: {result.trials}")


if __name__ == "__main__":
    main()

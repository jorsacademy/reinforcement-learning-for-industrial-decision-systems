# Transfer RL and Domain Randomization

The transfer benchmark separates four questions that are often conflated:

1. **source policy** — train on Factory A;
2. **zero-shot transfer** — deploy unchanged on shifted Factory B;
3. **fine-tuning** — continue learning on Factory B with a limited adaptation budget;
4. **domain randomization** — train across a family of plausible factories and deploy zero-shot to Factory B.

The target factory changes demand level, seasonality/noise and energy-price dynamics while preserving state/action semantics.

A defensible result should report both target-factory cost and adaptation sample budget. Fine-tuning should also be compared with training from scratch using the same target-data budget. Otherwise a transfer claim can simply hide extra training data.

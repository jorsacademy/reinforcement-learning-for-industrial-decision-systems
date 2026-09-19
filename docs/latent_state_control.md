# Autoencoder Latent-State Control

This experiment uses autoencoders for representation learning rather than anomaly detection.

A compact five-dimensional process state is mapped into a redundant 64-dimensional synthetic sensor representation. An autoencoder compresses the sensor vector into an eight-dimensional latent state. Two policies are then distilled from the same MPC teacher:

- a policy operating on all raw sensor features;
- a policy operating on the learned latent state.

The comparison reports reconstruction error, held-out action-imitation accuracy and true closed-loop operating cost.

The engineering question is whether a compact learned state retains enough decision-relevant information for control. A low reconstruction error alone is not sufficient; the latent representation must preserve downstream decision quality.

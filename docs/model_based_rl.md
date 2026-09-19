# Model-Based RL / Learned-Dynamics Control

This benchmark adds a missing control paradigm between known-model MPC and model-free RL.

The system first collects transitions from the stochastic production environment and fits a bootstrap ensemble

[
(s_t,a_t) mapsto (hat{s}_{t+1}, hat{r}_t).
]

At decision time, Cross-Entropy Method (CEM) searches over short action sequences using the learned dynamics. Ensemble disagreement is subtracted from predicted return, so trajectories in poorly learned regions are penalized.

The important comparison is not "world model versus nothing." Evaluate the learned-model controller against:

- the existing deterministic known-model MPC baseline;
- SAC/PPO where appropriate;
- simple operational policies.

Report operational cost, energy use, backlog, model prediction error and OOD degradation. A learned model can have low one-step error yet still produce poor multi-step decisions; decision quality is therefore the primary metric.

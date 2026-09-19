# Neural Offline RL for Industrial Inventory Decisions

This benchmark asks a deployment-relevant Industrial Engineering question:

> Can a replenishment policy be improved from historical operational logs without online exploration?

Online exploration can be expensive or unacceptable in inventory, production, workforce, and maintenance systems. The Phase 4 benchmark therefore freezes a logged dataset before any offline learner is fitted.

## Logged operations dataset

The environment is the observed-regime inventory model already used by the DQN benchmark.

The log contains:

```text
state observation
action taken
reward / cost
next state observation
terminal flag
period
inventory
demand regime
```

The behavior policy is a regime-aware base-stock rule with limited random exploration. Random exploration is used only while generating the synthetic historical log.

Once the dataset is created, Behavior Cloning, CQL, IQL, coverage diagnostics, and fitted-Q evaluation use the fixed table only.

## Dataset coverage diagnostics

The benchmark reports:

- transition count;
- fraction of discrete states observed;
- fraction of feasible state-action pairs observed;
- mean empirical action entropy at visited states;
- unsupported-action rate for every evaluated policy.

The unsupported-action rate measures how often a policy chooses an action that was never observed at the same discrete state in the logged dataset.

This is a support diagnostic, not a proof of safe deployment.

## Neural Behavior Cloning

Behavior Cloning fits a classifier to reproduce logged actions. It does not optimize downstream cost directly and therefore provides an important lower-complexity offline baseline.

## Discrete CQL

The discrete Conservative Q-Learning implementation minimizes a Bellman TD loss plus a conservative penalty:

```text
logsumexp_a Q(s,a) - Q(s,a_data)
```

Infeasible inventory actions are excluded from the log-sum-exp and Bellman maximization.

The conservative term discourages optimistic values for actions that are weakly supported by the historical data.

## Discrete IQL

The IQL implementation uses:

- expectile regression for V(s);
- Q regression toward r + gamma V(s_next);
- advantage-weighted behavior cloning for the actor.

This permits policy improvement without explicit maximization over out-of-dataset actions during the Q target.

## Fitted Q Evaluation

Before simulator testing, every policy receives a tabular fitted-Q-evaluation estimate using only the fixed dataset.

The FQE estimate must be interpreted together with action support. A policy with a high unsupported-action rate can have a misleading FQE value because the log does not identify its counterfactual consequences.

## Model-advantaged reference

Exact dynamic programming is included only as a model-advantaged reference. It knows the synthetic transition model and is not an offline learner.

## Final evaluation

After BC/CQL/IQL are fully trained and frozen, the simulator is used on a disjoint held-out seed range to report:

- mean cost;
- p90 cost;
- fill rate;
- stockout-period rate.

This separation mirrors the intended workflow:

```text
historical operational logs
        ↓
offline learning / OPE
        ↓
policy freeze
        ↓
digital-twin or simulator validation
        ↓
only then consider operational deployment
```

## Scope

This is a controlled synthetic benchmark. Real offline RL additionally requires logging-policy analysis, confounding assessment, data-quality checks, policy-change governance, and conservative rollout procedures.

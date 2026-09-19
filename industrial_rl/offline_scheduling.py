from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .neural_common import set_global_seed


FIFO = 0
EDD = 1
SPT = 2
ATC = 3
RULE_NAMES = ("FIFO", "EDD", "SPT", "ATC")


@dataclass(frozen=True)
class Job:
    job_id: int
    arrival: float
    processing: float
    due: float
    weight: float
    family: int


@dataclass(frozen=True)
class SchedulingConfig:
    n_jobs: int = 32
    n_families: int = 4
    mean_interarrival: float = 1.6
    processing_low: int = 2
    processing_high: int = 9
    due_allowance_low: float = 8.0
    due_allowance_high: float = 22.0
    setup_time: float = 2.0
    tardiness_weight: float = 1.0
    setup_cost_weight: float = 0.8
    flow_time_weight: float = 0.08
    atc_kappa: float = 2.0

    def validate(self) -> None:
        if self.n_jobs < 2:
            raise ValueError("n_jobs must be at least two")
        if self.n_families < 1:
            raise ValueError("n_families must be positive")
        if self.mean_interarrival <= 0:
            raise ValueError("mean_interarrival must be positive")
        if self.processing_low < 1 or self.processing_high < self.processing_low:
            raise ValueError("invalid processing-time bounds")
        if self.due_allowance_low <= 0 or self.due_allowance_high < self.due_allowance_low:
            raise ValueError("invalid due-date allowance bounds")
        if self.setup_time < 0:
            raise ValueError("setup_time must be non-negative")


def generate_job_stream(
    config: SchedulingConfig = SchedulingConfig(),
    *,
    seed: int,
) -> tuple[Job, ...]:
    config.validate()
    rng = np.random.default_rng(seed)

    arrivals = np.cumsum(
        rng.exponential(
            config.mean_interarrival,
            size=config.n_jobs,
        )
    )
    processing = rng.integers(
        config.processing_low,
        config.processing_high + 1,
        size=config.n_jobs,
    )
    allowance = rng.uniform(
        config.due_allowance_low,
        config.due_allowance_high,
        size=config.n_jobs,
    )
    weights = rng.choice(
        np.asarray([1.0, 2.0, 3.0]),
        size=config.n_jobs,
        p=np.asarray([0.55, 0.30, 0.15]),
    )
    families = rng.integers(
        0,
        config.n_families,
        size=config.n_jobs,
    )

    jobs = []
    for j in range(config.n_jobs):
        due = float(
            arrivals[j]
            + allowance[j]
            + 0.8 * processing[j]
        )
        jobs.append(
            Job(
                job_id=j,
                arrival=float(arrivals[j]),
                processing=float(processing[j]),
                due=due,
                weight=float(weights[j]),
                family=int(families[j]),
            )
        )
    return tuple(jobs)


class DynamicDispatchEnv:
    """Event-driven single-machine scheduling with dynamic arrivals and setups.

    Actions select a dispatching rule, not a job identifier. The selected rule
    chooses one job from the currently released queue.
    """

    def __init__(
        self,
        config: SchedulingConfig = SchedulingConfig(),
    ):
        config.validate()
        self.config = config
        self.jobs: tuple[Job, ...] = ()
        self.current_time = 0.0
        self.current_family = -1
        self.completed: set[int] = set()
        self.queue: list[int] = []
        self.last_metrics: dict[str, float] = {}

    @property
    def n_actions(self) -> int:
        return len(RULE_NAMES)

    @property
    def state_dim(self) -> int:
        return 10

    def reset(
        self,
        *,
        seed: int,
        jobs: tuple[Job, ...] | None = None,
    ) -> np.ndarray:
        self.jobs = (
            generate_job_stream(
                self.config,
                seed=seed,
            )
            if jobs is None
            else tuple(jobs)
        )
        self.current_time = 0.0
        self.current_family = -1
        self.completed = set()
        self.queue = []
        self.last_metrics = {
            "weighted_tardiness": 0.0,
            "setup_time": 0.0,
            "flow_time": 0.0,
        }
        self._release_or_advance()
        return self.observation()

    def _release(self) -> None:
        queued = set(self.queue)
        for job in self.jobs:
            if (
                job.job_id not in self.completed
                and job.job_id not in queued
                and job.arrival <= self.current_time + 1e-12
            ):
                self.queue.append(job.job_id)

    def _release_or_advance(self) -> None:
        self._release()
        if self.queue or len(self.completed) == len(self.jobs):
            return
        future = [
            job.arrival
            for job in self.jobs
            if job.job_id not in self.completed
        ]
        if future:
            self.current_time = max(
                self.current_time,
                float(min(future)),
            )
            self._release()

    def _queue_jobs(self) -> list[Job]:
        return [self.jobs[i] for i in self.queue]

    def _setup(self, job: Job, family: int | None = None) -> float:
        previous = self.current_family if family is None else int(family)
        if previous < 0 or previous == job.family:
            return 0.0
        return float(self.config.setup_time)

    def _slack(self, job: Job) -> float:
        return (
            job.due
            - self.current_time
            - self._setup(job)
            - job.processing
        )

    def observation(self) -> np.ndarray:
        queued = self._queue_jobs()
        if not queued:
            return np.zeros(self.state_dim, dtype=np.float32)

        p = np.asarray(
            [job.processing for job in queued],
            dtype=float,
        )
        slack = np.asarray(
            [self._slack(job) for job in queued],
            dtype=float,
        )
        weights = np.asarray(
            [job.weight for job in queued],
            dtype=float,
        )
        same_family = np.asarray(
            [
                float(
                    self.current_family < 0
                    or job.family == self.current_family
                )
                for job in queued
            ],
            dtype=float,
        )

        scale_time = max(
            self.config.n_jobs
            * self.config.mean_interarrival,
            1.0,
        )
        scale_work = max(
            self.config.n_jobs
            * self.config.processing_high,
            1.0,
        )

        return np.asarray(
            [
                min(len(queued) / self.config.n_jobs, 1.0),
                min(float(p.sum()) / scale_work, 1.0),
                np.clip(float(p.mean()) / self.config.processing_high, 0.0, 2.0),
                np.clip(float(p.min()) / self.config.processing_high, 0.0, 2.0),
                np.clip(float(slack.mean()) / 30.0, -2.0, 2.0),
                np.clip(float(slack.min()) / 30.0, -2.0, 2.0),
                float(np.mean(slack < 0.0)),
                float(np.mean(same_family)),
                np.clip(float(weights.mean()) / 3.0, 0.0, 1.0),
                np.clip(self.current_time / scale_time, 0.0, 3.0),
            ],
            dtype=np.float32,
        )

    def _select_job(self, rule: int) -> int:
        if not self.queue:
            raise RuntimeError("no released jobs available")
        queued = self._queue_jobs()

        if rule == FIFO:
            chosen = min(
                queued,
                key=lambda j: (
                    j.arrival,
                    j.job_id,
                ),
            )
        elif rule == EDD:
            chosen = min(
                queued,
                key=lambda j: (
                    j.due,
                    j.arrival,
                    j.job_id,
                ),
            )
        elif rule == SPT:
            chosen = min(
                queued,
                key=lambda j: (
                    j.processing,
                    j.due,
                    j.job_id,
                ),
            )
        elif rule == ATC:
            avg_p = max(
                float(
                    np.mean(
                        [j.processing for j in queued]
                    )
                ),
                1e-6,
            )

            def atc_index(job: Job) -> float:
                slack_term = max(
                    job.due
                    - self.current_time
                    - self._setup(job)
                    - job.processing,
                    0.0,
                )
                return (
                    job.weight
                    / max(job.processing, 1e-6)
                    * np.exp(
                        -slack_term
                        / (
                            self.config.atc_kappa
                            * avg_p
                        )
                    )
                )

            chosen = max(
                queued,
                key=lambda j: (
                    atc_index(j),
                    -j.due,
                    -j.job_id,
                ),
            )
        else:
            raise ValueError("invalid dispatch rule")

        return int(chosen.job_id)

    def _dispatch_job(self, job_id: int):
        job = self.jobs[int(job_id)]
        setup = self._setup(job)
        start = self.current_time + setup
        completion = start + job.processing
        tardiness = max(
            completion - job.due,
            0.0,
        )
        flow = completion - job.arrival

        incremental_cost = (
            self.config.tardiness_weight
            * job.weight
            * tardiness
            + self.config.setup_cost_weight
            * setup
            + self.config.flow_time_weight
            * flow
        )

        self.last_metrics["weighted_tardiness"] += (
            job.weight * tardiness
        )
        self.last_metrics["setup_time"] += setup
        self.last_metrics["flow_time"] += flow

        self.current_time = completion
        self.current_family = job.family
        self.queue.remove(job_id)
        self.completed.add(job_id)
        self._release_or_advance()

        info = {
            "job_id": int(job_id),
            "incremental_cost": float(incremental_cost),
            "weighted_tardiness": float(job.weight * tardiness),
            "setup_time": float(setup),
            "flow_time": float(flow),
            "completion_time": float(completion),
            "on_time": bool(tardiness <= 1e-12),
        }
        return -float(incremental_cost), info

    def step(self, rule: int):
        if len(self.completed) >= len(self.jobs):
            raise RuntimeError("episode finished")
        self._release_or_advance()
        job_id = self._select_job(int(rule))
        reward, info = self._dispatch_job(job_id)
        done = len(self.completed) >= len(self.jobs)
        return self.observation(), reward, done, info


def fixed_rule_policy(rule: int) -> Callable[[np.ndarray], int]:
    if not 0 <= int(rule) < len(RULE_NAMES):
        raise ValueError("invalid rule")

    def choose(state: np.ndarray) -> int:
        del state
        return int(rule)

    return choose


def legacy_rule_selector(state: np.ndarray) -> int:
    queue_frac = float(state[0])
    mean_slack = float(state[4])
    min_slack = float(state[5])
    overdue_ratio = float(state[6])
    same_family_ratio = float(state[7])

    if overdue_ratio >= 0.20 or min_slack < -0.10:
        return EDD
    if queue_frac >= 0.18:
        return SPT
    if same_family_ratio <= 0.25 and mean_slack > 0.10:
        return FIFO
    return ATC


@dataclass(frozen=True)
class DispatchTransition:
    state: tuple[float, ...]
    action: int
    reward: float
    next_state: tuple[float, ...]
    done: bool
    initial: bool


def generate_dispatch_log(
    config: SchedulingConfig = SchedulingConfig(),
    *,
    episodes: int = 500,
    epsilon: float = 0.12,
    seed: int = 180_000,
) -> tuple[DispatchTransition, ...]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0,1]")

    rng = np.random.default_rng(seed)
    rows: list[DispatchTransition] = []

    for episode in range(episodes):
        env = DynamicDispatchEnv(config)
        state = env.reset(
            seed=seed + episode
        )
        done = False
        first = True

        while not done:
            if rng.random() < epsilon:
                action = int(
                    rng.integers(
                        0,
                        env.n_actions,
                    )
                )
            else:
                action = int(
                    legacy_rule_selector(state)
                )

            next_state, reward, done, _ = env.step(action)
            rows.append(
                DispatchTransition(
                    state=tuple(float(x) for x in state),
                    action=action,
                    reward=float(reward),
                    next_state=tuple(float(x) for x in next_state),
                    done=bool(done),
                    initial=first,
                )
            )
            first = False
            state = next_state

    return tuple(rows)


def evaluate_dispatch_policy(
    policy: Callable[[np.ndarray], int],
    config: SchedulingConfig = SchedulingConfig(),
    *,
    episodes: int = 250,
    seed: int = 190_000,
) -> dict[str, float]:
    costs = []
    tardiness = []
    setup = []
    flow = []
    on_time = []

    for episode in range(episodes):
        env = DynamicDispatchEnv(config)
        state = env.reset(
            seed=seed + episode
        )
        done = False
        episode_on_time = 0

        while not done:
            action = int(policy(state.copy()))
            state, reward, done, info = env.step(action)
            episode_on_time += int(info["on_time"])

        costs.append(
            config.tardiness_weight
            * env.last_metrics["weighted_tardiness"]
            + config.setup_cost_weight
            * env.last_metrics["setup_time"]
            + config.flow_time_weight
            * env.last_metrics["flow_time"]
        )
        tardiness.append(
            env.last_metrics["weighted_tardiness"]
        )
        setup.append(
            env.last_metrics["setup_time"]
        )
        flow.append(
            env.last_metrics["flow_time"]
            / config.n_jobs
        )
        on_time.append(
            episode_on_time
            / config.n_jobs
        )

    a = np.asarray(costs, dtype=float)
    return {
        "mean_cost": float(a.mean()),
        "p90_cost": float(np.quantile(a, 0.90)),
        "mean_weighted_tardiness": float(np.mean(tardiness)),
        "mean_setup_time": float(np.mean(setup)),
        "mean_flow_time": float(np.mean(flow)),
        "mean_on_time_rate": float(np.mean(on_time)),
    }


def rolling_horizon_sequence_policy(
    config: SchedulingConfig = SchedulingConfig(),
    *,
    depth: int = 4,
    candidate_cap: int = 7,
):
    """Model-based local search over currently released jobs.

    Future arrivals are not known. The policy enumerates short permutations of
    current candidates and dispatches the first job from the least-cost local
    sequence. It is a rolling-horizon reference, not a global optimum.
    """

    if depth < 1 or candidate_cap < 1:
        raise ValueError("depth and candidate_cap must be positive")

    def choose_from_env(env: DynamicDispatchEnv) -> int:
        queued = env._queue_jobs()
        if not queued:
            raise RuntimeError("empty queue")

        candidates = sorted(
            queued,
            key=lambda j: (
                j.due,
                j.processing,
                j.job_id,
            ),
        )[:candidate_cap]

        r = min(depth, len(candidates))
        best_cost = np.inf
        best_job = candidates[0].job_id

        for seq in permutations(
            candidates,
            r=r,
        ):
            time = env.current_time
            family = env.current_family
            cost = 0.0

            for job in seq:
                setup = (
                    0.0
                    if family < 0 or family == job.family
                    else config.setup_time
                )
                completion = (
                    time
                    + setup
                    + job.processing
                )
                tardiness = max(
                    completion - job.due,
                    0.0,
                )
                flow = completion - job.arrival
                cost += (
                    config.tardiness_weight
                    * job.weight
                    * tardiness
                    + config.setup_cost_weight
                    * setup
                    + config.flow_time_weight
                    * flow
                )
                time = completion
                family = job.family

            if cost < best_cost - 1e-12:
                best_cost = cost
                best_job = seq[0].job_id

        # Return the dispatch rule that would select the chosen first job if one
        # exists. If several rules select it, use the first stable rule. If no
        # rule selects it, use ATC as a fixed-action-space approximation.
        for rule in range(len(RULE_NAMES)):
            if env._select_job(rule) == best_job:
                return rule
        return ATC

    return choose_from_env


def evaluate_rolling_horizon(
    config: SchedulingConfig = SchedulingConfig(),
    *,
    episodes: int = 100,
    seed: int = 190_000,
    depth: int = 4,
    candidate_cap: int = 7,
) -> dict[str, float]:
    selector = rolling_horizon_sequence_policy(
        config,
        depth=depth,
        candidate_cap=candidate_cap,
    )

    costs = []
    tardiness = []
    setup = []
    flow = []
    on_time = []

    for episode in range(episodes):
        env = DynamicDispatchEnv(config)
        env.reset(seed=seed + episode)
        done = False
        episode_on_time = 0

        while not done:
            action = int(selector(env))
            _, _, done, info = env.step(action)
            episode_on_time += int(info["on_time"])

        total_cost = (
            config.tardiness_weight
            * env.last_metrics["weighted_tardiness"]
            + config.setup_cost_weight
            * env.last_metrics["setup_time"]
            + config.flow_time_weight
            * env.last_metrics["flow_time"]
        )
        costs.append(total_cost)
        tardiness.append(env.last_metrics["weighted_tardiness"])
        setup.append(env.last_metrics["setup_time"])
        flow.append(env.last_metrics["flow_time"] / config.n_jobs)
        on_time.append(episode_on_time / config.n_jobs)

    a = np.asarray(costs, dtype=float)
    return {
        "mean_cost": float(a.mean()),
        "p90_cost": float(np.quantile(a, 0.90)),
        "mean_weighted_tardiness": float(np.mean(tardiness)),
        "mean_setup_time": float(np.mean(setup)),
        "mean_flow_time": float(np.mean(flow)),
        "mean_on_time_rate": float(np.mean(on_time)),
    }


class SelectorNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class BCConfig:
    gradient_steps: int = 1200
    batch_size: int = 128
    learning_rate: float = 1e-3
    seed: int = 808


class BehaviorCloningSelector:
    def __init__(
        self,
        state_dim: int = 10,
        n_actions: int = 4,
        config: BCConfig = BCConfig(),
    ):
        self.config = config
        set_global_seed(config.seed)
        self.model = SelectorNetwork(
            state_dim,
            n_actions,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
        )
        self.rng = np.random.default_rng(config.seed)

    def fit(
        self,
        dataset: tuple[DispatchTransition, ...],
    ):
        if not dataset:
            raise ValueError("dataset must be non-empty")
        states = np.asarray(
            [row.state for row in dataset],
            dtype=np.float32,
        )
        actions = np.asarray(
            [row.action for row in dataset],
            dtype=np.int64,
        )
        n = len(dataset)
        batch = min(self.config.batch_size, n)

        for _ in range(self.config.gradient_steps):
            idx = self.rng.integers(
                0,
                n,
                size=batch,
            )
            s = torch.as_tensor(
                states[idx],
                dtype=torch.float32,
            )
            a = torch.as_tensor(
                actions[idx],
                dtype=torch.long,
            )
            loss = F.cross_entropy(
                self.model(s),
                a,
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return self

    def probabilities(
        self,
        state: np.ndarray,
    ) -> np.ndarray:
        with torch.no_grad():
            logits = self.model(
                torch.as_tensor(
                    state,
                    dtype=torch.float32,
                ).unsqueeze(0)
            )[0]
            probs = torch.softmax(
                logits,
                dim=-1,
            )
        return probs.cpu().numpy()

    def policy(self):
        def choose(state: np.ndarray) -> int:
            return int(
                np.argmax(
                    self.probabilities(state)
                )
            )
        return choose


@dataclass(frozen=True)
class OfflineDispatchConfig:
    gradient_steps: int = 3500
    batch_size: int = 128
    learning_rate: float = 8e-4
    gamma: float = 1.0
    conservative_alpha: float = 0.8
    target_update: int = 200
    seed: int = 809


class ConservativeDispatchDQN:
    def __init__(
        self,
        state_dim: int = 10,
        n_actions: int = 4,
        config: OfflineDispatchConfig = OfflineDispatchConfig(),
    ):
        self.config = config
        self.n_actions = n_actions
        set_global_seed(config.seed)

        self.online = SelectorNetwork(
            state_dim,
            n_actions,
        )
        self.target = SelectorNetwork(
            state_dim,
            n_actions,
        )
        self.target.load_state_dict(
            self.online.state_dict()
        )
        self.optimizer = torch.optim.Adam(
            self.online.parameters(),
            lr=config.learning_rate,
        )
        self.rng = np.random.default_rng(config.seed)
        self.reward_scale = 1.0

    def fit(
        self,
        dataset: tuple[DispatchTransition, ...],
    ):
        if not dataset:
            raise ValueError("dataset must be non-empty")

        states = np.asarray(
            [row.state for row in dataset],
            dtype=np.float32,
        )
        actions = np.asarray(
            [row.action for row in dataset],
            dtype=np.int64,
        )
        rewards = np.asarray(
            [row.reward for row in dataset],
            dtype=np.float32,
        )
        next_states = np.asarray(
            [row.next_state for row in dataset],
            dtype=np.float32,
        )
        dones = np.asarray(
            [row.done for row in dataset],
            dtype=np.float32,
        )

        n = len(dataset)
        batch = min(
            self.config.batch_size,
            n,
        )

        for step in range(self.config.gradient_steps):
            idx = self.rng.integers(
                0,
                n,
                size=batch,
            )

            s = torch.as_tensor(
                states[idx],
                dtype=torch.float32,
            )
            a = torch.as_tensor(
                actions[idx],
                dtype=torch.long,
            )
            r = torch.as_tensor(
                rewards[idx],
                dtype=torch.float32,
            )
            ns = torch.as_tensor(
                next_states[idx],
                dtype=torch.float32,
            )
            d = torch.as_tensor(
                dones[idx],
                dtype=torch.float32,
            )

            q_all = self.online(s)
            q_data = q_all.gather(
                1,
                a[:, None],
            ).squeeze(1)

            with torch.no_grad():
                next_value = self.target(
                    ns
                ).max(dim=1).values
                target = (
                    r
                    + (1.0 - d)
                    * self.config.gamma
                    * next_value
                )

            td_loss = F.smooth_l1_loss(
                q_data,
                target,
            )
            conservative_gap = (
                torch.logsumexp(
                    q_all,
                    dim=1,
                )
                - q_data
            ).mean()
            loss = (
                td_loss
                + self.config.conservative_alpha
                * conservative_gap
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.online.parameters(),
                5.0,
            )
            self.optimizer.step()

            if (
                (step + 1)
                % self.config.target_update
                == 0
            ):
                self.target.load_state_dict(
                    self.online.state_dict()
                )

        return self

    def policy(self):
        def choose(state: np.ndarray) -> int:
            with torch.no_grad():
                q = self.online(
                    torch.as_tensor(
                        state,
                        dtype=torch.float32,
                    ).unsqueeze(0)
                )[0]
            return int(
                torch.argmax(q).item()
            )
        return choose


class ProbabilitySupportGuard:
    def __init__(
        self,
        target_policy: Callable[[np.ndarray], int],
        behavior_model: BehaviorCloningSelector,
        *,
        threshold: float = 0.10,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0,1]")
        self.target_policy = target_policy
        self.behavior_model = behavior_model
        self.threshold = threshold
        self.decisions = 0
        self.interventions = 0

    def reset_stats(self) -> None:
        self.decisions = 0
        self.interventions = 0

    def __call__(self, state: np.ndarray) -> int:
        proposed = int(
            self.target_policy(state)
        )
        probs = self.behavior_model.probabilities(
            state
        )
        self.decisions += 1

        if probs[proposed] >= self.threshold:
            return proposed

        self.interventions += 1
        return int(np.argmax(probs))

    @property
    def intervention_rate(self) -> float:
        return float(
            self.interventions
            / max(self.decisions, 1)
        )


def target_behavior_support(
    dataset: tuple[DispatchTransition, ...],
    policy: Callable[[np.ndarray], int],
    behavior_model: BehaviorCloningSelector,
) -> float:
    if not dataset:
        raise ValueError("dataset must be non-empty")
    probs = []
    for row in dataset:
        state = np.asarray(
            row.state,
            dtype=np.float32,
        )
        action = int(policy(state))
        probs.append(
            behavior_model.probabilities(
                state
            )[action]
        )
    return float(np.mean(probs))


@dataclass(frozen=True)
class DispatchFQEConfig:
    gradient_steps: int = 1800
    batch_size: int = 128
    learning_rate: float = 8e-4
    gamma: float = 1.0
    target_update: int = 200
    seed: int = 810


class NeuralDispatchFQE:
    def __init__(
        self,
        state_dim: int = 10,
        n_actions: int = 4,
        config: DispatchFQEConfig = DispatchFQEConfig(),
    ):
        self.config = config
        set_global_seed(config.seed)
        self.online = SelectorNetwork(
            state_dim,
            n_actions,
        )
        self.target = SelectorNetwork(
            state_dim,
            n_actions,
        )
        self.target.load_state_dict(
            self.online.state_dict()
        )
        self.optimizer = torch.optim.Adam(
            self.online.parameters(),
            lr=config.learning_rate,
        )
        self.rng = np.random.default_rng(config.seed)

    def fit(
        self,
        dataset: tuple[DispatchTransition, ...],
        target_policy: Callable[[np.ndarray], int],
    ):
        if not dataset:
            raise ValueError("dataset must be non-empty")

        states = np.asarray(
            [row.state for row in dataset],
            dtype=np.float32,
        )
        actions = np.asarray(
            [row.action for row in dataset],
            dtype=np.int64,
        )
        rewards = np.asarray(
            [row.reward for row in dataset],
            dtype=np.float32,
        )
        self.reward_scale = max(
            float(np.mean(np.abs(rewards))),
            1.0,
        )
        rewards = rewards / self.reward_scale
        next_states = np.asarray(
            [row.next_state for row in dataset],
            dtype=np.float32,
        )
        dones = np.asarray(
            [row.done for row in dataset],
            dtype=np.float32,
        )

        n = len(dataset)
        batch = min(
            self.config.batch_size,
            n,
        )

        for step in range(self.config.gradient_steps):
            idx = self.rng.integers(
                0,
                n,
                size=batch,
            )
            s = torch.as_tensor(
                states[idx],
                dtype=torch.float32,
            )
            a = torch.as_tensor(
                actions[idx],
                dtype=torch.long,
            )
            r = torch.as_tensor(
                rewards[idx],
                dtype=torch.float32,
            )
            ns_np = next_states[idx]
            ns = torch.as_tensor(
                ns_np,
                dtype=torch.float32,
            )
            d = torch.as_tensor(
                dones[idx],
                dtype=torch.float32,
            )

            q = self.online(s).gather(
                1,
                a[:, None],
            ).squeeze(1)

            next_actions = torch.as_tensor(
                [
                    int(
                        target_policy(
                            row
                        )
                    )
                    for row in ns_np
                ],
                dtype=torch.long,
            )

            with torch.no_grad():
                next_q = self.target(
                    ns
                ).gather(
                    1,
                    next_actions[:, None],
                ).squeeze(1)
                target = (
                    r
                    + (1.0 - d)
                    * self.config.gamma
                    * next_q
                )

            loss = F.smooth_l1_loss(
                q,
                target,
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (
                (step + 1)
                % self.config.target_update
                == 0
            ):
                self.target.load_state_dict(
                    self.online.state_dict()
                )
        return self

    def initial_cost(
        self,
        dataset: tuple[DispatchTransition, ...],
        target_policy: Callable[[np.ndarray], int],
    ) -> float:
        initial_states = np.asarray(
            [
                row.state
                for row in dataset
                if row.initial
            ],
            dtype=np.float32,
        )
        if len(initial_states) == 0:
            raise ValueError("dataset has no initial states")

        actions = torch.as_tensor(
            [
                int(
                    target_policy(state)
                )
                for state in initial_states
            ],
            dtype=torch.long,
        )
        with torch.no_grad():
            q = self.online(
                torch.as_tensor(
                    initial_states,
                    dtype=torch.float32,
                )
            ).gather(
                1,
                actions[:, None],
            ).squeeze(1)
        return float(
            -q.mean().item()
            * self.reward_scale
        )

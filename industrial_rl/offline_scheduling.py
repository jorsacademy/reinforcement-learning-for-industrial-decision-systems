from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .neural_common import set_global_seed


EDD = 0
SPT = 1
SETUP_AWARE = 2
ATC = 3


@dataclass(frozen=True)
class SchedulingConfig:
    n_jobs: int = 32
    n_families: int = 3
    mean_interarrival: float = 2.2
    processing_low: int = 2
    processing_high: int = 9
    due_date_factor_low: float = 2.0
    due_date_factor_high: float = 4.2
    setup_time: int = 2
    setup_cost_weight: float = 1.8
    tardiness_weight_low: int = 1
    tardiness_weight_high: int = 4
    atc_k: float = 2.0

    def validate(self) -> None:
        if self.n_jobs < 2 or self.n_families < 2:
            raise ValueError("n_jobs and n_families must be at least two")
        if self.mean_interarrival <= 0:
            raise ValueError("mean_interarrival must be positive")
        if not 1 <= self.processing_low <= self.processing_high:
            raise ValueError("invalid processing-time bounds")
        if self.due_date_factor_low <= 0 or self.due_date_factor_high < self.due_date_factor_low:
            raise ValueError("invalid due-date factors")
        if self.setup_time < 0 or self.setup_cost_weight < 0:
            raise ValueError("setup parameters must be non-negative")
        if not 1 <= self.tardiness_weight_low <= self.tardiness_weight_high:
            raise ValueError("invalid tardiness-weight bounds")


@dataclass(frozen=True)
class Job:
    job_id: int
    release: float
    processing: float
    due: float
    family: int
    weight: float


@dataclass(frozen=True)
class SchedulingTransition:
    state: tuple[float, ...]
    action: int
    reward: float
    next_state: tuple[float, ...]
    done: bool


class DynamicDispatchEnv:
    """Single-machine dynamic dispatching with releases, due dates, and setups."""

    def __init__(self, config: SchedulingConfig = SchedulingConfig()):
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(0)
        self.jobs: list[Job] = []
        self.completed: set[int] = set()
        self.time = 0.0
        self.machine_family = -1
        self.total_setup = 0.0
        self.total_weighted_tardiness = 0.0
        self.total_flow_time = 0.0
        self.on_time = 0

    @property
    def n_actions(self) -> int:
        return 4

    @property
    def state_dim(self) -> int:
        return 9

    def _generate_jobs(self) -> list[Job]:
        release = 0.0
        jobs = []
        for j in range(self.config.n_jobs):
            if j > 0:
                release += float(
                    self.rng.exponential(self.config.mean_interarrival)
                )
            processing = float(
                self.rng.integers(
                    self.config.processing_low,
                    self.config.processing_high + 1,
                )
            )
            due_factor = float(
                self.rng.uniform(
                    self.config.due_date_factor_low,
                    self.config.due_date_factor_high,
                )
            )
            due = release + due_factor * processing + float(
                self.rng.uniform(0.0, 6.0)
            )
            family = int(
                self.rng.integers(0, self.config.n_families)
            )
            weight = float(
                self.rng.integers(
                    self.config.tardiness_weight_low,
                    self.config.tardiness_weight_high + 1,
                )
            )
            jobs.append(
                Job(
                    job_id=j,
                    release=release,
                    processing=processing,
                    due=due,
                    family=family,
                    weight=weight,
                )
            )
        return jobs

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.jobs = self._generate_jobs()
        self.completed = set()
        self.time = 0.0
        self.machine_family = -1
        self.total_setup = 0.0
        self.total_weighted_tardiness = 0.0
        self.total_flow_time = 0.0
        self.on_time = 0
        self._advance_to_next_release_if_needed()
        return self.state_vector()

    def _available_jobs(self) -> list[Job]:
        return [
            job
            for job in self.jobs
            if job.job_id not in self.completed
            and job.release <= self.time + 1e-12
        ]

    def _advance_to_next_release_if_needed(self) -> None:
        if self._available_jobs() or len(self.completed) == len(self.jobs):
            return
        future = [
            job.release
            for job in self.jobs
            if job.job_id not in self.completed
        ]
        if future:
            self.time = float(min(future))

    def _setup(self, job: Job) -> float:
        if self.machine_family < 0 or self.machine_family == job.family:
            return 0.0
        return float(self.config.setup_time)

    def state_vector(self) -> np.ndarray:
        self._advance_to_next_release_if_needed()
        queue = self._available_jobs()
        if not queue:
            return np.zeros(self.state_dim, dtype=np.float32)

        proc = np.asarray([j.processing for j in queue], dtype=float)
        slack = np.asarray(
            [j.due - self.time - j.processing for j in queue],
            dtype=float,
        )
        weights = np.asarray([j.weight for j in queue], dtype=float)
        same_family = np.asarray(
            [
                float(
                    self.machine_family < 0
                    or j.family == self.machine_family
                )
                for j in queue
            ],
            dtype=float,
        )

        max_proc = float(self.config.processing_high)
        max_weight = float(self.config.tardiness_weight_high)
        horizon_proxy = max(
            self.config.n_jobs * self.config.mean_interarrival,
            1.0,
        )

        return np.asarray(
            [
                len(queue) / self.config.n_jobs,
                proc.mean() / max_proc,
                proc.min() / max_proc,
                np.clip(slack.min() / 30.0, -1.0, 1.0),
                np.mean(slack < 0.0),
                same_family.mean(),
                weights.mean() / max_weight,
                np.mean(weights * np.maximum(-slack, 0.0)) / 20.0,
                np.clip(self.time / horizon_proxy, 0.0, 2.0),
            ],
            dtype=np.float32,
        )

    def _select_job(self, action: int) -> Job:
        queue = self._available_jobs()
        if not queue:
            raise RuntimeError("no available jobs at dispatch epoch")
        if action == EDD:
            return min(queue, key=lambda j: (j.due, j.processing, j.job_id))
        if action == SPT:
            return min(queue, key=lambda j: (j.processing, j.due, j.job_id))
        if action == SETUP_AWARE:
            return min(
                queue,
                key=lambda j: (
                    self._setup(j),
                    j.due,
                    j.processing,
                    j.job_id,
                ),
            )
        if action == ATC:
            avg_p = max(
                float(np.mean([j.processing for j in queue])),
                1e-6,
            )
            def score(job: Job) -> float:
                slack = max(
                    job.due - self.time - job.processing,
                    0.0,
                )
                return (
                    job.weight / job.processing
                    * np.exp(
                        -slack
                        / (self.config.atc_k * avg_p)
                    )
                )
            return max(queue, key=lambda j: (score(j), -j.job_id))
        raise ValueError("invalid dispatch-rule action")

    def _candidate_objective_increment(
        self,
        job: Job,
        *,
        time: float | None = None,
        machine_family: int | None = None,
    ) -> tuple[float, float, float]:
        now = self.time if time is None else float(time)
        family = (
            self.machine_family
            if machine_family is None
            else int(machine_family)
        )
        setup = 0.0 if family < 0 or family == job.family else float(self.config.setup_time)
        completion = now + setup + job.processing
        tardiness = max(completion - job.due, 0.0)
        increment = (
            job.weight * tardiness
            + self.config.setup_cost_weight * setup
        )
        return float(increment), float(completion), float(setup)

    def step(self, action: int):
        if len(self.completed) >= len(self.jobs):
            raise RuntimeError("episode finished")
        self._advance_to_next_release_if_needed()
        state_before = self.state_vector()
        job = self._select_job(int(action))

        increment, completion, setup = self._candidate_objective_increment(job)
        tardiness = max(completion - job.due, 0.0)
        flow = completion - job.release

        self.time = completion
        self.machine_family = job.family
        self.completed.add(job.job_id)
        self.total_setup += setup
        self.total_weighted_tardiness += job.weight * tardiness
        self.total_flow_time += flow
        self.on_time += int(tardiness <= 1e-12)

        done = len(self.completed) >= len(self.jobs)
        if not done:
            self._advance_to_next_release_if_needed()
        next_state = (
            np.zeros(self.state_dim, dtype=np.float32)
            if done
            else self.state_vector()
        )
        info = {
            "job_id": job.job_id,
            "incremental_objective": float(increment),
            "completion": float(completion),
            "tardiness": float(tardiness),
            "weighted_tardiness": float(job.weight * tardiness),
            "setup": float(setup),
            "flow_time": float(flow),
            "on_time": bool(tardiness <= 1e-12),
            "state_before": state_before,
        }
        return next_state, -float(increment), done, info

    def two_step_lookahead_action(self) -> int:
        """Current-information two-dispatch lookahead over the four rule actions."""
        self._advance_to_next_release_if_needed()
        queue = self._available_jobs()
        if not queue:
            return ATC

        best_action = ATC
        best_score = np.inf

        for action in range(self.n_actions):
            first = self._select_job(action)
            first_inc, first_completion, _ = self._candidate_objective_increment(first)

            remaining = [j for j in queue if j.job_id != first.job_id]
            if not remaining:
                score = first_inc
            else:
                second_candidates = []
                for second_action in range(self.n_actions):
                    if second_action == EDD:
                        second = min(remaining, key=lambda j: (j.due, j.processing, j.job_id))
                    elif second_action == SPT:
                        second = min(remaining, key=lambda j: (j.processing, j.due, j.job_id))
                    elif second_action == SETUP_AWARE:
                        second = min(
                            remaining,
                            key=lambda j: (
                                0.0 if j.family == first.family else float(self.config.setup_time),
                                j.due,
                                j.processing,
                                j.job_id,
                            ),
                        )
                    else:
                        avg_p = max(float(np.mean([j.processing for j in remaining])), 1e-6)
                        def atc_score(job: Job) -> float:
                            slack = max(job.due - first_completion - job.processing, 0.0)
                            return job.weight / job.processing * np.exp(
                                -slack / (self.config.atc_k * avg_p)
                            )
                        second = max(remaining, key=lambda j: (atc_score(j), -j.job_id))
                    second_inc, _, _ = self._candidate_objective_increment(
                        second,
                        time=first_completion,
                        machine_family=first.family,
                    )
                    second_candidates.append(second_inc)
                score = first_inc + min(second_candidates)

            if score < best_score - 1e-12:
                best_score = float(score)
                best_action = int(action)

        return best_action


def fixed_rule_policy(action: int):
    def choose(env: DynamicDispatchEnv) -> int:
        return int(action)
    return choose


def lookahead_policy(env: DynamicDispatchEnv) -> int:
    return env.two_step_lookahead_action()


def evaluate_scheduling_policy(
    policy: Callable[[DynamicDispatchEnv], int],
    config: SchedulingConfig = SchedulingConfig(),
    *,
    episodes: int = 300,
    seed: int = 190_000,
) -> dict[str, float]:
    weighted_tardiness = []
    setup_time = []
    mean_flow = []
    on_time_rate = []
    total_objective = []

    for ep in range(episodes):
        env = DynamicDispatchEnv(config)
        env.reset(seed=seed + ep)
        done = False
        while not done:
            action = int(policy(env))
            _, _, done, _ = env.step(action)

        weighted_tardiness.append(env.total_weighted_tardiness)
        setup_time.append(env.total_setup)
        mean_flow.append(env.total_flow_time / config.n_jobs)
        on_time_rate.append(env.on_time / config.n_jobs)
        total_objective.append(
            env.total_weighted_tardiness
            + config.setup_cost_weight * env.total_setup
        )

    obj = np.asarray(total_objective, dtype=float)
    return {
        "mean_objective": float(obj.mean()),
        "p90_objective": float(np.quantile(obj, 0.90)),
        "mean_weighted_tardiness": float(np.mean(weighted_tardiness)),
        "mean_setup_time": float(np.mean(setup_time)),
        "mean_flow_time": float(np.mean(mean_flow)),
        "mean_on_time_rate": float(np.mean(on_time_rate)),
    }


def historical_contextual_action(
    env: DynamicDispatchEnv,
    rng: np.random.Generator,
    *,
    exploration: float = 0.18,
) -> int:
    """Noisy state-dependent planner used to generate historical logs."""
    if not 0.0 <= exploration <= 1.0:
        raise ValueError("exploration must be in [0,1]")
    if rng.random() < exploration:
        return int(rng.integers(0, env.n_actions))

    state = env.state_vector()
    overdue_ratio = float(state[4])
    same_family_ratio = float(state[5])
    weighted_urgency = float(state[7])
    min_slack_scaled = float(state[3])
    queue_fraction = float(state[0])

    if overdue_ratio >= 0.18 or weighted_urgency >= 0.12:
        return ATC
    if (
        env.machine_family >= 0
        and same_family_ratio >= 0.45
        and queue_fraction >= 0.08
    ):
        return SETUP_AWARE
    if min_slack_scaled <= 0.12:
        return EDD
    return SPT


def generate_scheduling_dataset(
    config: SchedulingConfig = SchedulingConfig(),
    *,
    episodes: int = 600,
    seed: int = 185_000,
    exploration: float = 0.18,
) -> tuple[SchedulingTransition, ...]:
    rows = []
    rng = np.random.default_rng(seed)

    for ep in range(episodes):
        env = DynamicDispatchEnv(config)
        state = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = historical_contextual_action(
                env,
                rng,
                exploration=exploration,
            )
            next_state, reward, done, _ = env.step(action)
            rows.append(
                SchedulingTransition(
                    state=tuple(float(x) for x in state),
                    action=action,
                    reward=float(reward),
                    next_state=tuple(float(x) for x in next_state),
                    done=bool(done),
                )
            )
            state = next_state

    return tuple(rows)


class SchedulingQNetwork(nn.Module):
    def __init__(self, state_dim: int = 9, n_actions: int = 4, hidden_dim: int = 64):
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
class SchedulingCQLConfig:
    gradient_steps: int = 3500
    batch_size: int = 128
    learning_rate: float = 8e-4
    conservative_alpha: float = 0.8
    target_update: int = 200
    gamma: float = 1.0
    seed: int = 808


class OfflineSchedulingCQL:
    def __init__(
        self,
        config: SchedulingCQLConfig = SchedulingCQLConfig(),
    ):
        self.config = config
        set_global_seed(config.seed)
        self.online = SchedulingQNetwork()
        self.target = SchedulingQNetwork()
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(
            self.online.parameters(),
            lr=config.learning_rate,
        )
        self.rng = np.random.default_rng(config.seed)

    def fit(self, dataset: tuple[SchedulingTransition, ...]):
        if not dataset:
            raise ValueError("dataset must be non-empty")
        states = np.asarray([r.state for r in dataset], dtype=np.float32)
        actions = np.asarray([r.action for r in dataset], dtype=np.int64)
        rewards = np.asarray([r.reward for r in dataset], dtype=np.float32)
        next_states = np.asarray([r.next_state for r in dataset], dtype=np.float32)
        dones = np.asarray([r.done for r in dataset], dtype=np.float32)
        n = len(dataset)
        batch = min(self.config.batch_size, n)

        for step in range(self.config.gradient_steps):
            idx = self.rng.integers(0, n, size=batch)
            s = torch.as_tensor(states[idx], dtype=torch.float32)
            a = torch.as_tensor(actions[idx], dtype=torch.long)
            r = torch.as_tensor(rewards[idx], dtype=torch.float32)
            ns = torch.as_tensor(next_states[idx], dtype=torch.float32)
            d = torch.as_tensor(dones[idx], dtype=torch.float32)

            q_all = self.online(s)
            q_data = q_all.gather(1, a[:, None]).squeeze(1)
            with torch.no_grad():
                next_v = self.target(ns).max(dim=1).values
                target = r + (1.0 - d) * self.config.gamma * next_v

            td_loss = F.smooth_l1_loss(q_data, target)
            conservative = (
                torch.logsumexp(q_all, dim=1) - q_data
            ).mean()
            loss = td_loss + self.config.conservative_alpha * conservative

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
            self.optimizer.step()

            if (step + 1) % self.config.target_update == 0:
                self.target.load_state_dict(self.online.state_dict())
        return self

    def policy(self):
        def choose(env: DynamicDispatchEnv) -> int:
            state = env.state_vector()
            with torch.no_grad():
                q = self.online(
                    torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                )
            return int(torch.argmax(q, dim=-1).item())
        return choose


class SchedulingBehaviorCloner(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_behavior_cloner(
    dataset: tuple[SchedulingTransition, ...],
    *,
    epochs: int = 30,
    batch_size: int = 128,
    seed: int = 809,
):
    set_global_seed(seed)
    model = SchedulingBehaviorCloner()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    states = np.asarray([r.state for r in dataset], dtype=np.float32)
    actions = np.asarray([r.action for r in dataset], dtype=np.int64)

    for _ in range(epochs):
        order = rng.permutation(len(dataset))
        for start in range(0, len(dataset), batch_size):
            idx = order[start:start + batch_size]
            s = torch.as_tensor(states[idx], dtype=torch.float32)
            a = torch.as_tensor(actions[idx], dtype=torch.long)
            loss = F.cross_entropy(model(s), a)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def policy(env: DynamicDispatchEnv) -> int:
        with torch.no_grad():
            logits = model(
                torch.as_tensor(
                    env.state_vector(),
                    dtype=torch.float32,
                ).unsqueeze(0)
            )
        return int(torch.argmax(logits, dim=-1).item())

    return model, policy

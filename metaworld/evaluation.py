from __future__ import annotations

from typing import NamedTuple, Protocol

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from metaworld.env_dict import ALL_V3_ENVIRONMENTS


class Agent(Protocol):
    def eval_action(
        self, observations: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...


class MetaLearningAgent(Agent):
    def adapt_action(
        self, observations: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray]]: ...

    def adapt(self, rollouts: Rollout) -> None: ...


def _get_task_names(
    envs: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
) -> list[str]:
    metaworld_cls_to_task_name = {v.__name__: k for k, v in ALL_V3_ENVIRONMENTS.items()}
    return [
        metaworld_cls_to_task_name[task_name]
        for task_name in envs.get_attr("task_name")
    ]


def evaluation(
    agent: Agent,
    eval_envs: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
    num_episodes: int = 50,
) -> tuple[float, float, dict[str, float]]:
    terminate_on_success = np.all(eval_envs.get_attr("terminate_on_success")).item()
    eval_envs.call("toggle_terminate_on_success", True)

    obs: npt.NDArray[np.float64]
    obs, _ = eval_envs.reset()
    task_names = _get_task_names(eval_envs)
    successes = {task_name: 0 for task_name in set(task_names)}
    episodic_returns: dict[str, list[float]] = {
        task_name: [] for task_name in set(task_names)
    }

    def eval_done(returns):
        return all(len(r) >= num_episodes for _, r in returns.items())

    while not eval_done(episodic_returns):
        actions = agent.eval_action(obs)
        obs, _, terminations, truncations, infos = eval_envs.step(actions)
        for i, env_ended in enumerate(np.logical_or(terminations, truncations)):
            if env_ended:
                episodic_returns[task_names[i]].append(float(infos["episode"]["r"][i]))
                if len(episodic_returns[task_names[i]]) <= num_episodes:
                    successes[task_names[i]] += int(infos["success"][i])

    episodic_returns = {
        task_name: returns[:num_episodes]
        for task_name, returns in episodic_returns.items()
    }

    success_rate_per_task = {
        task_name: task_successes / num_episodes
        for task_name, task_successes in successes.items()
    }
    mean_success_rate = np.mean(list(success_rate_per_task.values()))
    mean_returns = np.mean(list(episodic_returns.values()))

    eval_envs.call("toggle_terminate_on_success", terminate_on_success)

    return float(mean_success_rate), float(mean_returns), success_rate_per_task


def metalearning_evaluation(
    agent: MetaLearningAgent,
    eval_envs: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
    adaptation_steps: int = 1,
    max_episode_steps: int = 500,
    adaptation_episodes: int = 10,
    num_episodes: int = 50,
    num_evals: int = 1,
) -> tuple[float, float, dict[str, float]]:
    task_names = _get_task_names(eval_envs)

    total_mean_success_rate = 0.0
    total_mean_return = 0.0

    success_rate_per_task = np.zeros((num_evals, len(set(task_names))))

    for i in range(num_evals):
        eval_envs.call("toggle_sample_tasks_on_reset", False)
        eval_envs.call("toggle_terminate_on_success", False)
        eval_envs.call("sample_tasks")
        obs: npt.NDArray[np.float64]
        obs, _ = eval_envs.reset()
        obs = np.stack(obs)  # type: ignore
        has_autoreset = np.full((eval_envs.num_envs,), False)
        eval_buffer = _MultiTaskRolloutBuffer(
            num_tasks=eval_envs.num_envs,
            rollouts_per_task=adaptation_episodes,
            max_episode_steps=max_episode_steps,
        )

        for _ in range(adaptation_steps):
            while not eval_buffer.ready:
                actions, aux_policy_outs = agent.adapt_action(obs)
                next_obs: npt.NDArray[np.float64]
                rewards: npt.NDArray[np.float64]
                next_obs, rewards, terminations, truncations, _ = eval_envs.step(
                    actions
                )
                if not has_autoreset.any():
                    eval_buffer.push(
                        obs,
                        actions,
                        rewards,
                        truncations,
                        log_probs=aux_policy_outs.get("log_probs"),
                        means=aux_policy_outs.get("means"),
                        stds=aux_policy_outs.get("stds"),
                    )
                has_autoreset = np.logical_or(terminations, truncations)
                obs = next_obs

            rollouts = eval_buffer.get()
            agent.adapt(rollouts)
            eval_buffer.reset()

        # Evaluation
        mean_success_rate, mean_return, _success_rate_per_task = evaluation(
            agent, eval_envs, num_episodes
        )
        total_mean_success_rate += mean_success_rate
        total_mean_return += mean_return
        success_rate_per_task[i] = np.array(list(_success_rate_per_task.values()))

    success_rates = (success_rate_per_task).mean(axis=0)
    task_success_rates = {
        task_name: success_rates[i] for i, task_name in enumerate(set(task_names))
    }

    return (
        total_mean_success_rate / num_evals,
        total_mean_return / num_evals,
        task_success_rates,
    )


class Rollout(NamedTuple):
    observations: npt.NDArray
    actions: npt.NDArray
    rewards: npt.NDArray
    dones: npt.NDArray

    # Auxiliary policy outputs
    log_probs: npt.NDArray | None = None
    means: npt.NDArray | None = None
    stds: npt.NDArray | None = None


class _MultiTaskRolloutBuffer:
    """A buffer to accumulate rollouts for multiple tasks.
    Useful for ML1, ML10, ML45, or on-policy MTRL algorithms.

    In Metaworld, all episodes are as long as the time limit (typically 500), thus in this buffer we assume
    fixed-length episodes and leverage that for optimisations."""

    rollouts: list[list[Rollout]]

    def __init__(
        self,
        num_tasks: int,
        rollouts_per_task: int,
        max_episode_steps: int,
    ):
        self.num_tasks = num_tasks
        self._rollouts_per_task = rollouts_per_task
        self._max_episode_steps = max_episode_steps

        self.reset()

    def reset(self):
        """Reset the buffer."""
        self.rollouts = [[] for _ in range(self.num_tasks)]
        self._running_rollouts = [[] for _ in range(self.num_tasks)]

    @property
    def ready(self) -> bool:
        """Returns whether or not a full batch of rollouts for each task has been sampled."""
        return all(len(t) == self._rollouts_per_task for t in self.rollouts)

    def get_single_task(
        self,
        task_idx: int,
    ) -> Rollout:
        """Compute returns and advantages for the collected rollouts.

        Returns a Rollout tuple for a single task where each array has the batch dimensions (Timestep,).
        The timesteps are multiple rollouts flattened into one time dimension."""
        assert task_idx < self.num_tasks, "Task index out of bounds."

        task_rollouts = Rollout(
            *map(lambda *xs: np.stack(xs), *self.rollouts[task_idx])
        )

        assert task_rollouts.observations.shape[:2] == (
            self._rollouts_per_task,
            self._max_episode_steps,
        ), "Buffer does not have the expected amount of data before sampling."

        return task_rollouts

    def get(
        self,
    ) -> Rollout:
        """Compute returns and advantages for the collected rollouts.

        Returns a Rollout tuple where each array has the batch dimensions (Task,Timestep,).
        The timesteps are multiple rollouts flattened into one time dimension."""
        rollouts_per_task = [
            Rollout(*map(lambda *xs: np.stack(xs), *t)) for t in self.rollouts
        ]
        all_rollouts = Rollout(*map(lambda *xs: np.stack(xs), *rollouts_per_task))
        assert all_rollouts.observations.shape[:3] == (
            self.num_tasks,
            self._rollouts_per_task,
            self._max_episode_steps,
        ), "Buffer does not have the expected amount of data before sampling."

        return all_rollouts

    def push(
        self,
        obs: npt.NDArray,
        actions: npt.NDArray,
        rewards: npt.NDArray,
        dones: npt.NDArray,
        log_probs: npt.NDArray | None = None,
        means: npt.NDArray | None = None,
        stds: npt.NDArray | None = None,
    ):
        """Add a batch of timesteps to the buffer. Multiple batch dims are supported, but they
        need to multiply to the buffer's meta batch size.

        If an episode finishes here for any of the envs, pop the full rollout into the rollout buffer.
        """
        assert np.prod(rewards.shape) == self.num_tasks

        obs = obs.copy()
        actions = actions.copy()
        assert obs.ndim == actions.ndim
        if (
            obs.ndim > 2 and actions.ndim > 2
        ):  # Flatten outer batch dims only if they exist
            obs = obs.reshape(-1, *obs.shape[2:])
            actions = actions.reshape(-1, *actions.shape[2:])

        rewards = rewards.reshape(-1, 1).copy()
        dones = dones.reshape(-1, 1).copy()
        if log_probs is not None:
            log_probs = log_probs.reshape(-1, 1).copy()
        if means is not None:
            means = means.copy()
            if means.ndim > 2:
                means = means.reshape(-1, *means.shape[2:])
        if stds is not None:
            stds = stds.copy()
            if stds.ndim > 2:
                stds = stds.reshape(-1, *stds.shape[2:])

        for i in range(self.num_tasks):
            timestep: tuple[npt.NDArray, ...] = (
                obs[i],
                actions[i],
                rewards[i],
                dones[i],
            )
            if log_probs is not None:
                timestep += (log_probs[i],)
            if means is not None:
                timestep += (means[i],)
            if stds is not None:
                timestep += (stds[i],)
            self._running_rollouts[i].append(timestep)

            if dones[i]:  # pop full rollouts into the rollouts buffer
                rollout = Rollout(
                    *map(lambda *xs: np.stack(xs), *self._running_rollouts[i])
                )
                self.rollouts[i].append(rollout)
                self._running_rollouts[i] = []

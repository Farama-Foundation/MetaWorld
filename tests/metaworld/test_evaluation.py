from __future__ import annotations

import random

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pytest

import metaworld  # noqa: F401
from metaworld import evaluation
from metaworld.policies import ENV_POLICY_MAP


class ScriptedPolicyAgent(evaluation.MetaLearningAgent):
    def __init__(
        self,
        envs: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
        num_rollouts: int | None = None,
        max_episode_steps: int | None = None,
    ):
        env_task_names = evaluation._get_task_names(envs)
        self.policies = [ENV_POLICY_MAP[task]() for task in env_task_names]  # type: ignore
        self.num_rollouts = num_rollouts
        self.max_episode_steps = max_episode_steps
        self.adapt_calls = 0

    def adapt_action(
        self, observations: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray]]:
        actions: list[npt.NDArray[np.float32]] = []
        num_envs = len(self.policies)
        for env_idx in range(num_envs):
            actions.append(self.policies[env_idx].get_action(observations[env_idx]))
        stacked_actions = np.stack(actions, axis=0, dtype=np.float64)
        return stacked_actions, {
            "log_probs": np.ones((num_envs,)),
            "means": stacked_actions,
            "stds": np.zeros((num_envs,)),
        }

    def eval_action(
        self, observations: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        actions: list[npt.NDArray[np.float32]] = []
        num_envs = len(self.policies)
        for env_idx in range(num_envs):
            actions.append(self.policies[env_idx].get_action(observations[env_idx]))
        stacked_actions = np.stack(actions, axis=0, dtype=np.float64)
        return stacked_actions

    def adapt(self, rollouts: evaluation.Rollout) -> None:
        assert self.num_rollouts is not None

        for key in [
            "observations",
            "rewards",
            "actions",
            "dones",
            "log_probs",
            "means",
            "stds",
        ]:
            assert len(getattr(rollouts, key).shape) >= 3
            assert getattr(rollouts, key).shape[0] == len(self.policies)
            assert getattr(rollouts, key).shape[1] == self.num_rollouts
            assert getattr(rollouts, key).shape[2] == self.max_episode_steps

        self.adapt_calls += 1


class RemovePartialObservabilityWrapper(gym.vector.VectorWrapper):
    def get_attr(self, name):
        return self.env.get_attr(name)

    def set_attr(self, name, values):
        return self.env.set_attr(name, values)

    def call(self, name, *args, **kwargs):
        return self.env.call(name, *args, **kwargs)

    def step(self, actions):
        self.env.set_attr("_partially_observable", False)
        return super().step(actions)


def test_evaluation():
    SEED = 42
    max_episode_steps = 300  # To speed up the test
    num_episodes = 50

    random.seed(SEED)
    np.random.seed(SEED)
    envs = gym.make_vec(
        "Meta-World/MT50",
        seed=SEED,
        max_episode_steps=max_episode_steps,
        vector_strategy="async",
    )
    agent = ScriptedPolicyAgent(envs)
    mean_success_rate, mean_returns, success_rate_per_task = evaluation.evaluation(
        agent, envs, num_episodes=num_episodes
    )
    assert isinstance(mean_returns, float)
    assert mean_success_rate >= 0.80
    assert len(success_rate_per_task) == envs.num_envs
    assert np.all(np.array(list(success_rate_per_task.values())) >= 0.80)


# @pytest.mark.skip
@pytest.mark.parametrize("benchmark", ("ML10", "ML45"))
def test_metalearning_evaluation(benchmark):
    SEED = 42

    max_episode_steps = 300
    meta_batch_size = 10  # Number of parallel envs

    adaptation_steps = 2  # Number of adaptation iterations
    adaptation_episodes = 2  # Number of train episodes per task in meta_batch_size per adaptation iteration
    num_evals = 50  # Number of different task vectors tested for each task
    num_episodes = 1  # Number of test episodes per task vector

    random.seed(SEED)
    np.random.seed(SEED)
    envs = gym.make_vec(
        f"Meta-World/{benchmark}-test",
        seed=SEED,
        vector_strategy="async",
        meta_batch_size=meta_batch_size,
        max_episode_steps=max_episode_steps,
    )
    envs = RemovePartialObservabilityWrapper(envs)
    agent = ScriptedPolicyAgent(envs, adaptation_episodes, max_episode_steps)
    (
        mean_success_rate,
        mean_returns,
        success_rate_per_task,
    ) = evaluation.metalearning_evaluation(
        agent,
        envs,
        max_episode_steps=max_episode_steps,
        num_episodes=num_episodes,
        adaptation_episodes=adaptation_episodes,
        adaptation_steps=adaptation_steps,
        num_evals=num_evals,
    )
    assert isinstance(mean_returns, float)
    assert mean_success_rate >= 0.80
    assert len(success_rate_per_task) == len(set(evaluation._get_task_names(envs)))
    assert np.all(np.array(list(success_rate_per_task.values())) >= 0.80)
    assert agent.adapt_calls == num_evals * adaptation_steps

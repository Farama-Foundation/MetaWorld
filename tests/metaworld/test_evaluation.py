from __future__ import annotations

import random
from contextlib import closing

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
        self.step_calls = 0
        self.resets = 0

    def reset(self, env_mask: npt.NDArray[np.bool_]) -> None:
        self.resets += np.sum(env_mask)  # type: ignore

    def init(self) -> None:
        return

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

    def step(self, timestep: evaluation.Timestep) -> None:
        assert "log_probs" in timestep.aux_policy_outputs
        assert "means" in timestep.aux_policy_outputs
        assert "stds" in timestep.aux_policy_outputs
        self.step_calls += 1

    def eval_action(
        self, observations: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        actions: list[npt.NDArray[np.float32]] = []
        num_envs = len(self.policies)
        for env_idx in range(num_envs):
            actions.append(self.policies[env_idx].get_action(observations[env_idx]))
        stacked_actions = np.stack(actions, axis=0, dtype=np.float64)
        return stacked_actions

    def adapt(self) -> None:
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
    mean_success_rate, mean_returns, success_rate_per_task, _ = evaluation.evaluation(
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
        evaluation_episodes=num_episodes,
        adaptation_episodes=adaptation_episodes,
        adaptation_steps=adaptation_steps,
        num_evals=num_evals,
    )
    assert isinstance(mean_returns, float)
    assert mean_success_rate >= 0.80
    assert len(success_rate_per_task) == len(set(evaluation._get_task_names(envs)))
    assert np.all(np.array(list(success_rate_per_task.values())) >= 0.80)
    assert agent.adapt_calls == num_evals * adaptation_steps
    assert (
        agent.step_calls
        == num_evals * adaptation_steps * adaptation_episodes * max_episode_steps
    )


class _FixedOutcomeEnv(gym.Env):
    """Toy env: succeeds at its first step, or never succeeds and truncates at `horizon`."""

    observation_space = gym.spaces.Box(-1.0, 1.0, (3,))
    action_space = gym.spaces.Box(-1.0, 1.0, (4,))

    def __init__(self, succeeds: bool, task_name: str, horizon: int = 5):
        self.succeeds = succeeds
        self.task_name = task_name
        self.horizon = horizon
        self.terminate_on_success = False
        self.t = 0

    def toggle_terminate_on_success(self, on: bool):
        self.terminate_on_success = on

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.zeros(3, np.float32), {}

    def step(self, action):
        self.t += 1
        terminated = self.succeeds and self.terminate_on_success
        truncated = (not terminated) and self.t >= self.horizon
        info = {"success": float(self.succeeds)}
        return np.zeros(3, np.float32), 0.0, terminated, truncated, info


class _ZeroAgent:
    def eval_action(self, observations):
        return np.zeros((observations.shape[0], 4))

    def reset(self, env_mask):
        pass


def _fixed_outcome_envs(outcomes, task_names):
    return gym.vector.SyncVectorEnv(
        [
            lambda s=s, n=n: gym.wrappers.RecordEpisodeStatistics(
                _FixedOutcomeEnv(s, n)
            )
            for s, n in zip(outcomes, task_names)
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


def test_evaluation_counts_episodes_per_env():
    # Four envs (goals) of one task, as in ML1: one succeeds immediately, three
    # never do. Each env must contribute its own first `num_episodes` episodes
    # -> 1/4, not the first episodes to finish for the task (which would all be
    # the fast successes -> 1.0).
    envs = _fixed_outcome_envs([True, False, False, False], ["SawyerReachEnvV3"] * 4)
    (
        mean_success_rate,
        _,
        success_rate_per_task,
        episodic_returns,
    ) = evaluation.evaluation(_ZeroAgent(), envs, num_episodes=3)
    assert mean_success_rate == pytest.approx(0.25)
    assert success_rate_per_task == {"reach-v3": pytest.approx(0.25)}
    assert len(episodic_returns["reach-v3"]) == 3 * 4


def test_evaluation_one_env_per_task_unchanged():
    # With one env per task name (the multi-task benchmarks) per-env and per-task
    # counting agree.
    envs = _fixed_outcome_envs([True, False], ["SawyerReachEnvV3", "SawyerPushEnvV3"])
    (
        mean_success_rate,
        _,
        success_rate_per_task,
        episodic_returns,
    ) = evaluation.evaluation(_ZeroAgent(), envs, num_episodes=3)
    assert success_rate_per_task == {
        "reach-v3": pytest.approx(1.0),
        "push-v3": pytest.approx(0.0),
    }
    assert mean_success_rate == pytest.approx(0.5)
    assert all(len(r) == 3 for r in episodic_returns.values())


class _ScheduledOutcomeEnv(_FixedOutcomeEnv):
    """Repeat (length, success, return) episodes, including after scoring stops."""

    def __init__(self, task_name, episodes):
        super().__init__(False, task_name)
        self.episodes = episodes
        self.episode_index = -1
        self.ends = []

    def reset(self, *, seed=None, options=None):
        self.episode_index += 1
        return super().reset(seed=seed, options=options)

    def step(self, action):
        self.t += 1
        length, success, episode_return = self.episodes[
            self.episode_index % len(self.episodes)
        ]
        ended = self.t >= length
        terminated = ended and success and self.terminate_on_success
        # Success and time limit may occur on the same transition.
        truncated = ended and (not success or length == 3)
        if terminated or truncated:
            self.ends.append((terminated, truncated))
        return (
            np.zeros(3, np.float32),
            episode_return if ended else 0.0,
            terminated,
            truncated,
            {"success": float(ended and success)},
        )


def _scheduled_envs(specs):
    return gym.vector.SyncVectorEnv(
        [
            lambda name=name, episodes=episodes: gym.wrappers.RecordEpisodeStatistics(
                _ScheduledOutcomeEnv(name, episodes)
            )
            for name, episodes in specs
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


def test_evaluation_unequal_goal_counts():
    # The documented denominator is num_episodes * num_envs, not task count.
    specs = [("SawyerReachEnvV3", [(1, True, 1.0)])] * 4 + [
        ("SawyerPushEnvV3", [(4, False, 0.0)])
    ]
    with closing(_scheduled_envs(specs)) as envs:
        success, returns, per_task, episodes = evaluation.evaluation(
            _ZeroAgent(), envs, num_episodes=2
        )
    assert per_task == {"reach-v3": 1.0, "push-v3": 0.0}
    assert episodes == {"reach-v3": [1.0] * 8, "push-v3": [0.0] * 2}
    assert returns == pytest.approx(0.8)
    assert success == pytest.approx(0.8)


@pytest.mark.parametrize("shared_task", [False, True])
def test_evaluation_mixed_episode_endings_and_returns(shared_task):
    second_task = "SawyerReachEnvV3" if shared_task else "SawyerPushEnvV3"
    specs = [
        ("SawyerReachEnvV3", [(1, True, 2.0), (2, False, -3.0), (1, True, 99.0)]),
        (second_task, [(3, True, 7.0), (4, False, 11.0)]),
    ]
    with closing(_scheduled_envs(specs)) as envs:
        result = evaluation.evaluation(_ZeroAgent(), envs, num_episodes=2)
        # The fast env continues running; none of its later 99 returns count.
        assert len(envs.envs[0].unwrapped.ends) > 2
        assert envs.envs[1].unwrapped.ends == [(True, True), (False, True)]
        assert envs.get_attr("terminate_on_success") == (False, False)
    success, returns, per_task, episodes = result
    expected = (
        {"reach-v3": [2.0, -3.0, 7.0, 11.0]}
        if shared_task
        else {"reach-v3": [2.0, -3.0], "push-v3": [7.0, 11.0]}
    )
    assert episodes == expected
    assert all(isinstance(r, float) for values in episodes.values() for r in values)
    assert per_task == {task: 0.5 for task in expected}
    assert success == pytest.approx(0.5)
    assert returns == pytest.approx(4.25)

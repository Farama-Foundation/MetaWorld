from __future__ import annotations

import random
from typing import Literal

import gymnasium as gym
import numpy as np
import pytest

import metaworld  # noqa: F401
from metaworld import _N_GOALS, SawyerXYZEnv
from metaworld.env_dict import (
    ALL_V3_ENVIRONMENTS,
    ALL_V3_ENVIRONMENTS_GOAL_HIDDEN,
    ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE,
    ML10_V3,
    ML45_V3,
    MT10_V3,
    MT50_V3,
    EnvDict,
    TrainTestEnvDict,
)


def _get_task_names(
    envs: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
) -> list[str]:
    metaworld_cls_to_task_name = {v.__name__: k for k, v in ALL_V3_ENVIRONMENTS.items()}
    return [
        metaworld_cls_to_task_name[task_name]
        for task_name in envs.get_attr("task_name")
    ]


@pytest.mark.parametrize("benchmark,env_dict", (("MT10", MT10_V3), ("MT50", MT50_V3)))
@pytest.mark.parametrize("vector_strategy", ("sync", "async"))
def test_mt_benchmarks(benchmark: str, env_dict: EnvDict, vector_strategy: str):
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    max_episode_steps = 10

    envs = gym.make_vec(
        f"Meta-World/{benchmark}",
        vector_strategy=vector_strategy,
        seed=SEED,
        use_one_hot=True,
        max_episode_steps=max_episode_steps,
    )

    # Assert vec is correct
    expected_vectorisation = getattr(
        gym.vector, f"{vector_strategy.capitalize()}VectorEnv"
    )
    assert isinstance(envs, expected_vectorisation)

    # Assert envs are correct
    task_names = _get_task_names(envs)
    assert envs.num_envs == len(env_dict.keys())
    assert set(task_names) == set(env_dict.keys())

    # Assert every env has N_GOALS goals
    envs_tasks = envs.get_attr("tasks")
    for env_tasks in envs_tasks:
        assert len(env_tasks) == _N_GOALS

    # Test wrappers: one hot obs, task sampling, max path length
    obs, _ = envs.reset()
    original_vecs = envs.get_attr("_last_rand_vec")

    has_truncated = False
    for _ in range(max_episode_steps + 1):
        obs, _, _, truncated, _ = envs.step(envs.action_space.sample())
        print(obs)
        env_one_hots = obs[:, -envs.num_envs :]
        env_ids = np.argmax(env_one_hots, axis=1)
        assert set(env_ids) == set(range(envs.num_envs))

        if any(truncated):
            has_truncated = True

    assert has_truncated

    new_vecs = envs.get_attr("_last_rand_vec")
    task_has_changed = False
    for og_vec, new_vec in zip(original_vecs, new_vecs):
        if np.any(og_vec != new_vec):
            task_has_changed = True
    assert task_has_changed

    partially_observable = all(envs.get_attr("_partially_observable"))
    assert not partially_observable


@pytest.mark.parametrize("env_name", ALL_V3_ENVIRONMENTS.keys())
def test_mt1(env_name: str):
    metaworld_cls_to_task_name = {v.__name__: k for k, v in ALL_V3_ENVIRONMENTS.items()}
    env = gym.make("Meta-World/MT1", env_name=env_name)
    assert isinstance(env.unwrapped, SawyerXYZEnv)
    assert len(env.get_wrapper_attr("tasks")) == _N_GOALS
    assert metaworld_cls_to_task_name[env.unwrapped.task_name] == env_name

    env.reset()
    assert not env.unwrapped._partially_observable


@pytest.mark.parametrize("env_name", ALL_V3_ENVIRONMENTS_GOAL_HIDDEN.keys())
def test_goal_hidden(env_name: str):
    env = gym.make("Meta-World/goal_hidden", env_name=env_name, seed=None)
    assert isinstance(env.unwrapped, SawyerXYZEnv)

    env.reset()
    assert env.unwrapped._partially_observable


@pytest.mark.parametrize("env_name", ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE.keys())
def test_goal_observable(env_name: str):
    env = gym.make("Meta-World/goal_observable", env_name=env_name, seed=None)
    assert isinstance(env.unwrapped, SawyerXYZEnv)

    env.reset()
    assert not env.unwrapped._partially_observable


@pytest.mark.parametrize("env_name", ALL_V3_ENVIRONMENTS.keys())
@pytest.mark.parametrize("split", ("train", "test"))
@pytest.mark.parametrize("vector_strategy", ("sync", "async"))
def test_ml1(env_name, split, vector_strategy):
    meta_batch_size = 10
    max_episode_steps = 10

    envs = gym.make_vec(
        f"Meta-World/ML1-{split}",
        env_name=env_name,
        vector_strategy=vector_strategy,
        meta_batch_size=meta_batch_size,
        max_episode_steps=max_episode_steps,
    )
    assert envs.num_envs == meta_batch_size
    task_names = _get_task_names(envs)
    assert all([task_name == env_name for task_name in task_names])

    # Assert vec is correct
    expected_vectorisation = getattr(
        gym.vector, f"{vector_strategy.capitalize()}VectorEnv"
    )
    assert isinstance(envs, expected_vectorisation)

    envs_tasks = envs.get_attr("tasks")
    total_tasks = sum([len(env_tasks) for env_tasks in envs_tasks])
    assert total_tasks == _N_GOALS

    partially_observable = all(envs.get_attr("_partially_observable"))
    assert partially_observable


@pytest.mark.parametrize("benchmark,env_dict", (("ML10", ML10_V3), ("ML45", ML45_V3)))
@pytest.mark.parametrize("split", ("train", "test"))
@pytest.mark.parametrize("vector_strategy", ("sync", "async"))
def test_ml_benchmarks(
    benchmark: str,
    env_dict: TrainTestEnvDict,
    split: Literal["train", "test"],
    vector_strategy: str,
):
    meta_batch_size = 20 if benchmark != "ML45" else 45
    total_tasks_per_cls = _N_GOALS
    if benchmark == "ML45":
        total_tasks_per_cls = 45
    elif benchmark == "ML10" and split == "test":
        total_tasks_per_cls = 40
    max_episode_steps = 10

    envs = gym.make_vec(
        f"Meta-World/{benchmark}-{split}",
        vector_strategy=vector_strategy,
        meta_batch_size=meta_batch_size,
        max_episode_steps=max_episode_steps,
        total_tasks_per_cls=total_tasks_per_cls,
    )
    assert envs.num_envs == meta_batch_size
    task_names = _get_task_names(envs)  # type: ignore
    assert set(task_names) == set(env_dict[split].keys())

    # Assert vec is correct
    expected_vectorisation = getattr(
        gym.vector, f"{vector_strategy.capitalize()}VectorEnv"
    )
    assert isinstance(envs, expected_vectorisation)

    envs_tasks = envs.get_attr("tasks")
    tasks_per_env = {}
    for task in env_dict[split].keys():
        tasks_per_env[task] = 0

    for env_tasks, env_name in zip(envs_tasks, task_names):
        tasks_per_env[env_name] += len(env_tasks)

    for task in env_dict[split].keys():
        assert tasks_per_env[task] == total_tasks_per_cls

    partially_observable = all(envs.get_attr("_partially_observable"))
    assert partially_observable

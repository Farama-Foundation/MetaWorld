import memory_profiler
import pytest

from metaworld.env_dict import ALL_V3_ENVIRONMENTS
from tests.helpers import step_env


def build_and_step(env_cls):
    env = env_cls()
    step_env(env, max_path_length=150, iterations=10, render=False)
    env.close()


def build_and_step_all(classes):
    envs = []
    for env_cls in classes:
        env = build_and_step(env_cls)
        envs += [env]


@pytest.fixture(scope="module")
def mt50_usage():
    profile = {}
    for env_cls in ALL_V3_ENVIRONMENTS.values():
        target = (build_and_step, [env_cls], {})
        memory_usage = memory_profiler.memory_usage(target)
        profile[env_cls] = max(memory_usage)

    return profile


@pytest.mark.skip
@pytest.mark.parametrize("env_cls", ALL_V3_ENVIRONMENTS.values())
def test_max_memory_usage(env_cls, mt50_usage):
    # No env should use more  than 250MB
    #
    # Note: this is quite a bit higher than the average usage cap, because
    # loading a single environment incurs a fixed memory overhead which can't
    # be shared among environment in the same process
    assert mt50_usage[env_cls] < 250


@pytest.mark.skip
def test_avg_memory_usage():
    # average usage no greater than 60MB/env
    target = (build_and_step_all, [ALL_V3_ENVIRONMENTS.values()], {})
    usage = memory_profiler.memory_usage(target)
    average = max(usage) / len(ALL_V3_ENVIRONMENTS)
    assert average < 60


@pytest.mark.skip
def test_from_task_memory_usage():
    target = (ALL_V3_ENVIRONMENTS["reach-v1"], (), {})
    usage = memory_profiler.memory_usage(target)
    assert max(usage) < 250

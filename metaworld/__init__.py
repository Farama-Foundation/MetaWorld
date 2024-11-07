"""The public-facing Metaworld API."""

from __future__ import annotations

import abc
import pickle
from collections import OrderedDict
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import gymnasium as gym  # type: ignore
import numpy as np
import numpy.typing as npt

# noqa: D104
from gymnasium.envs.registration import register
from numpy.typing import NDArray

import metaworld  # type: ignore
import metaworld.env_dict as _env_dict
from metaworld.env_dict import (
    ALL_V3_ENVIRONMENTS,
    ALL_V3_ENVIRONMENTS_GOAL_HIDDEN,
    ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE,
)
from metaworld.sawyer_xyz_env import SawyerXYZEnv  # type: ignore
from metaworld.types import Task  # type: ignore
from metaworld.wrappers import (
    AutoTerminateOnSuccessWrapper,
    OneHotWrapper,
    PseudoRandomTaskSelectWrapper,
    RandomTaskSelectWrapper,
)


class MetaWorldEnv(abc.ABC):
    """Environment that requires a task before use.

    Takes no arguments to its constructor, and raises an exception if used
    before `set_task` is called.
    """

    @abc.abstractmethod
    def set_task(self, task: Task) -> None:
        """Sets the task.
        Args:
            task: The task to set.
        Raises:
            ValueError: If `task.env_name` is different from the current task.
        """
        raise NotImplementedError


class Benchmark(abc.ABC):
    """A Benchmark.

    When used to evaluate an algorithm, only a single instance should be used.
    """

    _train_classes: _env_dict.EnvDict
    _test_classes: _env_dict.EnvDict
    _train_tasks: list[Task]
    _test_tasks: list[Task]

    @abc.abstractmethod
    def __init__(self):
        pass

    @property
    def train_classes(self) -> _env_dict.EnvDict:
        """Returns all of the environment classes used for training."""
        return self._train_classes

    @property
    def test_classes(self) -> _env_dict.EnvDict:
        """Returns all of the environment classes used for testing."""
        return self._test_classes

    @property
    def train_tasks(self) -> list[Task]:
        """Returns all of the training tasks for this benchmark."""
        return self._train_tasks

    @property
    def test_tasks(self) -> list[Task]:
        """Returns all of the test tasks for this benchmark."""
        return self._test_tasks


_ML_OVERRIDE = dict(partially_observable=True)
"""The overrides for the Meta-Learning benchmarks. Disables the inclusion of the goal position in the observation."""

_MT_OVERRIDE = dict(partially_observable=False)
"""The overrides for the Multi-Task benchmarks. Enables the inclusion of the goal position in the observation."""

_N_GOALS = 50
"""The number of goals to generate for each environment."""


def _encode_task(env_name, data) -> Task:
    """Instantiates a new `Task` object after pickling the data.

    Args:
        env_name: The name of the environment.
        data: The task data (will be pickled).

    Returns:
        A `Task` object.
    """
    return Task(env_name=env_name, data=pickle.dumps(data))


def _make_tasks(
    classes: _env_dict.EnvDict,
    args_kwargs: _env_dict.EnvArgsKwargsDict,
    kwargs_override: dict,
    seed: int | None = None,
) -> list[Task]:
    """Initialises goals for a given set of environments.

    Args:
        classes: The environment classes as an `EnvDict`.
        args_kwargs: The environment arguments and keyword arguments.
        kwargs_override: Any kwarg overrides.
        seed: The random seed to use.

    Returns:
        A flat list of `Task` objects, `_N_GOALS` for each environment in `classes`.
    """
    # Cache existing random state
    if seed is not None:
        st0 = np.random.get_state()
        np.random.seed(seed)

    tasks = []
    for env_name, args in args_kwargs.items():
        kwargs = args["kwargs"].copy()
        assert isinstance(kwargs, dict)
        assert len(args["args"]) == 0

        # Init env
        env = classes[env_name]()
        env._freeze_rand_vec = False
        env._set_task_called = True
        rand_vecs: list[npt.NDArray[Any]] = []

        # Set task
        del kwargs["task_id"]
        env._set_task_inner(**kwargs)

        for _ in range(_N_GOALS):  # Generate random goals
            env.reset()
            assert env._last_rand_vec is not None
            rand_vecs.append(env._last_rand_vec)
        unique_task_rand_vecs = np.unique(np.array(rand_vecs), axis=0)
        assert (
            unique_task_rand_vecs.shape[0] == _N_GOALS
        ), f"Only generated {unique_task_rand_vecs.shape[0]} unique goals, not {_N_GOALS}"
        env.close()

        # Create a task for each random goal
        for rand_vec in rand_vecs:
            kwargs = args["kwargs"].copy()
            assert isinstance(kwargs, dict)
            del kwargs["task_id"]

            kwargs.update(dict(rand_vec=rand_vec, env_cls=classes[env_name]))
            kwargs.update(kwargs_override)

            tasks.append(_encode_task(env_name, kwargs))

        del env

    # Restore random state
    if seed is not None:
        np.random.set_state(st0)

    return tasks


# MT Benchmarks


class MT1(Benchmark):
    """
    The MT1 benchmark.
    A goal-conditioned RL environment for a single Metaworld task.
    """

    ENV_NAMES = list(_env_dict.ALL_V3_ENVIRONMENTS.keys())

    def __init__(self, env_name, seed=None):
        super().__init__()
        if env_name not in _env_dict.ALL_V3_ENVIRONMENTS:
            raise ValueError(f"{env_name} is not a V3 environment")
        cls = _env_dict.ALL_V3_ENVIRONMENTS[env_name]
        self._train_classes = OrderedDict([(env_name, cls)])
        self._test_classes = OrderedDict([(env_name, cls)])
        args_kwargs = _env_dict.ML1_args_kwargs[env_name]

        self._train_tasks = _make_tasks(
            self._train_classes, {env_name: args_kwargs}, _MT_OVERRIDE, seed=seed
        )

        self._test_tasks = []


class MT10(Benchmark):
    """
    The MT10 benchmark.
    Contains 10 tasks in its train set.
    Has an empty test set.
    """

    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.MT10_V3
        self._test_classes = OrderedDict()
        train_kwargs = _env_dict.MT10_V3_ARGS_KWARGS
        self._train_tasks = _make_tasks(
            self._train_classes, train_kwargs, _MT_OVERRIDE, seed=seed
        )

        self._test_tasks = []
        self._test_classes = []


class MT50(Benchmark):
    """
    The MT50 benchmark.
    Contains all (50) tasks in its train set.
    Has an empty test set.
    """

    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.MT50_V3
        self._test_classes = OrderedDict()
        train_kwargs = _env_dict.MT50_V3_ARGS_KWARGS
        self._train_tasks = _make_tasks(
            self._train_classes, train_kwargs, _MT_OVERRIDE, seed=seed
        )

        self._test_tasks = []
        self._test_classes = None


# ML Benchmarks


class ML1(Benchmark):
    """
    The ML1 benchmark.
    A meta-RL environment for a single Metaworld task.
    The train and test set contain different goal positions.
    The goal position is not part of the observation.
    """

    ENV_NAMES = list(_env_dict.ALL_V3_ENVIRONMENTS.keys())

    def __init__(self, env_name, seed=None):
        super().__init__()
        if env_name not in _env_dict.ALL_V3_ENVIRONMENTS:
            raise ValueError(f"{env_name} is not a V3 environment")

        cls = _env_dict.ALL_V3_ENVIRONMENTS[env_name]
        self._train_classes = OrderedDict([(env_name, cls)])
        self._test_classes = self._train_classes
        args_kwargs = _env_dict.ML1_args_kwargs[env_name]

        self._train_tasks = _make_tasks(
            self._train_classes, {env_name: args_kwargs}, _ML_OVERRIDE, seed=seed
        )
        self._test_tasks = _make_tasks(
            self._test_classes,
            {env_name: args_kwargs},
            _ML_OVERRIDE,
            seed=(seed + 1 if seed is not None else seed),
        )


class ML10(Benchmark):
    """
    The ML10 benchmark.
    Contains 10 tasks in its train set and 5 tasks in its test set.
    The goal position is not part of the observation.
    """

    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.ML10_V3["train"]
        self._test_classes = _env_dict.ML10_V3["test"]
        train_kwargs = _env_dict.ML10_ARGS_KWARGS["train"]

        test_kwargs = _env_dict.ML10_ARGS_KWARGS["test"]
        self._train_tasks = _make_tasks(
            self._train_classes, train_kwargs, _ML_OVERRIDE, seed=seed
        )

        self._test_tasks = _make_tasks(
            self._test_classes, test_kwargs, _ML_OVERRIDE, seed=seed
        )


class ML45(Benchmark):
    """
    The ML45 benchmark.
    Contains 45 tasks in its train set and 5 tasks in its test set (50 in total).
    The goal position is not part of the observation.
    """

    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.ML45_V3["train"]
        self._test_classes = _env_dict.ML45_V3["test"]
        train_kwargs = _env_dict.ML45_ARGS_KWARGS["train"]
        test_kwargs = _env_dict.ML45_ARGS_KWARGS["test"]

        self._train_tasks = _make_tasks(
            self._train_classes, train_kwargs, _ML_OVERRIDE, seed=seed
        )
        self._test_tasks = _make_tasks(
            self._test_classes, test_kwargs, _ML_OVERRIDE, seed=seed
        )


def _make_single_env(
    name: str,
    seed: int | None = None,
    max_episode_steps: int | None = None,
    use_one_hot: bool = False,
    env_id: int | None = None,
    num_tasks: int | None = None,
    terminate_on_success: bool = False,
) -> gym.Env:
    def init_each_env(
        env_cls: type[SawyerXYZEnv], name: str, seed: int | None
    ) -> gym.Env:
        env = env_cls()
        env = gym.wrappers.TimeLimit(env, max_episode_steps or env.max_path_length)
        if terminate_on_success:
            env = AutoTerminateOnSuccessWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if use_one_hot:
            assert env_id is not None, "Need to pass env_id through constructor"
            assert num_tasks is not None, "Need to pass num_tasks through constructor"
            env = OneHotWrapper(env, env_id, num_tasks)
        tasks = [task for task in benchmark.train_tasks if task.env_name in name]
        env = RandomTaskSelectWrapper(env, tasks, seed=seed)
        return env

    if "MT1-" in name:
        name = name.replace("MT1-", "")
        benchmark = MT1(name, seed=seed)
        return init_each_env(
            env_cls=benchmark.train_classes[name], name=name, seed=seed
        )
    elif "ML1-" in name:
        benchmark = ML1(
            name.replace("ML1-train-" if "train" in name else "ML1-test-", ""),
            seed=seed,
        )  # type: ignore
        if "train" in name:
            return init_each_env(
                env_cls=benchmark.train_classes[name.replace("ML1-train-", "")],
                name=name + "-train",
                seed=seed,
            )  # type: ignore
        elif "test" in name:
            return init_each_env(
                env_cls=benchmark.test_classes[name.replace("ML1-test-", "")],
                name=name + "-test",
                seed=seed,
            )


make_single_mt = partial(_make_single_env, terminate_on_success=False)


def _make_single_ml(
    name: str,
    seed: int,
    tasks_per_env: int,
    env_num: int,
    max_episode_steps: int | None = None,
    split: str = "train",
    terminate_on_success: bool = False,
    task_select: str = "random",
    total_tasks_per_cls: int | None = None,
):
    benchmark = ML1(
        name.replace("ML1-train-" if "train" in name else "ML1-test-", ""),
        seed=seed,
    )  # type: ignore
    cls = (
        benchmark.train_classes[name.replace("ML1-train-", "")]
        if split == "train"
        else benchmark.test_classes[name.replace("ML1-test-", "")]
    )
    tasks = benchmark.train_tasks if split == "train" else benchmark.test_tasks

    if total_tasks_per_cls is not None:
        tasks = tasks[:total_tasks_per_cls]
    tasks = [tasks[i::tasks_per_env] for i in range(0, tasks_per_env)][env_num]

    def make_env(env_cls: type[SawyerXYZEnv], tasks: list) -> gym.Env:
        env = env_cls()
        env = gym.wrappers.TimeLimit(env, max_episode_steps or env.max_path_length)
        env = AutoTerminateOnSuccessWrapper(env)
        env.toggle_terminate_on_success(terminate_on_success)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if task_select != "random":
            env = PseudoRandomTaskSelectWrapper(env, tasks)
        else:
            env = RandomTaskSelectWrapper(env, tasks)
        return env

    return make_env(cls, tasks)


make_single_ml_train = partial(
    _make_single_ml,
    terminate_on_success=False,
    task_select="pseudorandom",
    split="train",
)
make_single_ml_test = partial(
    _make_single_ml, terminate_on_success=True, task_select="pseudorandom", split="test"
)


def register_mw_envs():
    for name in ALL_V3_ENVIRONMENTS:
        kwargs = {"name": "MT1-" + name}
        register(
            id=f"Meta-World/{name}",
            entry_point="metaworld:make_single_mt",
            kwargs=kwargs,
        )
        kwargs = {"name": "ML1-train-" + name}
        register(
            id=f"Meta-World/ML1-train-{name}",
            entry_point="metaworld:make_single_ml_train",
            kwargs=kwargs,
        )
        kwargs = {"name": "ML1-test-" + name}
        register(
            id=f"Meta-World/ML1-test-{name}",
            entry_point="metaworld:make_single_ml_test",
            kwargs=kwargs,
        )

    for name_hid in ALL_V3_ENVIRONMENTS_GOAL_HIDDEN:
        kwargs = {}
        register(
            id=f"Meta-World/{name_hid}",
            entry_point=lambda seed: ALL_V3_ENVIRONMENTS_GOAL_HIDDEN[name_hid](
                seed=seed
            ),
            kwargs=kwargs,
        )

    for name_obs in ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE:
        kwargs = {}
        register(
            id=f"Meta-World/{name_obs}",
            entry_point=lambda seed: ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[name_obs](
                seed=seed
            ),
            kwargs=kwargs,
        )

    kwargs = {}
    register(
        id="Meta-World/MT10-sync",
        vector_entry_point=lambda seed=None, use_one_hot=False, num_envs=None, *args, **lamb_kwargs: gym.vector.SyncVectorEnv(
            [
                partial(
                    make_single_mt,
                    "MT1-" + env_name,
                    num_tasks=10,
                    env_id=idx,
                    seed=None if not seed else seed + idx,
                    use_one_hot=use_one_hot,
                    *args,
                    **lamb_kwargs,
                )
                for idx, env_name in enumerate(list(_env_dict.MT10_V3.keys()))
            ],
        ),
        kwargs=kwargs,
    )
    register(
        id="Meta-World/MT50-sync",
        vector_entry_point=lambda seed=None, use_one_hot=False, num_envs=None, *args, **lamb_kwargs: gym.vector.SyncVectorEnv(
            [
                partial(
                    make_single_mt,
                    "MT1-" + env_name,
                    num_tasks=50,
                    env_id=idx,
                    seed=None if not seed else seed + idx,
                    use_one_hot=use_one_hot,
                    *args,
                    **lamb_kwargs,
                )
                for idx, env_name in enumerate(list(_env_dict.MT50_V3.keys()))
            ]
        ),
        kwargs=kwargs,
    )

    register(
        id="Meta-World/MT50-async",
        vector_entry_point=lambda seed=None, use_one_hot=False, num_envs=None, *args, **lamb_kwargs: gym.vector.AsyncVectorEnv(
            [
                partial(
                    make_single_mt,
                    "MT1-" + env_name,
                    num_tasks=50,
                    env_id=idx,
                    seed=None if not seed else seed + idx,
                    use_one_hot=use_one_hot,
                    *args,
                    **lamb_kwargs,
                )
                for idx, env_name in enumerate(list(_env_dict.MT50_V3.keys()))
            ]
        ),
        kwargs=kwargs,
    )

    register(
        id="Meta-World/MT10-async",
        vector_entry_point=lambda seed=None, use_one_hot=False, num_envs=None, *args, **lamb_kwargs: gym.vector.AsyncVectorEnv(
            [
                partial(
                    make_single_mt,
                    "MT1-" + env_name,
                    num_tasks=10,
                    env_id=idx,
                    seed=None if not seed else seed + idx,
                    use_one_hot=use_one_hot,
                    *args,
                    **lamb_kwargs,
                )
                for idx, env_name in enumerate(list(_env_dict.MT10_V3.keys()))
            ]
        ),
        kwargs=kwargs,
    )

    register(
        id="Meta-World/ML10-train-sync",
        vector_entry_point=lambda seed=None, meta_batch_size=20, num_envs=None, *args, **lamb_kwargs: gym.vector.SyncVectorEnv(
            [
                partial(
                    make_single_ml_train,
                    "ML1-train-" + env_name,
                    tasks_per_env=meta_batch_size // 10,
                    env_num=idx % (meta_batch_size // 10),
                    seed=None if not seed else seed + (idx // (meta_batch_size // 10)),
                    *args,
                    **lamb_kwargs,
                )
                for idx, env_name in enumerate(
                    sorted(
                        list(_env_dict.ML10_V3["train"].keys())
                        * (meta_batch_size // 10)
                    )
                )
            ]
        ),
        kwargs=kwargs,
    )

    register(
        id="Meta-World/ML10-test-sync",
        vector_entry_point=lambda seed=None, meta_batch_size=20, num_envs=None, *args, **lamb_kwargs: gym.vector.SyncVectorEnv(
            [
                partial(
                    make_single_ml_test,
                    "ML1-test-" + env_name,
                    tasks_per_env=meta_batch_size // 5,
                    env_num=idx % (meta_batch_size // 5),
                    seed=None if not seed else seed + (idx // (meta_batch_size // 5)),
                    total_tasks_per_cls=40,
                    *args,
                    **lamb_kwargs,
                )
                for idx, env_name in enumerate(
                    sorted(
                        list(_env_dict.ML10_V3["test"].keys()) * (meta_batch_size // 5)
                    )
                )
            ]
        ),
        kwargs=kwargs,
    )

    register(
        id="Meta-World/ML10-train-async",
        vector_entry_point=lambda seed=None, meta_batch_size=20, num_envs=None, *args, **lamb_kwargs: gym.vector.AsyncVectorEnv(
            [
                partial(
                    make_single_ml_train,
                    "ML1-train-" + env_name,
                    tasks_per_env=meta_batch_size // 10,
                    env_num=idx % (meta_batch_size // 10),
                    seed=None if not seed else seed + (idx // (meta_batch_size // 10)),
                    *args,
                    **lamb_kwargs,
                )
                for idx, env_name in enumerate(
                    sorted(
                        list(_env_dict.ML10_V3["train"].keys())
                        * (meta_batch_size // 10)
                    )
                )
            ]
        ),
        kwargs=kwargs,
    )

    register(
        id="Meta-World/ML10-test-async",
        vector_entry_point=lambda seed=None, meta_batch_size=20, num_envs=None, *args, **lamb_kwargs: gym.vector.AsyncVectorEnv(
            [
                partial(
                    make_single_ml_test,
                    "ML1-test-" + env_name,
                    tasks_per_env=meta_batch_size // 5,
                    env_num=idx % (meta_batch_size // 5),
                    seed=None if not seed else seed + (idx // (meta_batch_size // 5)),
                    total_tasks_per_cls=40,
                    *args,
                    **lamb_kwargs,
                )
                for idx, env_name in enumerate(
                    sorted(
                        list(_env_dict.ML10_V3["test"].keys()) * (meta_batch_size // 5)
                    )
                )
            ]
        ),
        kwargs=kwargs,
    )

    register(
        id="Meta-World/ML45-train-sync",
        vector_entry_point=lambda seed=None, meta_batch_size=45, num_envs=None, *args, **lamb_kwargs: gym.vector.SyncVectorEnv(
            [
                partial(
                    make_single_ml_train,
                    "ML1-train-" + env_name,
                    tasks_per_env=meta_batch_size // 45,
                    env_num=idx % (meta_batch_size // 45),
                    seed=None if not seed else seed + (idx // (meta_batch_size // 45)),
                    *args,
                    **lamb_kwargs,
                )
                for idx, env_name in enumerate(
                    sorted(
                        list(_env_dict.ML45_V3["train"].keys())
                        * (meta_batch_size // 45)
                    )
                )
            ]
        ),
        kwargs=kwargs,
    )

    register(
        id="Meta-World/ML45-test-sync",
        vector_entry_point=lambda seed=None, meta_batch_size=45, num_envs=None, *args, **lamb_kwargs: gym.vector.SyncVectorEnv(
            [
                partial(
                    make_single_ml_test,
                    "ML1-test-" + env_name,
                    tasks_per_env=meta_batch_size // 5,
                    env_num=idx % (meta_batch_size // 5),
                    seed=None if not seed else seed + (idx // (meta_batch_size // 5)),
                    total_tasks_per_cls=45,
                    *args,
                    **lamb_kwargs,
                )
                for idx, env_name in enumerate(
                    sorted(
                        list(_env_dict.ML45_V3["test"].keys()) * (meta_batch_size // 5)
                    )
                )
            ]
        ),
        kwargs=kwargs,
    )

    register(
        id="Meta-World/ML45-train-async",
        vector_entry_point=lambda seed=None, meta_batch_size=45, num_envs=None, *args, **lamb_kwargs: gym.vector.AsyncVectorEnv(
            [
                partial(
                    make_single_ml_train,
                    "ML1-train-" + env_name,
                    tasks_per_env=meta_batch_size // 45,
                    env_num=idx % (meta_batch_size // 45),
                    seed=None if not seed else seed + (idx // (meta_batch_size // 45)),
                    *args,
                    **lamb_kwargs,
                )
                for idx, env_name in enumerate(
                    sorted(
                        list(_env_dict.ML45_V3["train"].keys())
                        * (meta_batch_size // 45)
                    )
                )
            ]
        ),
        kwargs=kwargs,
    )

    register(
        id="Meta-World/ML45-test-async",
        vector_entry_point=lambda seed=None, meta_batch_size=45, num_envs=None, *args, **lamb_kwargs: gym.vector.AsyncVectorEnv(
            [
                partial(
                    make_single_ml_test,
                    "ML1-test-" + env_name,
                    tasks_per_env=meta_batch_size // 5,
                    env_num=idx % (meta_batch_size // 5),
                    seed=None if not seed else seed + (idx // (meta_batch_size // 5)),
                    total_tasks_per_cls=45,
                    *args,
                    **lamb_kwargs,
                )
                for idx, env_name in enumerate(
                    sorted(
                        list(_env_dict.ML45_V3["test"].keys()) * (meta_batch_size // 5)
                    )
                )
            ]
        ),
        kwargs=kwargs,
    )

    register(
        id="Meta-World/custom-mt-envs-sync",
        vector_entry_point=lambda seed=None, use_one_hot=False, envs_list=None, num_envs=None, *args, **lamb_kwargs: gym.vector.SyncVectorEnv(
            [
                partial(
                    make_single_mt,
                    "MT1-" + env_name,
                    num_tasks=len(envs_list),
                    env_id=idx,
                    seed=None if not seed else seed + idx,
                    use_one_hot=use_one_hot,
                    *args,
                    **lamb_kwargs,
                )
                for idx, env_name in enumerate(envs_list)
            ]
        ),
        kwargs=kwargs,
    )

    register(
        id="Meta-World/custom-mt-envs-async",
        vector_entry_point=lambda seed=None, use_one_hot=False, envs_list=None, num_envs=None, *args, **lamb_kwargs: gym.vector.AsyncVectorEnv(
            [
                partial(
                    make_single_mt,
                    "MT1-" + env_name,
                    num_tasks=len(envs_list),
                    env_id=idx,
                    seed=None if not seed else seed + idx,
                    use_one_hot=use_one_hot,
                    *args,
                    **lamb_kwargs,
                )
                for idx, env_name in enumerate(envs_list)
            ]
        ),
        kwargs=kwargs,
    )

    register(
        id="Meta-World/custom-ml-envs-sync",
        vector_entry_point=lambda envs_list, seed=None, num_envs=None, meta_batch_size=None, *args, **lamb_kwargs: gym.vector.SyncVectorEnv(
            [
                partial(
                    make_single_ml_train,
                    "ML1-train-" + env_name,
                    tasks_per_env=1,
                    env_num=0,
                    seed=None if not seed else seed + idx,
                    *args,
                    **lamb_kwargs,
                )
                for idx, env_name in enumerate(envs_list)
            ]
        ),
        kwargs=kwargs,
    )

    register(
        id="Meta-World/custom-ml-envs-async",
        vector_entry_point=lambda envs_list, seed=None, meta_batch_size=None, num_envs=None, *args, **lamb_kwargs: gym.vector.AsyncVectorEnv(
            [
                partial(
                    make_single_ml_train,
                    "ML1-train-" + env_name,
                    tasks_per_env=1,
                    env_num=0,
                    seed=None if not seed else seed + idx,
                    *args,
                    **lamb_kwargs,
                )
                for idx, env_name in enumerate(envs_list)
            ]
        ),
        kwargs=kwargs,
    )


register_mw_envs()
__all__: list[str] = []

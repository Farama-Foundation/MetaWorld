"""The public-facing Metaworld API."""

from __future__ import annotations

import abc
import pickle
from collections import OrderedDict
from functools import partial
from typing import Any, Literal

import gymnasium as gym  # type: ignore
import numpy as np
import numpy.typing as npt

# noqa: D104
from gymnasium.envs.registration import register

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
        self._test_classes = []


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


class CustomML(Benchmark):
    """
    A custom meta RL benchmark.
    Provide the desired train and test env names during initialisation.
    """

    def __init__(self, train_envs: list[str], test_envs: list[str], seed=None):
        if len(set(train_envs).intersection(set(test_envs))) != 0:
            raise ValueError("The test tasks cannot contain any of the train tasks.")

        self._train_classes = _env_dict._get_env_dict(train_envs)
        train_kwargs = _env_dict._get_args_kwargs(
            ALL_V3_ENVIRONMENTS, self._train_classes
        )

        self._test_classes = _env_dict._get_env_dict(test_envs)
        test_kwargs = _env_dict._get_args_kwargs(
            ALL_V3_ENVIRONMENTS, self._test_classes
        )

        self._train_tasks = _make_tasks(
            self._train_classes, train_kwargs, _ML_OVERRIDE, seed=seed
        )
        self._test_tasks = _make_tasks(
            self._test_classes, test_kwargs, _ML_OVERRIDE, seed=seed
        )


def _init_each_env(
    env_cls: type[SawyerXYZEnv],
    tasks: list[Task],
    seed: int | None = None,
    max_episode_steps: int | None = None,
    terminate_on_success: bool = False,
    use_one_hot: bool = False,
    env_id: int | None = None,
    num_tasks: int | None = None,
    task_select: Literal["random", "pseudorandom"] = "random",
) -> gym.Env:
    env: gym.Env = env_cls()
    if seed is not None:
        env.seed(seed)  # type: ignore
    env = gym.wrappers.TimeLimit(env, max_episode_steps or env.max_path_length)  # type: ignore
    env = AutoTerminateOnSuccessWrapper(env)
    env.toggle_terminate_on_success(terminate_on_success)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if use_one_hot:
        assert env_id is not None, "Need to pass env_id through constructor"
        assert num_tasks is not None, "Need to pass num_tasks through constructor"
        env = OneHotWrapper(env, env_id, num_tasks)
    if task_select != "random":
        env = PseudoRandomTaskSelectWrapper(env, tasks)
    else:
        env = RandomTaskSelectWrapper(env, tasks)
    return env


def make_mt_envs(
    name: str,
    seed: int | None = None,
    max_episode_steps: int | None = None,
    use_one_hot: bool = False,
    env_id: int | None = None,
    num_tasks: int | None = None,
    terminate_on_success: bool = False,
    vector_strategy: Literal["sync", "async"] = "sync",
    task_select: Literal["random", "pseudorandom"] = "random",
) -> gym.Env | gym.vector.VectorEnv:
    benchmark: Benchmark
    if name in ALL_V3_ENVIRONMENTS.keys():
        benchmark = MT1(name, seed=seed)
        tasks = [task for task in benchmark.train_tasks]
        return _init_each_env(
            env_cls=benchmark.train_classes[name],
            tasks=tasks,
            seed=seed,
            max_episode_steps=max_episode_steps,
            use_one_hot=use_one_hot,
            env_id=env_id,
            num_tasks=num_tasks or 1,
            terminate_on_success=terminate_on_success,
        )
    elif name == "MT10" or name == "MT50":
        benchmark = globals()[name](seed=seed)
        vectorizer: type[gym.vector.VectorEnv] = getattr(
            gym.vector, f"{vector_strategy.capitalize()}VectorEnv"
        )
        default_num_tasks = 10 if name == "MT10" else 50
        print(use_one_hot)
        return vectorizer(  # type: ignore
            [
                partial(
                    _init_each_env,
                    env_cls=env_cls,
                    tasks=[
                        task for task in benchmark.train_tasks if task.env_name == name
                    ],
                    seed=seed,
                    max_episode_steps=max_episode_steps,
                    use_one_hot=use_one_hot,
                    env_id=env_id,
                    num_tasks=num_tasks or default_num_tasks,
                    terminate_on_success=terminate_on_success,
                    task_select=task_select,
                )
                for env_id, (name, env_cls) in enumerate(
                    benchmark.train_classes.items()
                )
            ]
        )
    else:
        raise ValueError(
            "Invalid MT env name. Must either be a valid Metaworld task name (e.g. 'reach-v3'), 'MT10' or 'MT50'."
        )


def _make_ml_envs_inner(
    benchmark: Benchmark,
    meta_batch_size: int,
    seed: int | None = None,
    total_tasks_per_cls: int | None = None,
    max_episode_steps: int | None = None,
    split: Literal["train", "test"] = "train",
    terminate_on_success: bool = False,
    task_select: Literal["random", "pseudorandom"] = "pseudorandom",
    vector_strategy: Literal["sync", "async"] = "sync",
):
    all_classes = (
        benchmark.train_classes if split == "train" else benchmark.test_classes
    )
    all_tasks = benchmark.train_tasks if split == "train" else benchmark.test_tasks
    assert (
        meta_batch_size % len(all_classes) == 0
    ), "meta_batch_size must be divisible by envs_per_task"
    tasks_per_env = meta_batch_size // len(all_classes)

    env_tuples = []
    for env_name, env_cls in all_classes.items():
        tasks = [task for task in all_tasks if task.env_name == env_name]
        if total_tasks_per_cls is not None:
            tasks = tasks[:total_tasks_per_cls]
        subenv_tasks = [tasks[i::tasks_per_env] for i in range(0, tasks_per_env)]
        for tasks_for_subenv in subenv_tasks:
            assert (
                len(tasks_for_subenv) == len(tasks) // tasks_per_env
            ), f"Invalid division of subtasks, expected {len(tasks) // tasks_per_env} got {len(tasks_for_subenv)}"
            env_tuples.append((env_cls, tasks_for_subenv))

    vectorizer: type[gym.vector.VectorEnv] = getattr(
        gym.vector, f"{vector_strategy.capitalize()}VectorEnv"
    )
    return vectorizer(  # type: ignore
        [
            partial(
                _init_each_env,
                env_cls=env_cls,
                tasks=tasks,
                seed=seed,
                max_episode_steps=max_episode_steps,
                terminate_on_success=terminate_on_success,
                task_select=task_select,
            )
            for env_cls, tasks in env_tuples
        ]
    )


def make_ml_envs(
    name: str,
    seed: int | None = None,
    meta_batch_size: int = 20,
    total_tasks_per_cls: int | None = None,
    max_episode_steps: int | None = None,
    split: Literal["train", "test"] = "train",
    terminate_on_success: bool = False,
    task_select: Literal["random", "pseudorandom"] = "pseudorandom",
    vector_strategy: Literal["sync", "async"] = "sync",
) -> gym.vector.VectorEnv:
    benchmark: Benchmark
    if name in ALL_V3_ENVIRONMENTS.keys():
        benchmark = ML1(name, seed=seed)
    elif name == "ML10" or name == "ML45":
        benchmark = globals()[name](seed=seed)
    else:
        raise ValueError(
            "Invalid ML env name. Must either be a valid Metaworld task name (e.g. 'reach-v3'), 'ML10' or 'ML45'."
        )
    return _make_ml_envs_inner(
        benchmark,
        meta_batch_size=meta_batch_size,
        seed=seed,
        total_tasks_per_cls=total_tasks_per_cls,
        max_episode_steps=max_episode_steps,
        split=split,
        terminate_on_success=terminate_on_success,
        task_select=task_select,
        vector_strategy=vector_strategy,
    )


make_ml_envs_train = partial(
    make_ml_envs,
    terminate_on_success=False,
    task_select="pseudorandom",
    split="train",
)
make_ml_envs_test = partial(
    make_ml_envs, terminate_on_success=True, task_select="pseudorandom", split="test"
)


def register_mw_envs() -> None:
    def _mt_bench_vector_entry_point(
        mt_bench: str,
        vector_strategy: Literal["sync", "async"],
        seed=None,
        use_one_hot=False,
        num_envs=None,
        *args,
        **lamb_kwargs,
    ):
        return make_mt_envs(  # type: ignore
            mt_bench,
            seed=seed,
            use_one_hot=use_one_hot,
            vector_strategy=vector_strategy,  # type: ignore
            *args,
            **lamb_kwargs,
        )

    def _ml_bench_vector_entry_point(
        ml_bench: str,
        split: str,
        vector_strategy: Literal["sync", "async"],
        seed: int | None = None,
        meta_batch_size: int = 20,
        num_envs=None,
        *args,
        **lamb_kwargs,
    ):
        env_generator = make_ml_envs_train if split == "train" else make_ml_envs_test
        return env_generator(
            ml_bench,
            seed=seed,
            meta_batch_size=meta_batch_size,
            vector_strategy=vector_strategy,
            *args,
            **lamb_kwargs,
        )

    register(
        id="Meta-World/MT1",
        entry_point=lambda env_name, seed=None, vector_strategy="sync": _mt_bench_vector_entry_point(
            env_name, vector_strategy, seed
        ),
        kwargs={},
    )

    for split in ["train", "test"]:
        register(
            id=f"Meta-World/ML1-{split}",
            vector_entry_point=lambda env_name, vector_strategy="sync", seed=None, *args, **kwargs: _ml_bench_vector_entry_point(
                env_name,  # positional arguments
                split,
                vector_strategy,
                seed,
                *args,
                **kwargs,
            ),
            kwargs={},
        )

    register(
        id="Meta-World/goal_hidden",
        entry_point=lambda env_name, seed: ALL_V3_ENVIRONMENTS_GOAL_HIDDEN[env_name](  # type: ignore
            seed=seed
        ),
        kwargs={},
    )

    register(
        id="Meta-World/goal_observable",
        entry_point=lambda env_name, seed: ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[env_name](  # type: ignore
            seed=seed
        ),
        kwargs={},
    )

    for mt_bench in ["MT10", "MT50"]:
        register(
            id=f"Meta-World/{mt_bench}",
            vector_entry_point=lambda vector_strategy="sync", seed=None, use_one_hot=False, *args, _mt_bench=mt_bench, **kwargs: _mt_bench_vector_entry_point(
                _mt_bench,  # positional arguments
                vector_strategy,
                seed,
                use_one_hot,
                *args,
                **kwargs,
            ),
            kwargs={},
        )

    for ml_bench in ["ML10", "ML45"]:
        for split in ["train", "test"]:
            register(
                id=f"Meta-World/{ml_bench}-{split}",  # Fixed f-string
                vector_entry_point=lambda vector_strategy="sync", seed=None, *args, _ml_bench=ml_bench, _split=split, **kwargs: _ml_bench_vector_entry_point(
                    _ml_bench,
                    _split,
                    vector_strategy,
                    seed,
                    *args,
                    **kwargs,
                ),
                kwargs={},
            )

    def _custom_mt_vector_entry_point(
        vector_strategy: str,
        envs_list: list[str],
        seed=None,
        use_one_hot: bool = False,
        num_envs=None,
        *args,
        **lamb_kwargs,
    ):
        vectorizer: type[gym.vector.VectorEnv] = getattr(
            gym.vector, f"{vector_strategy.capitalize()}VectorEnv"
        )
        return (
            vectorizer(  # type: ignore
                [
                    partial(  # type: ignore
                        make_mt_envs,
                        env_name,
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
        )

    register(
        id="Meta-World/custom-mt-envs",
        vector_entry_point=lambda vector_strategy, envs_list, seed=None, use_one_hot=False, num_envs=None: _custom_mt_vector_entry_point(
            vector_strategy, envs_list, seed, use_one_hot, num_envs
        ),
        kwargs={},
    )

    def _custom_ml_vector_entry_point(
        vector_strategy: str,
        train_envs: list[str],
        test_envs: list[str],
        meta_batch_size: int = 20,
        seed=None,
        num_envs=None,
        *args,
        **lamb_kwargs,
    ):
        return _make_ml_envs_inner(  # type: ignore
            CustomML(train_envs, test_envs, seed=seed),
            meta_batch_size=meta_batch_size,
            vector_strategy=vector_strategy,  # type: ignore
            *args,
            **lamb_kwargs,
        )

    register(
        id="Meta-World/custom-ml-envs",
        vector_entry_point=lambda vector_strategy, train_envs, test_envs, meta_batch_size=20, seed=None, num_envs=None: _custom_ml_vector_entry_point(
            vector_strategy, train_envs, test_envs, meta_batch_size, seed, num_envs
        ),
        kwargs={},
    )


register_mw_envs()
__all__: list[str] = []

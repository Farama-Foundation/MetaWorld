"""Proposal for a simple, understandable MetaWorld API."""
import abc
import pickle
from collections import OrderedDict
from typing import List, NamedTuple, Type

import numpy as np

import metaworld.envs.mujoco.env_dict as _env_dict

EnvName = str


class Task(NamedTuple):
    """All data necessary to describe a single MDP.

    Should be passed into a MetaWorldEnv's set_task method.
    """

    env_name: EnvName
    data: bytes  # Contains env parameters like random_init and *a* goal


class MetaWorldEnv:
    """Environment that requires a task before use.

    Takes no arguments to its constructor, and raises an exception if used
    before `set_task` is called.
    """

    def set_task(self, task: Task) -> None:
        """Set the task.

        Raises:
            ValueError: If task.env_name is different from the current task.

        """


class Benchmark(abc.ABC):
    """A Benchmark.

    When used to evaluate an algorithm, only a single instance should be used.
    """

    @abc.abstractmethod
    def __init__(self):
        pass

    @property
    def train_classes(self) -> "OrderedDict[EnvName, Type]":
        """Get all of the environment classes used for training."""
        return self._train_classes

    @property
    def test_classes(self) -> "OrderedDict[EnvName, Type]":
        """Get all of the environment classes used for testing."""
        return self._test_classes

    @property
    def train_tasks(self) -> List[Task]:
        """Get all of the training tasks for this benchmark."""
        return self._train_tasks

    @property
    def test_tasks(self) -> List[Task]:
        """Get all of the test tasks for this benchmark."""
        return self._test_tasks


_ML_OVERRIDE = dict(partially_observable=True)
_MT_OVERRIDE = dict(partially_observable=False)

_N_GOALS = 50


def _encode_task(env_name, data):
    return Task(env_name=env_name, data=pickle.dumps(data))


def _make_tasks(classes, args_kwargs, kwargs_override, seed=None):
    if seed is not None:
        st0 = np.random.get_state()
        np.random.seed(seed)
    tasks = []
    for (env_name, args) in args_kwargs.items():
        assert len(args['args']) == 0
        env = classes[env_name]()
        env._freeze_rand_vec = False
        env._set_task_called = True
        rand_vecs = []
        kwargs = args["kwargs"].copy()
        del kwargs["task_id"]
        env._set_task_inner(**kwargs)
        for _ in range(_N_GOALS):
            print('reset')
            env.reset()
            rand_vecs.append(env._last_rand_vec)
        unique_task_rand_vecs = np.unique(np.array(rand_vecs), axis=0)
        assert unique_task_rand_vecs.shape[0] == _N_GOALS, unique_task_rand_vecs.shape[0]
        env.close()
        for rand_vec in rand_vecs:
            kwargs = args['kwargs'].copy()
            del kwargs['task_id']
            kwargs.update(dict(rand_vec=rand_vec, env_cls=env))
            kwargs.update(kwargs_override)
            tasks.append(_encode_task(env_name, kwargs))
        del env
    if seed is not None:
        np.random.set_state(st0)
    return tasks



def _ml1_env_names():
    tasks = list(_env_dict.ML1_V2["train"])
    assert len(tasks) == 50
    return tasks


class ML1(Benchmark):
    ENV_NAMES = _ml1_env_names()
    def __init__(self, env_name, seed=None):
        super().__init__()
        if env_name not in _env_dict.ALL_V2_ENVIRONMENTS:
            raise ValueError(f"{env_name} is not a V2 environment")
        train_cls = _env_dict.ML1_TRAIN_TEST_ENVS[env_name]['train']
        test_cls = _env_dict.ML1_TRAIN_TEST_ENVS[env_name]['test']

        self._train_classes = OrderedDict([(env_name, train_cls)])
        self._test_classes = OrderedDict([(env_name, test_cls)])
        args_kwargs = _env_dict.ML1_args_kwargs[env_name]

        self._train_tasks = _make_tasks(self._train_classes, {env_name: args_kwargs}, _ML_OVERRIDE, seed=(seed if seed is not None else None))
        tasks_for_cls = None
        for cls in self._train_classes:
            tasks_for_cls = [task for task in self._train_tasks if task.env_name == cls]
            assert len(tasks_for_cls) == _N_GOALS
            self._train_classes[cls].tasks = tasks_for_cls
            self._train_classes[cls].classes = train_cls
            self._train_classes[cls].classes_kwargs = args_kwargs
            self._train_classes[cls]._freeze_rand_vec = True
        self._test_tasks = _make_tasks(self._test_classes, {env_name: args_kwargs}, _ML_OVERRIDE, seed=(seed + 1 if seed is not None else None))
        test_tasks_for_cls = None
        for cls in self._test_classes:
            test_tasks_for_cls = [task for task in self._test_tasks if task.env_name == cls]
            assert len(test_tasks_for_cls) == _N_GOALS
            self._test_classes[cls].tasks = test_tasks_for_cls
            self._test_classes[cls].classes = test_cls
            self._test_classes[cls].classes_kwargs = args_kwargs
            self._test_classes[cls]._freeze_rand_vec = True


class MT1(Benchmark):
    ENV_NAMES = _ml1_env_names()

    def __init__(self, env_name, seed=None):
        super().__init__()
        if env_name not in _env_dict.ALL_V2_ENVIRONMENTS:
            raise ValueError(f"{env_name} is not a V2 environment")
        cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
        self._train_classes = OrderedDict([(env_name, cls())])
        self._test_classes = OrderedDict([(env_name, cls())])
        args_kwargs = _env_dict.ML1_args_kwargs[env_name]

        self._train_tasks = _make_tasks(self._train_classes, {env_name: args_kwargs}, _MT_OVERRIDE, seed=seed)
        for cls in self._train_classes:
            tasks_for_cls = [task for task in self._train_tasks if task.env_name == cls]
            assert len(tasks_for_cls) == _N_GOALS
            self._train_classes[cls].tasks = tasks_for_cls
            self._test_classes[cls].tasks = tasks_for_cls
            # because MT envs don't have different test tasks, use the same rand vecs


class ML10(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.ML10_V2["train"]
        self._test_classes = _env_dict.ML10_V2["test"]
        train_kwargs = _env_dict.ml10_train_args_kwargs

        test_kwargs = _env_dict.ml10_test_args_kwargs
        self._train_tasks = _make_tasks(self._train_classes, train_kwargs, _ML_OVERRIDE, seed=seed)
        for cls in self._train_classes:
            tasks_for_cls = [task for task in self._train_tasks if task.env_name == cls]
            assert len(tasks_for_cls) == _N_GOALS
            self._train_tasks.extend(tasks_for_cls)
            self._train_classes[cls].tasks = tasks_for_cls
            self._train_classes[cls].cls = self._train_classes[cls]
            self._train_classes[cls].cls_kwargs = train_kwargs[cls]
            self._train_classes[cls]._freeze_rand_vec = True

        self._test_tasks = _make_tasks(self._test_classes, test_kwargs, _ML_OVERRIDE, seed=seed)
        for cls in self._test_classes:
            tasks_for_cls = [task for task in self._test_tasks if task.env_name == cls]
            assert len(tasks_for_cls) == _N_GOALS
            self._test_tasks.extend(tasks_for_cls)
            self._test_classes[cls].tasks = tasks_for_cls
            self._test_classes[cls].cls = self._test_classes[cls]
            self._test_classes[cls].cls_kwargs = test_kwargs[cls]
            self._test_classes[cls]._freeze_rand_vec = True



class ML45(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.ML45_V2['train']
        self._test_classes = _env_dict.ML45_V2['test']
        self._test_tasks = []
        self._train_tasks = []
        train_kwargs = _env_dict.ml45_train_args_kwargs
        test_kwargs = _env_dict.ml45_test_args_kwargs

        self._train_tasks = _make_tasks(self._train_classes, train_kwargs, _ML_OVERRIDE, seed=seed)
        for cls in self._train_classes:
            tasks_for_cls = [task for task in self._train_tasks if task.env_name == cls]
            assert len(tasks_for_cls) == _N_GOALS
            self._train_tasks.extend(tasks_for_cls)
            self._train_classes[cls].tasks = tasks_for_cls
            self._train_classes[cls].cls = self._train_classes[cls]
            self._train_classes[cls].cls_kwargs = train_kwargs[cls]
            self._train_classes[cls]._freeze_rand_vec = True

        self._test_tasks = _make_tasks(self._test_classes, test_kwargs, _ML_OVERRIDE, seed=seed)

        for cls in self._test_classes:
            tasks_for_cls = [task for task in self._test_tasks if task.env_name == cls]
            assert len(tasks_for_cls) == _N_GOALS, len(tasks_for_cls)
            self._test_tasks.extend(tasks_for_cls)
            self._test_classes[cls].tasks = tasks_for_cls
            self._test_classes[cls].cls = self._test_classes[cls]
            self._test_classes[cls].cls_kwargs = test_kwargs[cls]
            self._test_classes[cls]._freeze_rand_vec = True


class MT10(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.MT10_V2
        self._test_classes = OrderedDict()
        train_kwargs = _env_dict.MT10_V2_ARGS_KWARGS
        self._train_tasks = _make_tasks(self._train_classes, train_kwargs, _MT_OVERRIDE, seed=seed)
        for cls in self._train_classes:
            tasks_for_cls = [task for task in self._train_tasks if task.env_name == cls]
            assert len(tasks_for_cls) == _N_GOALS
            self._train_tasks.extend(tasks_for_cls)
            self._train_classes[cls].tasks = tasks_for_cls
            self._train_classes[cls].cls = self._train_classes[cls]
            self._train_classes[cls].cls_kwargs = train_kwargs[cls]
            self._train_classes[cls]._freeze_rand_vec = True
        self._test_tasks = []
        self._test_classes = []


class MT50(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.MT50_V2
        self._train_tasks = []
        self._test_classes = OrderedDict()
        train_kwargs = _env_dict.MT50_V2_ARGS_KWARGS
        self._train_tasks = _make_tasks(self._train_classes, train_kwargs, _MT_OVERRIDE, seed=seed)
        for cls in self._train_classes:
            tasks_for_cls = [task for task in self._train_tasks if task.env_name == cls]
            assert len(tasks_for_cls) == _N_GOALS
            self._train_tasks.extend(tasks_for_cls)
            self._train_classes[cls].tasks = tasks_for_cls
            self._train_classes[cls].cls = self._train_classes[cls]
            self._train_classes[cls].cls_kwargs = train_kwargs[cls]
            self._train_classes[cls]._freeze_rand_vec = True

        self._test_tasks = []


__all__ = ["ML1", "MT1", "ML10", "MT10", "ML45", "MT50"]

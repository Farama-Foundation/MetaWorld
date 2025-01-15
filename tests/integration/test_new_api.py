import pickle

import numpy as np
import pytest

import metaworld
from metaworld import ML1, ML10, ML45, MT10, MT50
from tests.helpers import step_env

STEPS = 3


@pytest.mark.parametrize("env_name", ML1.ENV_NAMES)
def test_all_ml1(env_name):
    ml1 = ML1(env_name)
    train_env_instances = {
        env_name: env_cls() for (env_name, env_cls) in ml1.train_classes.items()
    }
    train_env_rand_vecs = check_tasks_unique(ml1.train_tasks, ml1._train_classes.keys())
    for task in ml1.train_tasks:
        env = train_env_instances[task.env_name]
        env.set_task(task)
        env.reset()
        old_obj_init = env.obj_init_pos
        old_target_pos = env._target_pos
        step_env(env, max_path_length=STEPS, render=False)
        assert np.all(np.allclose(old_obj_init, env.obj_init_pos))
        assert np.all(np.allclose(old_target_pos, env._target_pos))
    for env in train_env_instances.values():
        env.close()
    del train_env_instances

    test_env_instances = {
        env_name: env_cls() for (env_name, env_cls) in ml1.test_classes.items()
    }
    test_env_rand_vecs = check_tasks_unique(ml1.test_tasks, ml1._test_classes.keys())
    for task in ml1.test_tasks:
        env = test_env_instances[task.env_name]
        env.set_task(task)
        env.reset()
        old_obj_init = env.obj_init_pos
        old_target_pos = env._target_pos
        step_env(env, max_path_length=STEPS, render=False)
        assert np.all(np.allclose(old_obj_init, env.obj_init_pos))
        assert np.all(np.allclose(old_target_pos, env._target_pos))
    for env in test_env_instances.values():
        env.close()
    train_test_rand_vecs = set()
    for rand_vecs in train_env_rand_vecs.values():
        for rand_vec in rand_vecs:
            train_test_rand_vecs.add(tuple(rand_vec))
    for rand_vecs in test_env_rand_vecs.values():
        for rand_vec in rand_vecs:
            train_test_rand_vecs.add(tuple(rand_vec))
    assert (
        len(train_test_rand_vecs)
        == (len(ml1.test_classes.keys()) + len(ml1.train_classes.keys()))
        * metaworld._N_GOALS
    )
    del test_env_instances


def test_all_ml10():
    ml10 = ML10()
    train_env_instances = {
        env_name: env_cls() for (env_name, env_cls) in ml10.train_classes.items()
    }
    train_env_rand_vecs = check_tasks_unique(
        ml10.train_tasks, ml10._train_classes.keys()
    )
    for task in ml10.train_tasks:
        env = train_env_instances[task.env_name]
        env.set_task(task)
        env.reset()
        old_obj_init = env.obj_init_pos
        old_target_pos = env._target_pos
        step_env(env, max_path_length=STEPS, render=False)
        assert np.all(np.allclose(old_obj_init, env.obj_init_pos))
        assert np.all(np.allclose(old_target_pos, env._target_pos))
        step_env(env, max_path_length=STEPS, render=False)
    for env in train_env_instances.values():
        env.close()
    del train_env_instances

    test_env_instances = {
        env_name: env_cls() for (env_name, env_cls) in ml10.test_classes.items()
    }
    test_env_rand_vecs = check_tasks_unique(ml10.test_tasks, ml10._test_classes.keys())
    for task in ml10.test_tasks:
        env = test_env_instances[task.env_name]
        env.set_task(task)
        env.reset()
        old_obj_init = env.obj_init_pos
        old_target_pos = env._target_pos
        step_env(env, max_path_length=STEPS, render=False)
        assert np.all(np.allclose(old_obj_init, env.obj_init_pos))
        assert np.all(np.allclose(old_target_pos, env._target_pos))
        step_env(env, max_path_length=STEPS, render=False)
    for env in test_env_instances.values():
        env.close()
    train_test_rand_vecs = set()
    for rand_vecs in train_env_rand_vecs.values():
        for rand_vec in rand_vecs:
            train_test_rand_vecs.add(tuple(rand_vec))
    for rand_vecs in test_env_rand_vecs.values():
        for rand_vec in rand_vecs:
            train_test_rand_vecs.add(tuple(rand_vec))
    assert (
        len(train_test_rand_vecs)
        == (len(ml10.test_classes.keys()) + len(ml10.train_classes.keys()))
        * metaworld._N_GOALS
    )
    del test_env_instances


def test_all_ml45():
    ml45 = ML45()
    train_env_instances = {
        env_name: env_cls() for (env_name, env_cls) in ml45.train_classes.items()
    }
    train_env_rand_vecs = check_tasks_unique(
        ml45.train_tasks, ml45._train_classes.keys()
    )
    for task in ml45.train_tasks:
        env = train_env_instances[task.env_name]
        env.set_task(task)
        obs, info = env.reset()
        old_obj_init = env.obj_init_pos
        old_target_pos = env._target_pos
        step_env(env, max_path_length=STEPS, render=False)
        assert np.all(np.allclose(old_obj_init, env.obj_init_pos))
        assert np.all(np.allclose(old_target_pos, env._target_pos))
    for env in train_env_instances.values():
        env.close()

    del train_env_instances

    test_env_instances = {
        env_name: env_cls() for (env_name, env_cls) in ml45.test_classes.items()
    }
    test_env_rand_vecs = check_tasks_unique(ml45.test_tasks, ml45._test_classes.keys())
    for task in ml45.test_tasks:
        env = test_env_instances[task.env_name]
        env.set_task(task)
        obs, info = env.reset()
        assert np.all(obs[-3:] == np.array([0, 0, 0]))
        assert env.observation_space.shape == (39,)
        old_obj_init = env.obj_init_pos
        old_target_pos = env._target_pos
        step_env(env, max_path_length=STEPS, render=False)
        assert np.all(np.allclose(old_obj_init, env.obj_init_pos))
        assert np.all(np.allclose(old_target_pos, env._target_pos))
    for env in test_env_instances.values():
        env.close()
    train_test_rand_vecs = set()
    for rand_vecs in train_env_rand_vecs.values():
        for rand_vec in rand_vecs:
            train_test_rand_vecs.add(tuple(rand_vec))
    for rand_vecs in test_env_rand_vecs.values():
        for rand_vec in rand_vecs:
            train_test_rand_vecs.add(tuple(rand_vec))
    assert (
        len(train_test_rand_vecs)
        == (len(ml45.test_classes.keys()) + len(ml45.train_classes.keys()))
        * metaworld._N_GOALS
    )
    del test_env_instances


def test_all_mt10():
    mt10 = MT10()
    train_env_instances = {
        env_name: env_cls() for (env_name, env_cls) in mt10.train_classes.items()
    }
    train_env_rand_vecs = check_tasks_unique(
        mt10.train_tasks, mt10._train_classes.keys()
    )
    for task in mt10.train_tasks:
        env = train_env_instances[task.env_name]
        env.set_task(task)
        env.reset()
        old_obj_init = env.obj_init_pos
        old_target_pos = env._target_pos
        step_env(env, max_path_length=STEPS, render=False)
        assert np.all(np.allclose(old_obj_init, env.obj_init_pos))
        assert np.all(np.allclose(old_target_pos, env._target_pos))
    for env in train_env_instances.values():
        env.close()
    del train_env_instances

    assert len(mt10.test_classes) == 0
    assert len(mt10.test_tasks) == 0
    train_test_rand_vecs = set()
    for rand_vecs in train_env_rand_vecs.values():
        for rand_vec in rand_vecs:
            train_test_rand_vecs.add(tuple(rand_vec))
    assert len(train_test_rand_vecs) == 10 * 50


def test_all_mt50():
    mt50 = MT50()
    train_env_instances = {
        env_name: env_cls() for (env_name, env_cls) in mt50.train_classes.items()
    }
    train_env_rand_vecs = check_tasks_unique(
        mt50.train_tasks, mt50._train_classes.keys()
    )
    for task in mt50.train_tasks:
        env = train_env_instances[task.env_name]
        env.set_task(task)
        obs, info = env.reset()
        assert np.any(obs[-3:] != np.array([0, 0, 0]))
        assert env.observation_space.shape == (39,)
        old_obj_init = env.obj_init_pos
        old_target_pos = env._target_pos
        step_env(env, max_path_length=STEPS, render=False)
        assert np.all(np.allclose(old_obj_init, env.obj_init_pos))
        assert np.all(np.allclose(old_target_pos, env._target_pos))
    # only needs to be done for 50 environments once
    check_target_poss_unique(train_env_instances, train_env_rand_vecs)
    for env in train_env_instances.values():
        env.close()
    del train_env_instances

    assert len(mt50.test_classes) == 0
    assert len(mt50.test_tasks) == 0
    train_test_rand_vecs = set()
    for rand_vecs in train_env_rand_vecs.values():
        for rand_vec in rand_vecs:
            train_test_rand_vecs.add(tuple(rand_vec))
    assert len(train_test_rand_vecs) == 50 * 50


def check_tasks_unique(tasks, env_names):
    """Verify that all the rand_vecs that are sampled are unique."""
    env_to_rand_vecs = {}
    for env_name in env_names:
        env_to_rand_vecs[env_name] = np.array(
            [
                pickle.loads(task.data)["rand_vec"]
                for task in tasks
                if (task.env_name == env_name)
            ]
        )
        unique_task_rand_vecs = np.unique(np.array(env_to_rand_vecs[env_name]), axis=0)
        assert unique_task_rand_vecs.shape[0] == metaworld._N_GOALS
    return env_to_rand_vecs


def check_target_poss_unique(env_instances, env_rand_vecs):
    """Verify that all the state_goals are unique for the different rand_vecs that are sampled.

    Note: The following envs randomize object initial position but not state_goal.
    ['hammer-v3', 'sweep-into-v3', 'bin-picking-v3', 'basketball-v3']

    """
    for env_name, rand_vecs in env_rand_vecs.items():
        if env_name in {
            "hammer-v3",
            "sweep-into-v3",
            "bin-picking-v3",
            "basketball-v3",
        }:
            continue
        env = env_instances[env_name]
        state_goals = []
        for rand_vec in rand_vecs:
            env._last_rand_vec = rand_vec
            env.reset()
            state_goals.append(env._target_pos)
        state_goals = np.array(state_goals)
        unique_target_poss = np.unique(state_goals, axis=0)
        assert (
            unique_target_poss.shape[0] == metaworld._N_GOALS == len(rand_vecs)
        ), env_name


def test_identical_environments():
    def helper(env, env_2):
        for i in range(len(env.train_tasks)):
            rand_vec_1 = pickle.loads(env.train_tasks[i].data)["rand_vec"]
            rand_vec_2 = pickle.loads(env_2.train_tasks[i].data)["rand_vec"]
            np.testing.assert_equal(rand_vec_1, rand_vec_2)

    def helper_neq(env, env_2):
        for i in range(len(env.train_tasks)):
            rand_vec_1 = pickle.loads(env.train_tasks[i].data)["rand_vec"]
            rand_vec_2 = pickle.loads(env_2.train_tasks[i].data)["rand_vec"]
            assert not (rand_vec_1 == rand_vec_2).all()

    # testing MT1
    mt1_1 = metaworld.MT1("sweep-into-v3", seed=10)
    mt1_2 = metaworld.MT1("sweep-into-v3", seed=10)
    helper(mt1_1, mt1_2)

    # testing ML1
    ml1_1 = metaworld.ML1("sweep-into-v3", seed=10)
    ml1_2 = metaworld.ML1("sweep-into-v3", seed=10)
    helper(ml1_1, ml1_2)

    # testing MT10
    mt10_1 = metaworld.MT10(seed=10)
    mt10_2 = metaworld.MT10(seed=10)
    helper(mt10_1, mt10_2)

    # testing ML10
    ml10_1 = metaworld.ML10(seed=10)
    ml10_2 = metaworld.ML10(seed=10)
    helper(ml10_1, ml10_2)

    # testing ML45
    ml45_1 = metaworld.ML45(seed=10)
    ml45_2 = metaworld.ML45(seed=10)
    helper(ml45_1, ml45_2)

    # testing MT50
    mt50_1 = metaworld.MT50(seed=10)
    mt50_2 = metaworld.MT50(seed=10)
    helper(mt50_1, mt50_2)

    # test that 2 benchmarks with different seeds have different goals
    mt50_3 = metaworld.MT50(seed=50)
    helper_neq(mt50_1, mt50_3)

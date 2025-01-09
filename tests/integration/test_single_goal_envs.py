import numpy as np

from metaworld.env_dict import (
    ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
    ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
)
from tests.helpers import step_env


def test_hidden_goal_envs():
    for env_key, env_cls in ALL_V2_ENVIRONMENTS_GOAL_HIDDEN.items():
        assert "goal-hidden" in env_key
        assert "GoalHidden" in env_cls.__name__
        state_before = np.random.get_state()
        env = env_cls(seed=5)
        env2 = env_cls(seed=5)
        step_env(env, max_path_length=3, iterations=3, render=False)

        first_target = env._target_pos
        env.reset()
        second_target = env._target_pos

        assert (first_target == second_target).all()
        env.reset()
        env2.reset()
        assert (env._target_pos == env2._target_pos).all()
        state_after = np.random.get_state()
        for idx, (state_before_idx, state_after_idx) in enumerate(
            zip(state_before, state_after)
        ):
            if idx == 1:
                assert (state_before_idx == state_after_idx).all()
            else:
                assert state_before_idx == state_after_idx


def test_observable_goal_envs():
    for env_key, env_cls in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.items():
        assert "goal-observable" in env_key
        assert "GoalObservable" in env_cls.__name__
        state_before = np.random.get_state()
        env = env_cls(seed=10)
        env2 = env_cls(seed=10)
        step_env(env, max_path_length=3, iterations=3, render=False)

        first_target = env._target_pos
        env.reset()
        second_target = env._target_pos

        assert (first_target == second_target).all()
        env.reset()
        env2.reset()
        assert (env._target_pos == env2._target_pos).all()
        state_after = np.random.get_state()
        for idx, (state_before_idx, state_after_idx) in enumerate(
            zip(state_before, state_after)
        ):
            if idx == 1:
                assert (state_before_idx == state_after_idx).all()
            else:
                assert state_before_idx == state_after_idx


def test_seeding_observable():
    door_open_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[
        "door-open-v2-goal-observable"
    ]

    env1 = door_open_goal_observable_cls(seed=5)
    env2 = door_open_goal_observable_cls(seed=5)

    env1.reset()  # Reset environment
    env2.reset()
    a1 = env1.action_space.sample()  # Sample an action
    a2 = env2.action_space.sample()
    next_obs1, _, _, _, _ = env1.step(
        a1
    )  # Step the environoment with the sampled random action
    next_obs2, _, _, _, _ = env2.step(a2)
    assert (
        next_obs1[-3:] == next_obs2[-3:]
    ).all()  # 2 envs initialized with the same seed will have the same goal
    assert not (
        next_obs2[-3:] == np.zeros(3)
    ).all()  # The env's are goal observable, meaning the goal is not zero'd out

    env3 = door_open_goal_observable_cls(
        seed=10
    )  # Construct an environment with a different seed
    env1.reset()  # Reset environment
    env3.reset()
    a1 = env1.action_space.sample()  # Sample an action
    a3 = env3.action_space.sample()
    next_obs1, _, _, _, _ = env1.step(
        a1
    )  # Step the environoment with the sampled random action
    next_obs3, _, _, _, _ = env3.step(a3)

    assert not (
        next_obs1[-3:] == next_obs3[-3:]
    ).all()  # 2 envs initialized with different seeds will have different goals
    assert not (
        next_obs1[-3:] == np.zeros(3)
    ).all()  # The env's are goal observable, meaning the goal is not zero'd out


def test_seeding_hidden():
    door_open_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[
        "door-open-v2-goal-hidden"
    ]

    env1 = door_open_goal_hidden_cls(seed=5)
    env2 = door_open_goal_hidden_cls(seed=5)

    env1.reset()  # Reset environment
    env2.reset()
    a1 = env1.action_space.sample()  # Sample an action
    a2 = env2.action_space.sample()
    next_obs1, _, _, _, _ = env1.step(
        a1
    )  # Step the environoment with the sampled random action
    next_obs2, _, _, _, _ = env2.step(a2)
    assert (
        env1._target_pos == env2._target_pos
    ).all()  # 2 envs initialized with the same seed will have the same goal
    assert (next_obs2[-3:] == np.zeros(3)).all() and (
        next_obs1[-3] == np.zeros(3)
    ).all()  # The env's are goal observable, meaning the goal is zero'd out

    env3 = door_open_goal_hidden_cls(
        seed=10
    )  # Construct an environment with a different seed
    env1.reset()  # Reset environment
    env3.reset()
    a1 = env1.action_space.sample()  # Sample an action
    a3 = env3.action_space.sample()
    next_obs1, _, _, _, _ = env1.step(
        a1
    )  # Step the environoment with the sampled random action
    next_obs3, _, _, _, _ = env3.step(a3)

    assert not (
        env1._target_pos[-3:] == env3._target_pos[-3:]
    ).all()  # 2 envs initialized with different seeds will have different goals
    assert (next_obs1[-3:] == np.zeros(3)).all() and (
        np.zeros(3) == next_obs3[-3:]
    ).all()  # The env's are goal observable, meaning the goal is zero'd out

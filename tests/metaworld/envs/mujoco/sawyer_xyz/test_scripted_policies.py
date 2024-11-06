import random

import numpy as np
import pytest

from metaworld import MT1
from metaworld.policies import ENV_POLICY_MAP


@pytest.mark.parametrize("env_name", MT1.ENV_NAMES)
def test_policy(env_name):
    SEED = 42
    random.seed(SEED)
    np.random.random(SEED)

    mt1 = MT1(env_name, seed=SEED)
    env = mt1.train_classes[env_name]()
    env.seed(SEED)
    p = ENV_POLICY_MAP[env_name]()
    completed = 0
    for task in mt1.train_tasks:
        env.set_task(task)
        obs, info = env.reset()
        done = False
        count = 0
        while count < 500 and not done:
            count += 1
            a = p.get_action(obs)
            next_obs, _, trunc, termn, info = env.step(a)
            done = trunc or termn
            obs = next_obs
            if int(info["success"]) == 1:
                completed += 1
                break
    assert (float(completed) / 50) >= 0.80

import random

import numpy as np
import pytest

from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE


@pytest.mark.parametrize("env_name", sorted(ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE.keys()))
def test_observations_match(env_name):
    seed = random.randrange(1000)
    env1 = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[env_name](seed=seed)
    env1.seeded_rand_vec = True
    enV3 = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[env_name](seed=seed)
    enV3.seeded_rand_vec = True

    (obs1, _), (obs2, _) = env1.reset(), enV3.reset()
    assert (obs1 == obs2).all()

    for i in range(env1.max_path_length):
        a = np.random.uniform(low=-1, high=-1, size=4)
        obs1, r1, done1, _, _ = env1.step(a)
        obs2, r2, done2, _, _ = enV3.step(a)
        assert (obs1 == obs2).all()
        assert r1 == r2
        assert not done1
        assert not done2

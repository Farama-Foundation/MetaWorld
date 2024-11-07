import numpy as np
import pytest

from metaworld.env_dict import ALL_V3_ENVIRONMENTS
from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, move
from metaworld.sawyer_xyz_env import SawyerXYZEnv


class SawyerRandomReachPolicy(Policy):
    def __init__(self, target):
        self._target = target

    @staticmethod
    def _parse_obs(obs):
        return {"hand_pos": obs[:3], "unused_info": obs[3:]}

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})

        action["delta_pos"] = move(o_d["hand_pos"], to_xyz=self._target, p=25.0)
        action["grab_effort"] = 0.0

        return action.array


def sample_spherical(num_points, radius=1.0):
    """Samples points from the surface of a sphere centered at the origin

    Args:
        num_points (int): number of points to sample
        radius (float): radius of the sphere

    Returns:
        (np.ndarray): points array of shape (num_points, 3)
    """
    points = np.random.randn(3, num_points)
    points /= np.linalg.norm(points, axis=0)
    return points.T * radius


@pytest.mark.parametrize("target", sample_spherical(100, 10.0))
def test_reaching_limit(target):
    env = ALL_V3_ENVIRONMENTS["reach-V3"]()
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True

    policy = SawyerRandomReachPolicy(target)

    env.reset()
    env.reset_model()
    o_prev, info = env.reset()

    for _ in range(env.max_path_length):
        a = policy.get_action(o_prev)
        o = env.step(a)[0]
        if np.linalg.norm(o[:3] - o_prev[:3]) < 0.001:
            break
        o_prev = o

    assert SawyerXYZEnv._HAND_SPACE.contains(o[:3])

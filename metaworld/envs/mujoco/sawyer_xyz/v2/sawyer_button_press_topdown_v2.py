import mujoco
import numpy as np
from gymnasium.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerButtonPressTopdownEnvV2(SawyerXYZEnv):
    def __init__(self, tasks=None, render_mode=None):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.8, 0.115)
        obj_high = (0.1, 0.9, 0.115)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
        )
        if tasks is not None:
            self.tasks = tasks
        self.init_config = {
            "obj_init_pos": np.array([0, 0.8, 0.115], dtype=np.float32),
            "hand_init_pos": np.array([0, 0.4, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0, 0.88, 0.1])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_button_press_topdown.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            near_button,
            button_pressed,
        ) = self.compute_reward(action, obs)
        info = {
            "success": float(obj_to_target <= 0.024),
            "near_object": float(tcp_to_obj <= 0.05),
            "grasp_success": float(tcp_open > 0),
            "grasp_reward": near_button,
            "in_place_reward": button_pressed,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,
        }

        return reward, info

    @property
    def _target_site_config(self):
        return []

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id("btnGeom")

    def _get_pos_objects(self):
        return self.get_body_com("button") + np.array([0.0, 0.0, 0.193])

    def _get_quat_objects(self):
        return self.data.body("button").xquat

    def _set_obj_xyz(self, pos):
        arm_nqpos = self._QPOS_SPACE.low.size
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[arm_nqpos] = pos
        qvel[arm_nqpos] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()
        goal_pos = self._get_state_rand_vec()
        self.obj_init_pos = goal_pos
        self.model.body_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        ] = self.obj_init_pos
        mujoco.mj_forward(self.model, self.data)
        self._target_pos = self._get_site_pos("hole")

        self._obj_to_target_init = abs(
            self._target_pos[2] - self._get_site_pos("buttonStart")[2]
        )
        return self._get_obs()

    def compute_reward(self, action, obs):
        del action
        obj = obs[4:7]
        tcp = self.tcp_center

        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(obj - self.init_tcp)
        obj_to_target = abs(self._target_pos[2] - obj[2])

        tcp_closed = 1 - obs[3]
        near_button = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, 0.01),
            margin=tcp_to_obj_init,
            sigmoid="long_tail",
        )
        button_pressed = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, 0.005),
            margin=self._obj_to_target_init,
            sigmoid="long_tail",
        )

        reward = 5 * reward_utils.hamacher_product(tcp_closed, near_button)
        if tcp_to_obj <= 0.03:
            reward += 5 * button_pressed

        return (reward, tcp_to_obj, obs[3], obj_to_target, near_button, button_pressed)


class TrainButtonPressTopdownv2(SawyerButtonPressTopdownEnvV2):
    tasks = None

    def __init__(self):
        SawyerButtonPressTopdownEnvV2.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)


class TestButtonPressTopdownv2(SawyerButtonPressTopdownEnvV2):
    tasks = None

    def __init__(self):
        SawyerButtonPressTopdownEnvV2.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)

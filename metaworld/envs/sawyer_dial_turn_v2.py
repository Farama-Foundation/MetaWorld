from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.asset_path_utils import full_v2_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict
from metaworld.utils import reward_utils


class SawyerDialTurnEnvV2(SawyerXYZEnv):
    TARGET_RADIUS: float = 0.07

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
    ) -> None:
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.7, 0.0)
        obj_high = (0.1, 0.8, 0.0)
        goal_low = (-0.1, 0.73, 0.0299)
        goal_high = (0.1, 0.83, 0.0301)

        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
        )

        self.init_config: InitConfigDict = {
            "obj_init_pos": np.array([0, 0.7, 0.0]),
            "hand_init_pos": np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0.0, 0.73, 0.08])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)

    @property
    def model_name(self) -> str:
        return full_v2_path_for("sawyer_xyz/sawyer_dial.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        (
            reward,
            tcp_to_obj,
            _,
            target_to_obj,
            object_grasped,
            in_place,
        ) = self.compute_reward(action, obs)

        info = {
            "success": float(target_to_obj <= self.TARGET_RADIUS),
            "near_object": float(tcp_to_obj <= 0.01),
            "grasp_success": 1.0,
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
            "unscaled_reward": reward,
        }

        return reward, info

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        dial_center = self.get_body_com("dial").copy()
        dial_angle_rad = self.data.joint("knob_Joint_1").qpos

        offset = np.array(
            [np.sin(dial_angle_rad).item(), -np.cos(dial_angle_rad).item(), 0.0]
        )
        dial_radius = 0.05

        offset *= dial_radius

        return dial_center + offset

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("dial").xquat

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.prev_obs = self._get_curr_obs_combined_no_goal()

        goal_pos = self._get_state_rand_vec()
        self.obj_init_pos = goal_pos[:3]
        final_pos = goal_pos.copy() + np.array([0, 0.03, 0.03])
        self._target_pos = final_pos
        self.model.body("dial").pos = self.obj_init_pos
        self.dial_push_position = self._get_pos_objects() + np.array([0.05, 0.02, 0.09])
        self.model.site("goal").pos = self._target_pos
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert (
            self._target_pos is not None
        ), "`reset_model()` must be called before `compute_reward()`."
        obj = self._get_pos_objects()
        dial_push_position = self._get_pos_objects() + np.array([0.05, 0.02, 0.09])
        tcp = self.tcp_center
        target = self._target_pos.copy()

        target_to_obj = obj - target
        target_to_obj = float(np.linalg.norm(target_to_obj).item())
        target_to_obj_init = self.dial_push_position - target
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=abs(target_to_obj_init - self.TARGET_RADIUS),
            sigmoid="long_tail",
        )

        dial_reach_radius = 0.005
        tcp_to_obj = float(np.linalg.norm(dial_push_position - tcp).item())
        tcp_to_obj_init = float(
            np.linalg.norm(self.dial_push_position - self.init_tcp).item()
        )
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, dial_reach_radius),
            margin=abs(tcp_to_obj_init - dial_reach_radius),
            sigmoid="gaussian",
        )
        gripper_closed = min(max(0, action[-1]), 1)

        reach = reward_utils.hamacher_product(reach, gripper_closed)
        tcp_opened = 0
        object_grasped = reach

        reward = 10 * reward_utils.hamacher_product(reach, in_place)
        return (
            reward,
            tcp_to_obj,
            tcp_opened,
            target_to_obj,
            object_grasped,
            in_place,
        )

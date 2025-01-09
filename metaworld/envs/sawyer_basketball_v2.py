from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.asset_path_utils import full_v2_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict
from metaworld.utils import reward_utils


class SawyerBasketballEnvV2(SawyerXYZEnv):
    PAD_SUCCESS_MARGIN: float = 0.06
    TARGET_RADIUS: float = 0.08

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
    ) -> None:
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.6, 0.0299)
        obj_high = (0.1, 0.7, 0.0301)
        goal_low = (-0.1, 0.85, 0.0)
        goal_high = (0.1, 0.9 + 1e-7, 0.0)

        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
        )

        self.init_config: InitConfigDict = {
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0, 0.6, 0.03], dtype=np.float32),
            "hand_init_pos": np.array((0, 0.6, 0.2), dtype=np.float32),
        }
        self.goal = np.array([0, 0.9, 0])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
            dtype=np.float64,
        )
        self.goal_space = Box(
            np.array(goal_low) + np.array([0, -0.083, 0.2499]),
            np.array(goal_high) + np.array([0, -0.083, 0.2501]),
            dtype=np.float64,
        )

    @property
    def model_name(self) -> str:
        return full_v2_path_for("sawyer_xyz/sawyer_basketball.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        obj = obs[4:7]
        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            grasp_reward,
            in_place_reward,
        ) = self.compute_reward(action, obs)

        assert self.obj_init_pos is not None
        info = {
            "success": float(obj_to_target <= self.TARGET_RADIUS),
            "near_object": float(tcp_to_obj <= 0.05),
            "grasp_success": float(
                (tcp_open > 0) and (obj[2] - 0.03 > self.obj_init_pos[2])
            ),
            "grasp_reward": grasp_reward,
            "in_place_reward": in_place_reward,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,
        }

        return reward, info

    def _get_id_main_object(self) -> int:
        return self.model.geom_name2id("objGeom")

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.get_body_com("bsktball")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("bsktball").xquat

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self.prev_obs = self._get_curr_obs_combined_no_goal()
        goal_pos = self._get_state_rand_vec()
        basket_pos = goal_pos[3:]
        while np.linalg.norm(goal_pos[:2] - basket_pos[:2]) < 0.15:
            goal_pos = self._get_state_rand_vec()
            basket_pos = goal_pos[3:]
        assert self.obj_init_pos is not None
        self.obj_init_pos = np.concatenate([goal_pos[:2], [self.obj_init_pos[-1]]])
        self.model.body("basket_goal").pos = basket_pos
        self._target_pos = self.data.site("goal").xpos
        self._set_obj_xyz(self.obj_init_pos)
        self.model.site("goal").pos = self._target_pos
        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert (
            self._target_pos is not None and self.obj_init_pos is not None
        ), "`reset_model()` must be called before `compute_reward()`."

        obj = obs[4:7]
        # Force target to be slightly above basketball hoop
        target = self._target_pos.copy()
        target[2] = 0.3

        # Emphasize Z error
        scale = np.array([1.0, 1.0, 2.0])
        target_to_obj = (obj - target) * scale
        target_to_obj = float(np.linalg.norm(target_to_obj))
        target_to_obj_init = (self.obj_init_pos - target) * scale
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=target_to_obj_init,
            sigmoid="long_tail",
        )
        tcp_opened = float(obs[3])
        tcp_to_obj = float(np.linalg.norm(obj - self.tcp_center))

        object_grasped = self._gripper_caging_reward(
            action,
            obj,
            object_reach_radius=0.01,
            obj_radius=0.025,
            pad_success_thresh=0.06,
            xz_thresh=0.005,
            high_density=True,
        )
        if (
            tcp_to_obj < 0.035
            and tcp_opened > 0
            and obj[2] - 0.01 > self.obj_init_pos[2]
        ):
            object_grasped = 1.0
        reward = reward_utils.hamacher_product(object_grasped, in_place)

        if (
            tcp_to_obj < 0.035
            and tcp_opened > 0
            and obj[2] - 0.01 > self.obj_init_pos[2]
        ):
            reward += 1.0 + 5.0 * in_place
        if target_to_obj < self.TARGET_RADIUS:
            reward = 10.0
        return (reward, tcp_to_obj, tcp_opened, target_to_obj, object_grasped, in_place)

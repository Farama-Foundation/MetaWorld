from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.asset_path_utils import full_v2_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import HammerInitConfigDict
from metaworld.utils import reward_utils


class SawyerHammerEnvV2(SawyerXYZEnv):
    HAMMER_HANDLE_LENGTH = 0.14

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
    ) -> None:
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.4, 0.0)
        obj_high = (0.1, 0.5, 0.0)
        goal_low = (0.2399, 0.7399, 0.109)
        goal_high = (0.2401, 0.7401, 0.111)

        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
        )

        self.init_config: HammerInitConfigDict = {
            "hammer_init_pos": np.array([0, 0.5, 0.0]),
            "hand_init_pos": np.array([0, 0.4, 0.2]),
        }
        self.goal = self.init_config["hammer_init_pos"]
        self.hammer_init_pos = self.init_config["hammer_init_pos"]
        self.obj_init_pos = self.hammer_init_pos.copy()
        self.hand_init_pos = self.init_config["hand_init_pos"]
        self.nail_init_pos: npt.NDArray[Any] | None = None

        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)

    @property
    def model_name(self) -> str:
        return full_v2_path_for("sawyer_xyz/sawyer_hammer.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        (
            reward,
            reward_grab,
            reward_ready,
            reward_success,
            success,
        ) = self.compute_reward(action, obs)

        info = {
            "success": float(success),
            "near_object": reward_ready,
            "grasp_success": reward_grab >= 0.5,
            "grasp_reward": reward_grab,
            "in_place_reward": reward_success,
            "obj_to_target": 0,
            "unscaled_reward": reward,
        }

        return reward, info

    def _get_id_main_object(self) -> int:
        return self.model.geom_name2id("HammerHandle")

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return np.hstack(
            (self.get_body_com("hammer").copy(), self.get_body_com("nail_link").copy())
        )

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return np.hstack(
            (self.data.body("hammer").xquat, self.data.body("nail_link").xquat)
        )

    def _set_hammer_xyz(self, pos: npt.NDArray[Any]) -> None:
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()

        # Set position of box & nail (these are not randomized)
        self.model.body("box").pos = np.array([0.24, 0.85, 0.0])
        # Update _target_pos
        self._target_pos = self._get_site_pos("goal")

        # Randomize hammer position
        self.hammer_init_pos = self._get_state_rand_vec()
        self.nail_init_pos = self._get_site_pos("nailHead")
        self.obj_init_pos = self.hammer_init_pos.copy()
        self._set_hammer_xyz(self.hammer_init_pos)
        return self._get_obs()

    @staticmethod
    def _reward_quat(obs: npt.NDArray[np.float64]) -> float:
        # Ideal laid-down wrench has quat [1, 0, 0, 0]
        # Rather than deal with an angle between quaternions, just approximate:
        ideal = np.array([1.0, 0.0, 0.0, 0.0])
        error = float(np.linalg.norm(obs[7:11] - ideal).item())
        return max(1.0 - error / 0.4, 0.0)

    @staticmethod
    def _reward_pos(hammer_head, target_pos):
        pos_error = target_pos - hammer_head

        a = 0.1  # Relative importance of just *trying* to lift the hammer
        b = 0.9  # Relative importance of hitting the nail
        lifted = hammer_head[2] > 0.02
        in_place = a * float(lifted) + b * reward_utils.tolerance(
            np.linalg.norm(pos_error),
            bounds=(0, 0.02),
            margin=0.2,
            sigmoid="long_tail",
        )

        return in_place

    def compute_reward(
        self, actions: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, bool]:
        hand = obs[:3]
        hammer = obs[4:7]
        hammer_head = hammer + np.array([0.16, 0.06, 0.0])
        # `self._gripper_caging_reward` assumes that the target object can be
        # approximated as a sphere. This is not true for the hammer handle, so
        # to avoid re-writing the `self._gripper_caging_reward` we pass in a
        # modified hammer position.
        # This modified position's X value will perfect match the hand's X value
        # as long as it's within a certain threshold
        hammer_threshed = hammer.copy()
        threshold = SawyerHammerEnvV2.HAMMER_HANDLE_LENGTH / 2.0
        if abs(hammer[0] - hand[0]) < threshold:
            hammer_threshed[0] = hand[0]

        reward_quat = SawyerHammerEnvV2._reward_quat(obs)
        reward_grab = self._gripper_caging_reward(
            actions,
            hammer_threshed,
            object_reach_radius=0.01,
            obj_radius=0.015,
            pad_success_thresh=0.02,
            xz_thresh=0.01,
            high_density=True,
        )
        reward_in_place = SawyerHammerEnvV2._reward_pos(hammer_head, self._target_pos)

        reward = (2.0 * reward_grab + 6.0 * reward_in_place) * reward_quat
        # Override reward on success. We check that reward is above a threshold
        # because this env's success metric could be hacked easily
        success = bool(self.data.joint("NailSlideJoint").qpos > 0.09)
        if success and reward > 5.0:
            reward = 10.0

        return (
            reward,
            reward_grab,
            reward_quat,
            reward_in_place,
            success,
        )

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.asset_path_utils import full_v2_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict
from metaworld.utils import reward_utils


class SawyerDoorUnlockEnvV2(SawyerXYZEnv):
    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
    ) -> None:
        hand_low = (-0.5, 0.40, -0.15)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.8, 0.15)
        obj_high = (0.1, 0.85, 0.15)
        goal_low = (0.0, 0.64, 0.2100)
        goal_high = (0.2, 0.7, 0.2111)

        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
        )

        self.init_config: InitConfigDict = {
            "obj_init_pos": np.array([0, 0.85, 0.15]),
            "hand_init_pos": np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0, 0.85, 0.1])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._lock_length = 0.1

        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)

    @property
    def model_name(self) -> str:
        return full_v2_path_for("sawyer_xyz/sawyer_door_lock.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            near_button,
            button_pressed,
        ) = self.compute_reward(action, obs)

        info = {
            "success": float(obj_to_target <= 0.02),
            "near_object": float(tcp_to_obj <= 0.05),
            "grasp_success": float(tcp_open > 0),
            "grasp_reward": near_button,
            "in_place_reward": button_pressed,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,
        }

        return reward, info

    @property
    def _target_site_config(self) -> list[tuple[str, npt.NDArray[Any]]]:
        assert (
            self._target_pos is not None
        ), "`reset_model()` must be called before `_target_site_config`."
        return [
            ("goal_unlock", self._target_pos),
            ("goal_lock", np.array([10.0, 10.0, 10.0])),
        ]

    def _get_id_main_object(self):
        return None

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self._get_site_pos("lockStartUnlock")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("door_link").xquat

    def _set_obj_xyz(self, pos: npt.NDArray[Any]) -> None:
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        qvel[9] = 0
        self.set_state(qpos, qvel)

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self.model.body("door").pos = self._get_state_rand_vec()
        self._set_obj_xyz(np.array(1.5708))

        self.obj_init_pos = self.data.body("lock_link").xpos
        self._target_pos = self.obj_init_pos + np.array([0.1, -0.04, 0.0])

        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert (
            self._target_pos is not None
        ), "`reset_model()` must be called before `compute_reward()`."
        del action
        gripper = obs[:3]
        lock = obs[4:7]

        # Add offset to track gripper's shoulder, rather than fingers
        offset = np.array([0.0, 0.055, 0.07])

        scale = np.array([0.25, 1.0, 0.5])
        shoulder_to_lock = (gripper + offset - lock) * scale
        shoulder_to_lock_init = (self.init_tcp + offset - self.obj_init_pos) * scale

        # This `ready_to_push` reward should be a *hint* for the agent, not an
        # end in itself. Make sure to devalue it compared to the value of
        # actually unlocking the lock
        ready_to_push = reward_utils.tolerance(
            float(np.linalg.norm(shoulder_to_lock)),
            bounds=(0, 0.02),
            margin=np.linalg.norm(shoulder_to_lock_init),
            sigmoid="long_tail",
        )

        obj_to_target = abs(float(self._target_pos[0] - lock[0]))
        pushed = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, 0.005),
            margin=self._lock_length,
            sigmoid="long_tail",
        )

        reward = 2 * ready_to_push + 8 * pushed

        return (
            reward,
            float(np.linalg.norm(shoulder_to_lock)),
            obs[3],
            obj_to_target,
            ready_to_push,
            pushed,
        )

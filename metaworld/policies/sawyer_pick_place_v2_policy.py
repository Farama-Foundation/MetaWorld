import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerPickPlaceV2Policy(Policy):
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            "hand_pos": obs[:3],
            "gripper_distance_apart": obs[3],
            "puck_pos": obs[4:7],
            "puck_rot": obs[7:11],
            "goal_pos": obs[-3:],
            "unused_info_curr_obs": obs[11:18],
            "_prev_obs": obs[18:36],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})

        action["delta_pos"] = move(
            o_d["hand_pos"], to_xyz=self._desired_pos(o_d), p=10.0
        )
        action["grab_effort"] = self._grab_effort(o_d)

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d["hand_pos"]
        pos_puck = o_d["puck_pos"] + np.array([-0.005, 0, 0])
        pos_goal = o_d["goal_pos"]
        gripper_separation = o_d["gripper_distance_apart"]
        # If error in the XY plane is greater than 0.02, place end effector above the puck
        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.02:
            return pos_puck + np.array([0.0, 0.0, 0.1])
        # Once XY error is low enough, drop end effector down on top of puck
        elif abs(pos_curr[2] - pos_puck[2]) > 0.05 and pos_puck[-1] < 0.04:
            return pos_puck + np.array([0.0, 0.0, 0.03])
        # Wait for gripper to close before continuing to move
        elif gripper_separation > 0.73:
            return pos_curr
        # Move to goal
        else:
            return pos_goal

    @staticmethod
    def _grab_effort(o_d):
<<<<<<< HEAD
        pos_curr = o_d["hand_pos"]
        pos_puck = o_d["puck_pos"]
        if np.linalg.norm(pos_curr - pos_puck) < 0.07:
            return 1.0
=======
        pos_curr = o_d['hand_pos']
        pos_puck = o_d['puck_pos']
        if np.linalg.norm(pos_curr - pos_puck) < 0.045:
            return 1.
>>>>>>> 63655f9a8d1b47f289b5bc76c301ee84f35e06ce
        else:
            return 0.0

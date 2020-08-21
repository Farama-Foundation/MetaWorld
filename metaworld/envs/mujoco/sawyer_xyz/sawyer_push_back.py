import numpy as np
from gym.spaces import Box

from metaworld.envs.env_util import get_asset_full_path
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv, _assert_task_is_set


class SawyerPushBackEnv(SawyerXYZEnv):
    def __init__(self):

        goal_low = (-0.1, 0.6, 0.0199)
        goal_high = (0.1, 0.7, 0.0201)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.8, 0.02)
        obj_high = (0.1, 0.85, 0.02)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos':np.array([0, 0.8, 0.02]),
            'obj_init_angle': 0.3,
            'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0., 0.6, 0.02])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.max_path_length = 150

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_push_back.xml')

    @_assert_task_is_set
    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        obs_dict = self._get_obs_dict()
        reward, reachDist, pushDist = self.compute_reward(action, obs_dict)
        self.curr_path_length += 1

        info = {'reachDist': reachDist, 'goalDist': pushDist, 'epRew' : reward, 'pickRew':None, 'success': float(pushDist <= 0.07)}
        info['goal'] = self.goal

        return ob, reward, False, info

    def _get_pos_objects(self):
        return self.data.get_geom_xpos('objGeom')

    def _set_goal_marker(self, goal):
        self.data.site_xpos[self.model.site_name2id('goal')] = (
            goal[:3]
        )

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def adjust_initObjPos(self, orig_init_pos):
        # This is to account for meshes for the geom and object are not aligned
        # If this is not done, the object could be initialized in an extreme position
        diff = self.get_body_com('obj')[:2] - self.data.get_geom_xpos('objGeom')[:2]
        adjustedPos = orig_init_pos[:2] + diff

        # The convention we follow is that body_com[2] is always 0, and geom_pos[2] is the object height
        return [adjustedPos[0], adjustedPos[1],self.data.get_geom_xpos('objGeom')[-1]]

    def reset_model(self):
        self._reset_hand()
        self._state_goal = self.goal.copy()
        self.obj_init_pos = self.adjust_initObjPos(self.init_config['obj_init_pos'])
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.objHeight = self.data.get_geom_xpos('objGeom')[2]

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            self._state_goal = np.concatenate((goal_pos[-3:-1], [self.obj_init_pos[-1]]))
            while np.linalg.norm(goal_pos[:2] - self._state_goal[:2]) < 0.15:
                goal_pos = self._get_state_rand_vec()
                self._state_goal = np.concatenate((goal_pos[-3:-1], [self.obj_init_pos[-1]]))
            self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))

        self._set_goal_marker(self._state_goal)
        self._set_obj_xyz(self.obj_init_pos)
        self.maxPushDist = np.linalg.norm(self.obj_init_pos[:2] - np.array(self._state_goal)[:2])

        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.reachCompleted = False

    def compute_reward(self, actions, obs):
        obs = obs['state_observation']

        objPos = obs[3:6]

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        goal = self._state_goal

        c1 = 1000
        c2 = 0.01
        c3 = 0.001
        assert np.all(goal == self.get_site_pos('goal'))
        reachDist = np.linalg.norm(fingerCOM - objPos)
        pushDist = np.linalg.norm(objPos[:2] - goal[:2])
        reachRew = -reachDist
        reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
        zRew = np.linalg.norm(fingerCOM[-1] - self.hand_init_pos[-1])

        if reachDistxy < 0.05:
            reachRew = -reachDist
        else:
            reachRew =  -reachDistxy - 2*zRew

        # incentive to close fingers when reachDist is small
        if reachDist < 0.05:
            reachRew = -reachDist + max(actions[-1],0)/50

        self.reachCompleted = reachDist < 0.05

        if self.reachCompleted:
            pushRew = 1000*(self.maxPushDist - pushDist) + c1*(np.exp(-(pushDist**2)/c2) + np.exp(-(pushDist**2)/c3))
            pushRew = max(pushRew, 0)
        else:
            pushRew = 0

        reward = reachRew + pushRew

        return [reward, reachDist, pushDist]

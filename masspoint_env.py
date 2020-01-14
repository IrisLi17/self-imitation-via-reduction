import os
import copy
from gym import utils
from masspoint_base import MasspointPushEnv
import gym.envs.robotics.utils as robot_utils
import numpy as np


MODEL_XML_PATH0 = os.path.join(os.path.dirname(__file__), 'assets', 'masspoint', 'single_obstacle.xml')
MODEL_XML_PATH2 = os.path.join(os.path.dirname(__file__), 'assets', 'masspoint', 'single_obstacle2.xml')
MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'masspoint', 'double_obstacle.xml')


class MasspointPushSingleObstacleEnv(MasspointPushEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', random_box=True,
                 random_ratio=1.0, random_pusher=False):
        XML_PATH = MODEL_XML_PATH0
        initial_qpos = {
            'masspoint:slidex': 1.25,
            'masspoint:slidey': 0.75,
            'masspoint:slidez': 0.025,
            # 'object0:slidex': 0.0,
            # 'object0:slidey': 0.0,
            'object0:joint': [1.2, 0.75, 0.025, 1., 0., 0., 0.],
            'object1:joint': [1.4, 0.47, 0.03, 1., 0., 0., 0.],
        }
        self.random_box = random_box
        self.random_ratio = random_ratio
        self.random_pusher = random_pusher
        MasspointPushEnv.__init__(
            self, XML_PATH, n_substeps=10,
            target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, n_object=2)
        utils.EzPickle.__init__(self)
        self.pos_wall0 = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall0')]
        self.size_wall = self.sim.model.geom_size[self.sim.model.geom_name2id('wall0')]
        self.size_obstacle = self.sim.model.geom_size[self.sim.model.geom_name2id('object1')]
        self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]

    def _reset_sim(self):
        # self.sim.set_state(self.initial_state)
        sim_state = copy.deepcopy(self.initial_state)
        # TODO: randomize masspoint pos
        if self.random_pusher:
            masspoint_jointx_i = self.sim.model.get_joint_qpos_addr('masspoint:slidex')
            masspoint_jointy_i = self.sim.model.get_joint_qpos_addr('masspoint:slidey')
            masspoint_pos = self.np_random.uniform([1.15, 0.6], [1.45, 0.9])
            sim_state.qpos[masspoint_jointx_i] = masspoint_pos[0]
            sim_state.qpos[masspoint_jointy_i] = masspoint_pos[1]
        else:
            masspoint_pos = self.initial_masspoint_xpos[:2]

        def config_valid(object_xpos, obstacle1_xpos):
            if np.linalg.norm(object_xpos - masspoint_pos) >= 0.1 \
                    and abs(object_xpos[0] - self.pos_wall0[0]) >= self.size_object[0] + self.size_wall[0] \
                    and abs(obstacle1_xpos[0] - self.pos_wall0[0]) >= self.size_obstacle[0] + self.size_wall[0] \
                    and (abs(object_xpos[0] - obstacle1_xpos[0]) >= self.size_object[0] + self.size_obstacle[0] or abs(
                            object_xpos[1] - obstacle1_xpos[1]) >= self.size_object[1] + self.size_obstacle[1]):
                return True
            else:
                return False

        # Randomize start position of object.
        if self.random_box and self.np_random.uniform() < self.random_ratio:
            self.sample_hard = False
            object_xpos = self.initial_masspoint_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            stick1_xpos = object_xpos.copy()
            while not config_valid(object_xpos, stick1_xpos):
                object_xpos = self.initial_masspoint_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                stick1_xpos = self.initial_masspoint_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
        else:
            self.sample_hard = True
            # TODO
            object_xpos = self.initial_masspoint_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                                 size=2)
            stick1_xpos = np.asarray(
                [self.pos_wall0[0] + self.size_wall[0] + self.size_obstacle[0], self.initial_masspoint_xpos[1]])
            while not (np.linalg.norm(object_xpos - masspoint_pos) >= 0.1 \
                    and abs(object_xpos[0] - self.pos_wall0[0]) >= self.size_object[0] + self.size_wall[0] \
                    and (abs(object_xpos[0] - stick1_xpos[0]) >= self.size_object[0] + self.size_obstacle[0] or abs(
                            object_xpos[1] - stick1_xpos[1]) >= self.size_object[1] + self.size_obstacle[1])):
                object_xpos = self.initial_masspoint_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                                     size=2)
        # Set the position of box. (two slide joints)
        # box_jointx_i = self.sim.model.get_joint_qpos_addr("object0:slidex")
        # box_jointy_i = self.sim.model.get_joint_qpos_addr("object0:slidey")
        # sim_state.qpos[box_jointx_i] = object_xpos[0]
        # sim_state.qpos[box_jointy_i] = object_xpos[1]
        self.sim.set_state(sim_state)
        # Set the position of obstacle. (free joint)
        box_qpos = self.sim.data.get_joint_qpos('object0:joint')
        stick1_qpos = self.sim.data.get_joint_qpos('object1:joint')
        assert box_qpos.shape == (7,)
        assert stick1_qpos.shape == (7,)
        box_qpos[:2] = object_xpos
        stick1_qpos[:2] = stick1_xpos
        self.sim.data.set_joint_qpos('object0:joint', box_qpos)
        self.sim.data.set_joint_qpos('object1:joint', stick1_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if not hasattr(self, 'size_wall'):
            self.size_wall = self.sim.model.geom_size[self.sim.model.geom_name2id('wall0')]
        if not hasattr(self, 'size_object'):
            self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]
        if not hasattr(self, 'pos_wall0'):
            self.pos_wall0 = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall0')]
        g_idx = np.random.randint(2)
        one_hot = np.zeros(2)
        one_hot[g_idx] = 1
        goal = self.initial_masspoint_xpos[:2] + self.target_offset + self.np_random.uniform(-self.target_range, self.target_range, size=2)
        if hasattr(self, 'sample_hard') and self.sample_hard and g_idx == 0:
            while (goal[0] - self.pos_wall0[0]) * (self.sim.data.get_site_xpos('object0')[0] - self.pos_wall0[0]) > 0:
                goal = self.initial_masspoint_xpos[:2] + self.target_offset + self.np_random.uniform(-self.target_range, self.target_range, size=2)
        goal = np.concatenate([goal, self.sim.data.get_site_xpos('object' + str(g_idx))[2:3], one_hot])
        if self.target_in_the_air and self.np_random.uniform() < 0.5:
            goal[2] += self.np_random.uniform(0, 0.45)
        return goal.copy()

    def compute_reward(self, observation, goal, info):
        # Note: the input is different from other environments.
        one_hot = goal[3:]
        idx = np.argmax(one_hot)
        # HACK: parse the corresponding object position from observation
        achieved_goal = observation[3 + 3 * idx : 3 + 3 * (idx + 1)]
        r = MasspointPushEnv.compute_reward(self, achieved_goal, goal[0:3], info)
        return r

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'][0:3], self.goal[0:3]),
        }
        reward = self.compute_reward(obs['observation'], self.goal, info)
        return obs, reward, done, info

class MasspointPushSingleObstacleEnv_v2(MasspointPushEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', random_box=True,
                 random_ratio=1.0, random_pusher=False):
        XML_PATH = MODEL_XML_PATH2
        initial_qpos = {
            'masspoint:slidex': 1.25,
            'masspoint:slidey': 0.75,
            'masspoint:slidez': 0.15,
            'object0:slidex': 0.0,
            'object0:slidey': 0.0,
            'object0:slidez': 0.0,
            'object1:slidex': 0.0,
            'object1:slidey': 0.0,
            'object1:slidez': 0.0,
            # 'object1:rz': 0.0,
            # 'object0:joint': [1.2, 0.75, 0.025, 1., 0., 0., 0.],
            # 'object1:joint': [1.4, 0.47, 0.03, 1., 0., 0., 0.],
        }
        self.random_box = random_box
        self.random_ratio = random_ratio
        self.random_pusher = random_pusher
        MasspointPushEnv.__init__(
            self, XML_PATH, n_substeps=5,
            target_in_the_air=False, target_offset=0.0,
            obj_range=1.0, target_range=1.0, distance_threshold=0.30,
            initial_qpos=initial_qpos, reward_type=reward_type, n_object=2)
        utils.EzPickle.__init__(self)
        self.pos_wall0 = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall0')]
        self.size_wall = self.sim.model.geom_size[self.sim.model.geom_name2id('wall0')]
        self.size_obstacle = self.sim.model.geom_size[self.sim.model.geom_name2id('object1')]
        self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]

    def _set_action(self, action):
        action *= 20
        MasspointPushEnv._set_action(self, action)

    def inside_wall(self, pos):
        if abs(pos[0] - self.pos_wall0[0]) < self.size_wall[0] and abs(pos[1] - self.initial_masspoint_xpos[1]) > 0.5:
            return True
        return False

    def _reset_sim(self):
        # self.sim.set_state(self.initial_state)
        sim_state = copy.deepcopy(self.initial_state)
        # TODO: randomize masspoint pos
        if self.random_pusher:
            masspoint_jointx_i = self.sim.model.get_joint_qpos_addr('masspoint:slidex')
            masspoint_jointy_i = self.sim.model.get_joint_qpos_addr('masspoint:slidey')
            masspoint_pos = self.initial_masspoint_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            while self.inside_wall(masspoint_pos):
                masspoint_pos = self.initial_masspoint_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            sim_state.qpos[masspoint_jointx_i] = masspoint_pos[0]
            sim_state.qpos[masspoint_jointy_i] = masspoint_pos[1]
        else:
            masspoint_pos = self.initial_masspoint_xpos[:2]

        def config_valid(object_xpos, obstacle1_xpos):
            if np.linalg.norm(object_xpos - masspoint_pos) >= 0.6 \
                    and abs(object_xpos[0] - self.pos_wall0[0]) >= self.size_object[0] + self.size_wall[0] \
                    and abs(obstacle1_xpos[0] - self.pos_wall0[0]) >= self.size_obstacle[0] + self.size_wall[0] \
                    and (abs(object_xpos[0] - obstacle1_xpos[0]) >= self.size_object[0] + self.size_obstacle[0] or abs(
                            object_xpos[1] - obstacle1_xpos[1]) >= self.size_object[1] + self.size_obstacle[1]) \
                    and (abs(obstacle1_xpos[0] - masspoint_pos[0]) > self.size_obstacle[0] + 0.15
                         or abs(obstacle1_xpos[1] - masspoint_pos[1]) > self.size_obstacle[1] + 0.15):
                return True
            else:
                return False

        # Randomize start position of object.
        if self.random_box and self.np_random.uniform() < self.random_ratio:
            self.sample_hard = False
            object_xpos = self.initial_masspoint_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            stick1_xpos = object_xpos.copy()
            while not config_valid(object_xpos, stick1_xpos):
                object_xpos = self.initial_masspoint_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                stick1_xpos = self.initial_masspoint_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
        else:
            self.sample_hard = True
            # TODO
            object_xpos = self.initial_masspoint_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                                 size=2)
            stick1_xpos = np.asarray(
                [np.random.choice([self.pos_wall0[0] + self.size_wall[0] + self.size_obstacle[0],
                                   self.pos_wall0[0] - self.size_wall[0] - self.size_obstacle[0]]), self.initial_masspoint_xpos[1]])
            while not (np.linalg.norm(object_xpos - masspoint_pos) >= 0.6 \
                    and abs(object_xpos[0] - self.pos_wall0[0]) >= self.size_object[0] + self.size_wall[0] \
                    and (abs(object_xpos[0] - stick1_xpos[0]) >= self.size_object[0] + self.size_obstacle[0] or abs(
                            object_xpos[1] - stick1_xpos[1]) >= self.size_object[1] + self.size_obstacle[1])):
                object_xpos = self.initial_masspoint_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                                     size=2)
        # Set the position of box. (two slide joints)
        box_jointx_i = self.sim.model.get_joint_qpos_addr("object0:slidex")
        box_jointy_i = self.sim.model.get_joint_qpos_addr("object0:slidey")
        stick_jointx_i = self.sim.model.get_joint_qpos_addr("object1:slidex")
        stick_jointy_i = self.sim.model.get_joint_qpos_addr("object1:slidey")
        sim_state.qpos[box_jointx_i] = object_xpos[0]
        sim_state.qpos[box_jointy_i] = object_xpos[1]
        sim_state.qpos[stick_jointx_i] = stick1_xpos[0]
        sim_state.qpos[stick_jointy_i] = stick1_xpos[1]
        self.sim.set_state(sim_state)
        # Set the position of obstacle. (free joint)
        # box_qpos = self.sim.data.get_joint_qpos('object0:joint')
        # stick1_qpos = self.sim.data.get_joint_qpos('object1:joint')
        # assert box_qpos.shape == (7,)
        # assert stick1_qpos.shape == (7,)
        # box_qpos[:2] = object_xpos
        # stick1_qpos[:2] = stick1_xpos
        # self.sim.data.set_joint_qpos('object0:joint', box_qpos)
        # self.sim.data.set_joint_qpos('object1:joint', stick1_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if not hasattr(self, 'size_wall'):
            self.size_wall = self.sim.model.geom_size[self.sim.model.geom_name2id('wall0')]
        if not hasattr(self, 'size_object'):
            self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]
        if not hasattr(self, 'pos_wall0'):
            self.pos_wall0 = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall0')]
        g_idx = np.random.randint(2)
        one_hot = np.zeros(2)
        one_hot[g_idx] = 1
        goal = self.initial_masspoint_xpos[:2] + self.target_offset + self.np_random.uniform(-self.target_range, self.target_range, size=2)
        if hasattr(self, 'sample_hard') and self.sample_hard and g_idx == 0:
            while self.inside_wall(goal) or (goal[0] - self.pos_wall0[0]) * (self.sim.data.get_site_xpos('object0')[0] - self.pos_wall0[0]) > 0:
                goal = self.initial_masspoint_xpos[:2] + self.target_offset + self.np_random.uniform(-self.target_range, self.target_range, size=2)
        else:
            while self.inside_wall(goal):
                goal = self.initial_masspoint_xpos[:2] + self.target_offset + self.np_random.uniform(-self.target_range, self.target_range, size=2)
        goal = np.concatenate([goal, self.sim.data.get_site_xpos('object' + str(g_idx))[2:3], one_hot])
        if self.target_in_the_air and self.np_random.uniform() < 0.5:
            goal[2] += self.np_random.uniform(0, 0.45)
        return goal.copy()

    def compute_reward(self, observation, goal, info):
        # Note: the input is different from other environments.
        one_hot = goal[3:]
        idx = np.argmax(one_hot)
        # HACK: parse the corresponding object position from observation
        achieved_goal = observation[3 + 3 * idx : 3 + 3 * (idx + 1)]
        r = MasspointPushEnv.compute_reward(self, achieved_goal, goal[0:3], info)
        return r

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'][0:3], self.goal[0:3]),
        }
        reward = self.compute_reward(obs['observation'], self.goal, info)
        return obs, reward, done, info

    def _viewer_setup(self):
        # body_id = self.sim.model.body_name2id('robot0:gripper_link')
        # lookat = self.sim.data.body_xpos[body_id]
        lookat = [1.25, 0.75, 0.0]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 10.0
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -60.

class MasspointPushDoubleObstacleEnv(MasspointPushEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', random_box=True,
                 random_ratio=1.0, random_pusher=False):
        XML_PATH = MODEL_XML_PATH
        initial_qpos = {
            'masspoint:slidex': 2.5,
            'masspoint:slidey': 2.5,
            'masspoint:slidez': 0.15,
            # 'object0:slidex': 0.0,
            # 'object0:slidey': 0.0,
            'object0:slidez': 0.15,
            # 'object1:slidex': 0,
            # 'object1:slidey': 0,
            'object1:slidez': 0.15,
            # 'object2:slidex': 0,
            # 'object2:slidey': 0,
            'object2:slidez': 0.15,
        }
        self.random_box = random_box
        self.random_ratio = random_ratio
        self.random_pusher = random_pusher
        MasspointPushEnv.__init__(
            self, XML_PATH, n_substeps=10,
            target_in_the_air=False, target_offset=0.0,
            obj_range=1.5, target_range=1.5, distance_threshold=0.30,
            initial_qpos=initial_qpos, reward_type=reward_type, n_object=3)
        utils.EzPickle.__init__(self)
        self.pos_wall0 = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall0')]
        self.pos_wall2 = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall2')]
        self.size_wall = self.sim.model.geom_size[self.sim.model.geom_name2id('wall0')]
        self.size_obstacle = self.sim.model.geom_size[self.sim.model.geom_name2id('object1')]
        self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]

    def _reset_sim(self):
        # self.sim.set_state(self.initial_state)
        sim_state = copy.deepcopy(self.initial_state)
        # TODO: randomize masspoint pos
        if self.random_pusher:
            masspoint_jointx_i = self.sim.model.get_joint_qpos_addr('masspoint:slidex')
            masspoint_jointy_i = self.sim.model.get_joint_qpos_addr('masspoint:slidey')
            masspoint_pos = self.initial_masspoint_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            while self.inside_wall(masspoint_pos):
                masspoint_pos = self.initial_masspoint_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            sim_state.qpos[masspoint_jointx_i] = masspoint_pos[0]
            sim_state.qpos[masspoint_jointy_i] = masspoint_pos[1]
        else:
            masspoint_pos = self.initial_masspoint_xpos[:2]

        def config_valid(object_xpos, obstacle1_xpos, obstacle2_xpos):
            if np.linalg.norm(object_xpos - masspoint_pos) >= 0.3 \
                    and abs(object_xpos[0] - self.pos_wall0[0]) >= self.size_object[0] + self.size_wall[0] \
                    and abs(object_xpos[0] - self.pos_wall2[0]) >= self.size_object[0] + self.size_wall[0] \
                    and abs(obstacle1_xpos[0] - self.pos_wall0[0]) >= self.size_obstacle[0] + self.size_wall[0] \
                    and abs(obstacle1_xpos[0] - self.pos_wall2[0]) >= self.size_obstacle[0] + self.size_wall[0] \
                    and abs(obstacle2_xpos[0] - self.pos_wall0[0]) >= self.size_obstacle[0] + self.size_wall[0] \
                    and abs(obstacle2_xpos[0] - self.pos_wall2[0]) >= self.size_obstacle[0] + self.size_wall[0] \
                    and (abs(object_xpos[0] - obstacle1_xpos[0]) >= self.size_object[0] + self.size_obstacle[0] or abs(
                            object_xpos[1] - obstacle1_xpos[1]) >= self.size_object[1] + self.size_obstacle[1]) \
                    and (abs(object_xpos[0] - obstacle2_xpos[0]) >= self.size_object[0] + self.size_obstacle[0] or abs(
                            object_xpos[1] - obstacle2_xpos[1]) >= self.size_object[1] + self.size_obstacle[1]) \
                    and (abs(obstacle1_xpos[0] - obstacle2_xpos[0]) >= self.size_obstacle[0] * 2 or abs(
                            obstacle1_xpos[1] - obstacle2_xpos[1]) >= self.size_obstacle[1] * 2):
                return True
            else:
                return False

        # Randomize start position of object.
        if self.random_box and self.np_random.uniform() < self.random_ratio:
            self.sample_hard = False
            object_xpos = self.initial_masspoint_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            stick1_xpos = object_xpos.copy()
            stick2_xpos = object_xpos.copy()
            while not config_valid(object_xpos, stick1_xpos, stick2_xpos):
                object_xpos = self.initial_masspoint_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                stick1_xpos = self.initial_masspoint_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                stick2_xpos = self.initial_masspoint_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
        else:
            self.sample_hard = True
            if self.np_random.uniform() < 0.5:
                self.sample_harder = True
            else:
                self.sample_harder = False
            if masspoint_pos[0] < self.pos_wall0[0]:
                stick1_xpos = np.array([self.pos_wall0[0] - self.size_wall[0] - self.size_obstacle[0], 2.5])
                stick2_xpos = np.array([self.pos_wall2[0] - self.size_wall[0] - self.size_obstacle[0], 2.5])
            elif masspoint_pos[0] < self.pos_wall2[0]:
                stick1_xpos = np.array([self.pos_wall0[0] + self.size_wall[0] + self.size_obstacle[0], 2.5])
                stick2_xpos = np.array([self.pos_wall2[0] - self.size_wall[0] - self.size_obstacle[0], 2.5])
            else:
                stick1_xpos = np.array([self.pos_wall0[0] + self.size_wall[0] + self.size_obstacle[0], 2.5])
                stick2_xpos = np.array([self.pos_wall2[0] + self.size_wall[0] + self.size_obstacle[0], 2.5])
            # stick1_xpos = np.array([np.random.choice([self.pos_wall0[0] - self.size_wall[0] - self.size_obstacle[0],
            #                                           self.pos_wall0[0] + self.size_wall[0] + self.size_obstacle[0]]), 2.5])
            # stick2_xpos = np.array([np.random.choice([self.pos_wall2[0] - self.size_wall[0] - self.size_obstacle[0],
            #                                           self.pos_wall2[0] + self.size_wall[0] + self.size_obstacle[0]]), 2.5])
            object_xpos = self.initial_masspoint_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            if not self.sample_harder:
                while not (np.linalg.norm(object_xpos - masspoint_pos) >= 0.3
                           and abs(object_xpos[0] - self.pos_wall0[0]) >= self.size_object[0] + self.size_wall[0]
                           and abs(object_xpos[0] - self.pos_wall2[0]) >= self.size_object[0] + self.size_wall[0]
                           and (abs(object_xpos[0] - stick1_xpos[0]) >= self.size_object[0] + self.size_obstacle[0] or abs(
                                object_xpos[1] - stick1_xpos[1]) >= self.size_object[1] + self.size_obstacle[1])
                           and (abs(object_xpos[0] - stick2_xpos[0]) >= self.size_object[0] + self.size_obstacle[0] or abs(
                                object_xpos[1] - stick2_xpos[1]) >= self.size_object[1] + self.size_obstacle[1])):
                    object_xpos = self.initial_masspoint_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            else:
                while not (np.linalg.norm(object_xpos - masspoint_pos) >= 0.3
                           and abs(object_xpos[0] - self.pos_wall0[0]) >= self.size_object[0] + self.size_wall[0]
                           and abs(object_xpos[0] - self.pos_wall2[0]) >= self.size_object[0] + self.size_wall[0]
                           and (abs(object_xpos[0] - stick1_xpos[0]) >= self.size_object[0] + self.size_obstacle[0] or abs(
                                object_xpos[1] - stick1_xpos[1]) >= self.size_object[1] + self.size_obstacle[1])
                           and (abs(object_xpos[0] - stick2_xpos[0]) >= self.size_object[0] + self.size_obstacle[0] or abs(
                                object_xpos[1] - stick2_xpos[1]) >= self.size_object[1] + self.size_obstacle[1])
                           and ((object_xpos[0] - self.pos_wall0[0]) * (masspoint_pos[0] - self.pos_wall0[0]) < 0
                                or (object_xpos[0] - self.pos_wall2[0]) * (masspoint_pos[0] - self.pos_wall2[0]) < 0)):
                    object_xpos = self.initial_masspoint_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
        # Set the position of box. (two slide joints)
        box_jointx_i = self.sim.model.get_joint_qpos_addr("object0:slidex")
        box_jointy_i = self.sim.model.get_joint_qpos_addr("object0:slidey")
        stick1_jointx_i = self.sim.model.get_joint_qpos_addr("object1:slidex")
        stick1_jointy_i = self.sim.model.get_joint_qpos_addr("object1:slidey")
        stick2_jointx_i = self.sim.model.get_joint_qpos_addr("object2:slidex")
        stick2_jointy_i = self.sim.model.get_joint_qpos_addr("object2:slidey")
        sim_state.qpos[box_jointx_i] = object_xpos[0]
        sim_state.qpos[box_jointy_i] = object_xpos[1]
        sim_state.qpos[stick1_jointx_i] = stick1_xpos[0]
        sim_state.qpos[stick1_jointy_i] = stick1_xpos[1]
        sim_state.qpos[stick2_jointx_i] = stick2_xpos[0]
        sim_state.qpos[stick2_jointy_i] = stick2_xpos[1]
        self.sim.set_state(sim_state)
        self.sim.forward()
        return True

    def _sample_goal(self):
        if not hasattr(self, 'size_wall'):
            self.size_wall = self.sim.model.geom_size[self.sim.model.geom_name2id('wall0')]
        if not hasattr(self, 'size_object'):
            self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]
        if not hasattr(self, 'pos_wall0'):
            self.pos_wall0 = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall0')]
        if not hasattr(self, 'pos_wall2'):
            self.pos_wall2 = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall2')]
        g_idx = np.random.randint(self.n_object)
        one_hot = np.zeros(self.n_object)
        one_hot[g_idx] = 1
        goal = self.initial_masspoint_xpos[:2] + self.target_offset + self.np_random.uniform(-self.target_range, self.target_range, size=2)

        def same_side(pos0, pos1, sep):
            if (pos0 - sep) * (pos1 - sep) > 0:
                return True
            return False

        if hasattr(self, 'sample_hard') and self.sample_hard:
            g_idx = 0
            one_hot = np.zeros(self.n_object)
            one_hot[g_idx] = 1
            if hasattr(self, 'sample_harder') and self.sample_harder:
                # print('sample harder')
                masspoint_pos = self.sim.data.get_site_xpos('masspoint')
                object_pos = self.sim.data.get_site_xpos('object0')
                while (same_side(goal[0], object_pos[0], self.pos_wall0[0]) and same_side(goal[0], object_pos[0], self.pos_wall2[0])
                       or (same_side(goal[0], masspoint_pos[0], self.pos_wall0[0]) and same_side(goal[0], masspoint_pos[0], self.pos_wall2[0]))
                       or self.inside_wall(goal)):
                    goal = self.initial_masspoint_xpos[:2] + self.target_offset + self.np_random.uniform(-self.target_range, self.target_range, size=2)
            else:
                while (same_side(goal[0], self.sim.data.get_site_xpos('object0')[0], self.pos_wall0[0]) and
                       same_side(goal[0], self.sim.data.get_site_xpos('object0')[0], self.pos_wall2[0])) \
                        or self.inside_wall(goal):
                    goal = self.initial_masspoint_xpos[:2] + self.target_offset + self.np_random.uniform(-self.target_range, self.target_range, size=2)
        else:
            if g_idx != 0:
                # while self.inside_wall(goal) \
                #         or (not same_side(goal[0], self.sim.data.get_site_xpos('object' + str(g_idx))[0], self.pos_wall0[0])) \
                #         or (not same_side(goal[0], self.sim.data.get_site_xpos('object' + str(g_idx))[0], self.pos_wall2[0])):
                while self.inside_wall(goal):
                    goal = self.initial_masspoint_xpos[:2] + self.target_offset + self.np_random.uniform(-self.target_range, self.target_range, size=2)
            else:
                while self.inside_wall(goal):
                    goal = self.initial_masspoint_xpos[:2] + self.target_offset + self.np_random.uniform(-self.target_range, self.target_range, size=2)
        goal = np.concatenate([goal, self.sim.data.get_site_xpos('object' + str(g_idx))[2:3], one_hot])
        if self.target_in_the_air and self.np_random.uniform() < 0.5:
            goal[2] += self.np_random.uniform(0, 0.45)
        return goal.copy()

    def compute_reward(self, observation, goal, info):
        # Note: the input is different from other environments.
        one_hot = goal[3:]
        idx = np.argmax(one_hot)
        # HACK: parse the corresponding object position from observation
        achieved_goal = observation[3 + 3 * idx : 3 + 3 * (idx + 1)]
        r = MasspointPushEnv.compute_reward(self, achieved_goal, goal[0:3], info)
        return r

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action * 20)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'][0:3], self.goal[0:3]),
        }
        reward = self.compute_reward(obs['observation'], self.goal, info)
        return obs, reward, done, info

    def _viewer_setup(self):
        # body_id = self.sim.model.body_name2id('robot0:gripper_link')
        # lookat = self.sim.data.body_xpos[body_id]
        lookat = [2.5, 2.5, 0.0]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 10.0
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -60.

    def inside_wall(self, pos):
        if (abs(pos[0] - self.pos_wall0[0]) < self.size_wall[0] or abs(pos[0] - self.pos_wall2[0]) < self.size_wall[0]) \
                and abs(pos[1] - 2.5) > 0.5:
            return True
        return False
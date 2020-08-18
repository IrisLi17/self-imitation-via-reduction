import os
import copy
from gym import utils
from masspoint_base import MasspointPushEnv
import gym.envs.robotics.utils as robot_utils
from gym.envs.robotics import rotations
import numpy as np
from mujoco_py import MujocoException
import tempfile
from assets.masspoint.generate_xml import generate_xml


MODEL_XML_PATH0 = os.path.join(os.path.dirname(__file__), 'assets', 'masspoint', 'single_obstacle.xml')
MODEL_XML_PATH2 = os.path.join(os.path.dirname(__file__), 'assets', 'masspoint', 'single_obstacle2.xml')
MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'masspoint', 'double_obstacle.xml')
MAZE_XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'masspoint', 'maze.xml')
SMAZE_XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'masspoint', 'smaze.xml')


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

        # if hasattr(self, 'sample_hard') and self.sample_hard and g_idx == 0:
        if hasattr(self, 'sample_hard') and self.sample_hard:
            if self.np_random.uniform() < 0.6:
                g_idx = 0
            else:
                g_idx = np.random.randint(1, self.n_object)
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
        # # Note: the input is different from other environments.
        # one_hot = goal[3:]
        # idx = np.argmax(one_hot)
        # # HACK: parse the corresponding object position from observation
        # achieved_goal = observation[3 + 3 * idx : 3 + 3 * (idx + 1)]
        # r = MasspointPushEnv.compute_reward(self, achieved_goal, goal[0:3], info)
        # return r
        r, _ = self.compute_reward_and_success(observation, goal, info)
        return r

    def compute_reward_and_success(self, observation, goal, info):
        one_hot = goal[3:]
        idx = np.argmax(one_hot)
        achieved_goal = observation[3 + 3 * idx: 3 + 3 * (idx + 1)]
        success = np.linalg.norm(achieved_goal - goal[0:3]) < self.distance_threshold
        if self.reward_type == "dense":
            r = 0.1 * MasspointPushEnv.compute_reward(self, achieved_goal, goal[0:3], info) + success
        else:
            r = MasspointPushEnv.compute_reward(self, achieved_goal, goal[0:3], info)
        return r, success

    def step(self, action):
        try:
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
        except MujocoException:
            obs = self.reset()
            done = True
            info = {'is_success': False}
            reward = -1
            print('catch mujoco error, reset env')
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


class MasspointPushDoubleObstacleEnv_v2(MasspointPushDoubleObstacleEnv):
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
            initial_qpos=initial_qpos, reward_type=reward_type, n_object=4)
        utils.EzPickle.__init__(self)
        self.pos_wall0 = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall0')]
        self.pos_wall2 = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall2')]
        self.size_wall = self.sim.model.geom_size[self.sim.model.geom_name2id('wall0')]
        self.size_obstacle = self.sim.model.geom_size[self.sim.model.geom_name2id('object1')]
        self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]

    def _get_obs(self):
        # Agent itself goes into objects as well
        # positions
        # grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        masspoint_pos = self.sim.data.get_site_xpos('masspoint')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        # grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        masspoint_velp = self.sim.data.get_site_xvelp('masspoint') * dt
        # robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        object_pos = [self.sim.data.get_site_xpos('object' + str(i)) for i in range(self.n_object - 1)]
        object_pos += [masspoint_pos]
        # rotations
        object_rot = [rotations.mat2euler(self.sim.data.get_site_xmat('object' + str(i))) for i in range(self.n_object - 1)]
        object_rot += [rotations.mat2euler(self.sim.data.get_site_xmat('masspoint'))]
        # velocities
        object_velp = [self.sim.data.get_site_xvelp('object' + str(i)) * dt for i in range(self.n_object - 1)]
        object_velp += [self.sim.data.get_site_xvelp('masspoint') * dt]
        object_velr = [self.sim.data.get_site_xvelr('object' + str(i)) * dt for i in range(self.n_object - 1)]
        object_velr += [self.sim.data.get_site_xvelr('masspoint') * dt]
        # gripper state
        object_rel_pos = [object_pos[i] - masspoint_pos for i in range(self.n_object)]
        object_velp = [object_velp[i] - masspoint_velp for i in range(self.n_object)]
        object_pos = np.concatenate(object_pos)
        object_rot = np.concatenate(object_rot)
        object_velp = np.concatenate(object_velp)
        object_velr = np.concatenate(object_velr)
        object_rel_pos = np.concatenate(object_rel_pos)
        # gripper_state = robot_qpos[-2:]
        # gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        # achieved_goal = np.squeeze(object_pos.copy())
        one_hot = self.goal[3:]
        idx = np.argmax(one_hot)
        achieved_goal = np.concatenate([object_pos[3 * idx: 3 * (idx + 1)].copy(), one_hot])
        obs = np.concatenate([
            masspoint_pos, object_pos.ravel(), object_rel_pos.ravel(), object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), masspoint_velp,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

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

        if hasattr(self, 'sample_hard') and self.sample_hard and g_idx == 0:
            # g_idx = 0
            # one_hot = np.zeros(self.n_object)
            # one_hot[g_idx] = 1
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
            while self.inside_wall(goal):
                goal = self.initial_masspoint_xpos[:2] + self.target_offset + self.np_random.uniform(-self.target_range, self.target_range, size=2)
        goal_height = self.sim.data.get_site_xpos('object' + str(g_idx))[2:3] \
            if g_idx < self.n_object - 1 else self.initial_masspoint_xpos[2:3]
        goal = np.concatenate([goal, goal_height, one_hot])
        if self.target_in_the_air and self.np_random.uniform() < 0.5:
            goal[2] += self.np_random.uniform(0, 0.45)
        return goal.copy()

class MasspointMazeEnv(MasspointPushEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', random_box=True,
                 random_ratio=1.0, random_pusher=False):
        XML_PATH = MAZE_XML_PATH
        initial_qpos = {
            'masspoint:slidex': 2.5,
            'masspoint:slidey': 2.5,
            'masspoint:slidez': 0.15,
        }
        self.random_ratio = random_ratio
        self.random_pusher = random_pusher
        MasspointPushEnv.__init__(
            self, XML_PATH, n_substeps=5,
            target_in_the_air=False, target_offset=0.0,
            obj_range=2.0, target_range=2.0, distance_threshold=0.30,
            initial_qpos=initial_qpos, reward_type=reward_type, n_object=0)
        utils.EzPickle.__init__(self)
        self.pos_wall0 = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall0')]
        self.size_wall = self.sim.model.geom_size[self.sim.model.geom_name2id('wall0')]

    def _get_obs(self):
        # positions
        masspoint_pos = self.sim.data.get_site_xpos('masspoint')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        # velocity
        masspoint_velp = self.sim.data.get_site_xvelp('masspoint') * dt
        achieved_goal = masspoint_pos.copy()
        obs = np.concatenate([
            masspoint_pos, masspoint_velp,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _reset_sim(self):
        # self.sim.set_state(self.initial_state)
        sim_state = copy.deepcopy(self.initial_state)
        # TODO: randomize masspoint pos
        masspoint_jointx_i = self.sim.model.get_joint_qpos_addr('masspoint:slidex')
        masspoint_jointy_i = self.sim.model.get_joint_qpos_addr('masspoint:slidey')
        if self.random_pusher:
            masspoint_pos = np.array([2.5, 2.5]) + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            while self.inside_wall(masspoint_pos):
                masspoint_pos = np.array([2.5, 2.5]) + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
        else:
            masspoint_pos = self.initial_masspoint_xpos[:2]
        sim_state.qpos[masspoint_jointx_i] = masspoint_pos[0]
        sim_state.qpos[masspoint_jointy_i] = masspoint_pos[1]

        self.sim.set_state(sim_state)
        self.sim.forward()
        return True

    def _sample_goal(self):
        if not hasattr(self, 'size_wall'):
            self.size_wall = self.sim.model.geom_size[self.sim.model.geom_name2id('wall0')]
        if not hasattr(self, 'pos_wall0'):
            self.pos_wall0 = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall0')]
        # g_idx = np.random.randint(self.n_object)
        # one_hot = np.zeros(self.n_object)
        # one_hot[g_idx] = 1
        goal = np.array([2.5, 2.5]) + self.target_offset + self.np_random.uniform(-self.target_range, self.target_range, size=2)

        def same_side(pos0, pos1, sep):
            if (pos0 - sep) * (pos1 - sep) > 0:
                return True
            return False

        goal = np.concatenate([goal, self.initial_masspoint_xpos[2:3]])
        return goal.copy()

    def switch_obs_goal(self, obs, goal):
        obs = obs.copy()
        if isinstance(obs, dict):
            obs['achieved_goal'] = obs['observation'][:3]
            obs['desired_goal'] = goal[:]
        elif isinstance(obs, np.ndarray):
            obs_dim = self.observation_space['observation'].shape[0]
            goal_dim = self.observation_space['achieved_goal'].shape[0]
            obs[obs_dim:obs_dim+goal_dim] = obs[:3]
            obs[obs_dim+goal_dim:obs_dim+goal_dim*2] = goal[:]
        else:
            raise TypeError
        return obs

    def compute_reward(self, observation, goal, info):
        achieved_goal = observation[:3]
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
        reward = self.compute_reward(obs['observation'][0:3], self.goal, info)
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
        if abs(pos[0] - self.pos_wall0[0]) < self.size_wall[0] and abs(pos[1] - 2.0) < 2.0:
            return True
        return False

class MasspointSMazeEnv(MasspointPushEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', random_box=True,
                 random_ratio=1.0, random_pusher=False):
        XML_PATH = SMAZE_XML_PATH
        initial_qpos = {
            'masspoint:slidex': 2.5,
            'masspoint:slidey': 2.5,
            'masspoint:slidez': 0.15,
        }
        self.random_ratio = random_ratio
        self.random_pusher = random_pusher
        MasspointPushEnv.__init__(
            self, XML_PATH, n_substeps=5,
            target_in_the_air=False, target_offset=0.0,
            obj_range=2.0, target_range=2.0, distance_threshold=0.30,
            initial_qpos=initial_qpos, reward_type=reward_type, n_object=0)
        utils.EzPickle.__init__(self)
        self.pos_wall0 = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall0')]
        self.pos_wall1 = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall1')]
        self.size_wall = self.sim.model.geom_size[self.sim.model.geom_name2id('wall0')]
        self.size_obstacle = np.array([1., 1., 1.])  # Used to determine noise_mag only.

    def _get_obs(self):
        # positions
        masspoint_pos = self.sim.data.get_site_xpos('masspoint')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        # velocity
        masspoint_velp = self.sim.data.get_site_xvelp('masspoint') * dt
        achieved_goal = masspoint_pos.copy()
        obs = np.concatenate([
            masspoint_pos, masspoint_velp,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _reset_sim(self):
        # self.sim.set_state(self.initial_state)
        sim_state = copy.deepcopy(self.initial_state)
        # TODO: randomize masspoint pos
        masspoint_jointx_i = self.sim.model.get_joint_qpos_addr('masspoint:slidex')
        masspoint_jointy_i = self.sim.model.get_joint_qpos_addr('masspoint:slidey')
        if self.random_pusher:
            masspoint_pos = np.array([2.5, 2.5]) + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            while self.inside_wall(masspoint_pos):
                masspoint_pos = np.array([2.5, 2.5]) + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
        else:
            masspoint_pos = self.initial_masspoint_xpos[:2]
        sim_state.qpos[masspoint_jointx_i] = masspoint_pos[0]
        sim_state.qpos[masspoint_jointy_i] = masspoint_pos[1]

        self.sim.set_state(sim_state)
        self.sim.forward()
        return True

    def _sample_goal(self):
        if not hasattr(self, 'size_wall'):
            self.size_wall = self.sim.model.geom_size[self.sim.model.geom_name2id('wall0')]
        if not hasattr(self, 'pos_wall0'):
            self.pos_wall0 = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall0')]
        if not hasattr(self, 'pos_wall1'):
            self.pos_wall1 = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall1')]
        # g_idx = np.random.randint(self.n_object)
        # one_hot = np.zeros(self.n_object)
        # one_hot[g_idx] = 1
        goal = np.array([2.5, 2.5]) + self.target_offset + self.np_random.uniform(-self.target_range, self.target_range, size=2)

        def same_side(pos0, pos1, sep):
            if (pos0 - sep) * (pos1 - sep) > 0:
                return True
            return False

        goal = np.concatenate([goal, self.initial_masspoint_xpos[2:3]])
        return goal.copy()

    def switch_obs_goal(self, obs, goal):
        obs = obs.copy()
        if isinstance(obs, dict):
            obs['achieved_goal'] = obs['observation'][:3]
            obs['desired_goal'] = goal[:]
        elif isinstance(obs, np.ndarray):
            obs_dim = self.observation_space['observation'].shape[0]
            goal_dim = self.observation_space['achieved_goal'].shape[0]
            obs[obs_dim:obs_dim+goal_dim] = obs[:3]
            obs[obs_dim+goal_dim:obs_dim+goal_dim*2] = goal[:]
        else:
            raise TypeError
        return obs

    def compute_reward(self, observation, goal, info):
        achieved_goal = observation[:3]
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
        reward = self.compute_reward(obs['observation'][0:3], self.goal, info)
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
        if abs(pos[0] - self.pos_wall0[0]) < self.size_wall[0] and abs(pos[1] - 2.0) < 2.0:
            return True
        if abs(pos[0] - self.pos_wall1[0]) < self.size_wall[0] and abs(pos[1] - 3.0) < 2.0:
            return True
        return False


class MasspointPushMultiObstacleEnv(MasspointPushEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', n_object=3 + 1, random_box=True,
                 random_ratio=1.0, random_pusher=False):
        initial_qpos = {
            'masspoint:slidex': 1.7 * n_object / 2,
            'masspoint:slidey': 2.5,
            'masspoint:slidez': 0.15,
        }
        for i in range(n_object):
            initial_qpos['object%d:slidez' % i] = 0.15
        self.random_box = random_box
        self.random_ratio = random_ratio
        self.random_pusher = random_pusher
        self.n_object = n_object
        with tempfile.NamedTemporaryFile(mode='wt',
                                         dir=os.path.join(os.path.dirname(__file__), 'assets', 'fetch'),
                                         delete=False, suffix=".xml") as fp:
            fp.write(generate_xml(self.n_object))
            model_path = fp.name
        MasspointPushEnv.__init__(
            self, model_path, n_substeps=10,
            target_in_the_air=False, target_offset=0.0,
            obj_range=1.5, target_range=1.5, distance_threshold=0.30,
            initial_qpos=initial_qpos, reward_type=reward_type, n_object=n_object)
        os.remove(model_path)
        utils.EzPickle.__init__(self)
        self.pos_walls = [self.sim.model.geom_pos[self.sim.model.geom_name2id('wall%d' % (2*i))] for i in range(self.n_object-1)]
        self.size_wall = self.sim.model.geom_size[self.sim.model.geom_name2id('wall0')]
        self.size_obstacle = self.sim.model.geom_size[self.sim.model.geom_name2id('object1')]
        self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]
        self.obj_range = np.array([1.7 * n_object / 2 - self.size_object[0], 2.5 - self.size_object[1]])
        self.obstacle_range = np.array([1.7 - self.size_obstacle[0] - self.size_wall[0], 2.5 - self.size_obstacle[1]])

    def _reset_sim(self):
        # self.sim.set_state(self.initial_state)
        sim_state = copy.deepcopy(self.initial_state)
        # TODO: randomize masspoint pos
        if self.random_pusher:
            masspoint_jointx_i = self.sim.model.get_joint_qpos_addr('masspoint:slidex')
            masspoint_jointy_i = self.sim.model.get_joint_qpos_addr('masspoint:slidey')
            masspoint_pos = self.initial_masspoint_xpos[:2] + self.np_random.uniform(size=2) * self.obj_range
            while self.inside_wall(masspoint_pos):
                masspoint_pos = self.initial_masspoint_xpos[:2] + self.np_random.uniform(size=2) * self.obj_range
            sim_state.qpos[masspoint_jointx_i] = masspoint_pos[0]
            sim_state.qpos[masspoint_jointy_i] = masspoint_pos[1]
        else:
            masspoint_pos = self.initial_masspoint_xpos[:2]

        def config_valid(object_xpos, obstacles_xpos):
            assert isinstance(obstacles_xpos, list)
            conditions = [np.linalg.norm(object_xpos - masspoint_pos) >= 0.3,]
            for wall_pos in self.pos_walls:
                conditions.append(abs(object_xpos[0] - wall_pos[0]) >= self.size_object[0] + self.size_wall[0])
                for obstacle_xpos in obstacles_xpos:
                    conditions.append(abs(obstacle_xpos[0] - wall_pos[0]) >= self.size_obstacle[0] + self.size_wall[0])
            for obstacle_xpos in obstacles_xpos:
                conditions.append(abs(object_xpos[0] - obstacle_xpos[0]) >= self.size_object[0] + self.size_obstacle[0]
                                  or abs(object_xpos[1] - obstacle_xpos[1]) >= self.size_object[1] + self.size_obstacle[1])
            return np.all(conditions)

        # Randomize start position of object.
        if self.random_box and self.np_random.uniform() < self.random_ratio:
            self.sample_hard = False
            object_xpos = self.initial_masspoint_xpos[:2] + self.np_random.uniform(size=2) * self.obj_range
            obstacles_xpos = [np.array([1.7*(i+1), 2.5]) + self.np_random.uniform(size=2) * self.obstacle_range for i in range(self.n_object-1)]
            while not config_valid(object_xpos, obstacles_xpos):
                object_xpos = self.initial_masspoint_xpos[:2] + self.np_random.uniform(size=2) * self.obj_range
                obstacles_xpos = [np.array([1.7 * (i + 1), 2.5]) + self.np_random.uniform(size=2) * self.obstacle_range
                                  for i in range(self.n_object - 1)]
        else:
            self.sample_hard = True
            object_xpos = self.initial_masspoint_xpos[:2] + self.np_random.uniform(size=2) * self.obj_range
            obstacles_xpos = [np.array([1.7 * (i + 1) + self.np_random.choice([-1, 1]) * (self.size_wall[0] + self.size_obstacle[0]), 2.5])
                              for i in range(self.n_object - 1)]
        # Set the position of box. (two slide joints)
        box_jointx_i = self.sim.model.get_joint_qpos_addr("object0:slidex")
        box_jointy_i = self.sim.model.get_joint_qpos_addr("object0:slidey")
        sim_state.qpos[box_jointx_i] = object_xpos[0]
        sim_state.qpos[box_jointy_i] = object_xpos[1]
        for i in range(1, self.n_object):
            obstacle_jointx_i = self.sim.model.get_joint_qpos_addr("object%d:slidex" % i)
            obstacle_jointy_i = self.sim.model.get_joint_qpos_addr("object%d:slidey" % i)
            sim_state.qpos[obstacle_jointx_i] = obstacles_xpos[i - 1][0]
            sim_state.qpos[obstacle_jointy_i] = obstacles_xpos[i - 1][1]
        self.sim.set_state(sim_state)
        self.sim.forward()
        return True

    def _sample_goal(self):
        if not hasattr(self, 'size_wall'):
            self.size_wall = self.sim.model.geom_size[self.sim.model.geom_name2id('wall0')]
        if not hasattr(self, 'size_object'):
            self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]
        if not hasattr(self, 'size_obstacle'):
            self.size_obstacle = self.sim.model.geom_size[self.sim.model.geom_name2id('object1')]
        if not hasattr(self, 'pos_walls'):
            self.pos_walls = [self.sim.model.geom_pos[self.sim.model.geom_name2id('wall%d' % (2 * i))] for i in
                              range(self.n_object - 1)]
        self.obj_range = np.array([1.7 * self.n_object / 2 - self.size_object[0], 2.5 - self.size_object[1]])
        self.obstacle_range = np.array([1.7 - self.size_obstacle[0] - self.size_wall[0], 2.5 - self.size_obstacle[1]])

        g_idx = np.random.randint(self.n_object)
        one_hot = np.zeros(self.n_object)
        one_hot[g_idx] = 1
        goal = self.initial_masspoint_xpos[:2] + self.target_offset + self.np_random.uniform(size=2) * self.obj_range

        def same_side(pos0, pos1, sep):
            if (pos0 - sep) * (pos1 - sep) > 0:
                return True
            return False

        if hasattr(self, 'sample_hard') and self.sample_hard:
            g_idx = 0
            # if self.np_random.uniform() < 0.6:
            #     g_idx = 0
            # else:
            #     g_idx = np.random.randint(1, self.n_object)
            one_hot = np.zeros(self.n_object)
            one_hot[g_idx] = 1
            object_pos = self.sim.data.get_site_xpos('object0')
            if g_idx == 0:
                while np.all([same_side(goal[0], object_pos[0], pos_wall[0]) for pos_wall in
                              self.pos_walls]) or self.inside_wall(goal):
                    goal = self.initial_masspoint_xpos[:2] + self.target_offset + self.np_random.uniform(
                        size=2) * self.obj_range
        else:
            while self.inside_wall(goal):
                goal = self.initial_masspoint_xpos[:2] + self.target_offset + self.np_random.uniform(
                    size=2) * self.obj_range
        goal = np.concatenate([goal, self.sim.data.get_site_xpos('object' + str(g_idx))[2:3], one_hot])
        if self.target_in_the_air and self.np_random.uniform() < 0.5:
            goal[2] += self.np_random.uniform(0, 0.45)
        return goal.copy()

    def compute_reward(self, observation, goal, info):
        r, _ = self.compute_reward_and_success(observation, goal, info)
        return r

    def compute_reward_and_success(self, observation, goal, info):
        one_hot = goal[3:]
        idx = np.argmax(one_hot)
        achieved_goal = observation[3 + 3 * idx: 3 + 3 * (idx + 1)]
        success = np.linalg.norm(achieved_goal - goal[0:3]) < self.distance_threshold
        if self.reward_type == "dense":
            r = 0.1 * MasspointPushEnv.compute_reward(self, achieved_goal, goal[0:3], info) + success
        else:
            r = MasspointPushEnv.compute_reward(self, achieved_goal, goal[0:3], info)
        return r, success

    def step(self, action):
        try:
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
        except MujocoException:
            obs = self.reset()
            done = True
            info = {'is_success': False}
            reward = -1
            print('catch mujoco error, reset env')
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
        for pos_wall in self.pos_walls:
            if abs(pos[0] - pos_wall[0]) < self.size_wall[0] and abs(pos[1] - 2.5) > 0.5:
                return True
        return False
        # return np.any([abs(pos[0] - pos_wall[0]) < self.size_wall[0]
        #                and abs(pos[1] - 2.5) > 0.5] for pos_wall in self.pos_walls)

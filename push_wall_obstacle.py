import os
from gym import utils
from gym.envs.robotics import fetch_env, rotations
import gym.envs.robotics.utils as robot_utils
import numpy as np


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'fetch', 'push_wall_obstacle.xml')
MODEL_XML_PATH2 = os.path.join(os.path.dirname(__file__), 'assets', 'fetch', 'push_wall_heavy_obstacle.xml')
MODEL_XML_PATH3 = os.path.join(os.path.dirname(__file__), 'assets', 'fetch', 'push_wall_heavy_obstacle_v2.xml')
MODEL_XML_PATH4 = os.path.join(os.path.dirname(__file__), 'assets', 'fetch', 'push_wall_heavy_obstacle_v5.xml')


class FetchPushWallObstacleEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', penaltize_height=False, heavy_obstacle=False, random_box=True, 
                 random_ratio=1.0, hack_obstacle=False, random_gripper=False, hide_velocity=False):
        if heavy_obstacle:
            XML_PATH = MODEL_XML_PATH2
        else:
            XML_PATH = MODEL_XML_PATH
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.2, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.35, 0.75, 0.4, 1., 0., 0., 0.],
        }
        self.n_object = sum([('object' in item) for item in initial_qpos.keys()])
        self.penaltize_height = penaltize_height
        self.random_box = random_box
        self.random_ratio = random_ratio
        self.hack_obstacle = hack_obstacle
        self.random_gripper = random_gripper
        self.hide_velocity = hide_velocity
        fetch_env.FetchEnv.__init__(
            self, XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
        self.pos_wall = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall0')]
        self.size_wall = self.sim.model.geom_size[self.sim.model.geom_name2id('wall0')]
        self.size_obstacle = self.sim.model.geom_size[self.sim.model.geom_name2id('object1')]
        self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = robot_utils.robot_get_obs(self.sim)
        if self.has_object:
            # object_pos = self.sim.data.get_site_xpos('object0')
            object_pos = [self.sim.data.get_site_xpos('object' + str(i)) for i in range(self.n_object)]
            # rotations
            # object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            object_rot = [rotations.mat2euler(self.sim.data.get_site_xmat('object' + str(i))) for i in range(self.n_object)]
            # velocities
            # object_velp = self.sim.data.get_site_xvelp('object0') * dt
            # object_velr = self.sim.data.get_site_xvelr('object0') * dt
            object_velp = [self.sim.data.get_site_xvelp('object' + str(i)) * dt for i in range(self.n_object)]
            object_velr = [self.sim.data.get_site_xvelr('object' + str(i)) * dt for i in range(self.n_object)]
            # gripper state
            # object_rel_pos = object_pos - grip_pos
            object_rel_pos = [pos - grip_pos for pos in object_pos]
            # object_velp -= grip_velp
            object_velp = [velp - grip_velp for velp in object_velp]

            object_pos = np.concatenate(object_pos)
            object_rot = np.concatenate(object_rot)
            object_velp = np.concatenate(object_velp)
            object_velr = np.concatenate(object_velr)
            object_rel_pos = np.concatenate(object_rel_pos)
            # the stick
            # stick_pos = self.sim.data.get_site_xpos('object1')
            # stick_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object1'))
            # stick_velp = self.sim.data.get_site_xvelp('object1') * dt
            # stick_velr = self.sim.data.get_site_xvelr('object1') * dt
            # stick_rel_pos = stick_pos - grip_pos
            # stick_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
            # stick_pos = stick_rot = stick_velp = stick_velr = stick_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            # achieved_goal = np.squeeze(object_pos.copy())
            achieved_goal = self.sim.data.get_site_xpos('object0').copy()
        if not self.hide_velocity:
            obs = np.concatenate([
                grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
                object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
            ]) # dim 40
        else:
            obs = np.concatenate([
                grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
                grip_velp, gripper_vel,
            ])
        # print('grip_pos', grip_pos.shape) # (3,)
        # print('object_pos', object_pos.shape) # (6,)
        # print('object_rel_pos', object_rel_pos.shape) # (6,)
        # print('object_rot', object_rot.shape) # (6,)
        # print('gripper_state', gripper_state.shape) # (2,)
        # print('object_velp', object_velp.shape) # (6,)
        # print('object_velr', object_velr.shape) # (6,)
        # print('grip_velp', grip_velp.shape) # (3,)
        # print('gripper_vel', gripper_vel.shape) # (2,)

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }
    
    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        # TODO: randomize mocap_pos
        if self.random_gripper:
            mocap_pos = np.concatenate([self.np_random.uniform([1.19, 0.6], [1.49, 0.9]), [0.355]])
            self.sim.data.set_mocap_pos('robot0:mocap', mocap_pos)
            for _ in range(10):
                self.sim.step()
            self._step_callback()

        # Randomize start position of object.
        if self.has_object:
            if self.random_box and self.np_random.uniform() < self.random_ratio:
                object_xpos = self.initial_gripper_xpos[:2]
                stick_xpos = object_xpos.copy()
                while (np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1
                       or abs(object_xpos[0] - self.pos_wall[0]) < self.size_wall[0] + self.size_object[0] or abs(stick_xpos[0] - self.pos_wall[0]) < self.size_wall[0] + self.size_obstacle[0]
                       or (abs(object_xpos[0] - stick_xpos[0]) < self.size_object[0] + self.size_obstacle[0] and abs(object_xpos[1] - stick_xpos[1]) < self.size_object[1] + self.size_obstacle[1])):
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                    stick_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            else:
                object_xpos = self.initial_gripper_xpos[:2] + np.asarray([self.obj_range * 0.9, self.obj_range / 2])
                stick_xpos = np.asarray([self.pos_wall[0] + self.size_wall[0] + self.size_obstacle[0], self.initial_gripper_xpos[1]])
                # stick_xpos = self.initial_gripper_xpos[:2] + np.asarray([self.obj_range / 4, 0])
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            stick_qpos = self.sim.data.get_joint_qpos('object1:joint')
            assert object_qpos.shape == (7,)
            assert stick_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            stick_qpos[:2] = stick_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)
            self.sim.data.set_joint_qpos('object1:joint', stick_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if not hasattr(self, 'size_wall'):
                self.size_wall = self.sim.model.geom_size[self.sim.model.geom_name2id('wall0')]
            if not hasattr(self, 'size_object'):
                self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]
            if not hasattr(self, 'pos_wall'):
                self.pos_wall = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall0')]
            # while (abs(goal[0] - self.pos_wall[0]) < self.size_wall[0]):
            #     goal = self.initial_gripper_xpos[:3] + self.target_offset + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            #     goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def compute_reward(self, achieved_goal, goal, info):
        r = fetch_env.FetchEnv.compute_reward(self, achieved_goal, goal, info)
        if self.hack_obstacle:
            r += -0.1 * info['is_blocked']
        return r

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
            'is_blocked': obs['observation'][7] + self.size_obstacle[1] * np.cos(obs['observation'][22]) > 0.85
                          and obs['observation'][7] - self.size_obstacle[1] * np.cos(obs['observation'][22]) < 0.65
                          and abs(obs['observation'][6] - self.pos_wall[0]) < self.size_wall[0] + self.size_obstacle[0] + self.size_object[0]
                          # and (obs['achieved_goal'][0] - self.pos_wall[0]) * (obs['desired_goal'][0] - self.pos_wall[0]) < 0

        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        # Box penalty.
        if self.penaltize_height:
            gripper_height= obs['observation'][2]
            height_penalty = gripper_height > 0.5 or gripper_height < 0.3
            reward = reward - 10 * height_penalty
        return obs, reward, done, info
    
    def goal2observation(self, goal):
        '''
        generate an observation that starts from the goal.
        '''
        obs = self._get_obs()
        assert isinstance(obs, dict)
        # object_pos
        obs['observation'][3:9] = goal
        # object_rel_pos
        obs['observation'][9:12] = obs['observation'][3:6] - obs['observation'][0:3]
        obs['observation'][12:15] = obs['observation'][6:9] - obs['observation'][0:3]
        return obs.copy()

class FetchPushWallObstacleEnv_v2(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', penaltize_height=False, heavy_obstacle=True, random_box=True,
                 random_ratio=1.0, random_gripper=False):
        assert heavy_obstacle
        XML_PATH = MODEL_XML_PATH3
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            # 'object0:joint': [1.2, 0.53, 0.4, 1., 0., 0., 0.],
            'object0:slidex': 0.0,
            'object0:slidey': 0.0,
            'object1:joint': [1.35, 0.75, 0.4, 1., 0., 0., 0.],
        }
        self.n_object = 2
        self.penaltize_height = penaltize_height
        self.random_box = random_box
        self.random_ratio = random_ratio
        self.random_gripper = random_gripper
        fetch_env.FetchEnv.__init__(
            self, XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.07,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
        self.pos_wall = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall0')]
        self.size_wall = self.sim.model.geom_size[self.sim.model.geom_name2id('wall0')]
        self.size_obstacle = self.sim.model.geom_size[self.sim.model.geom_name2id('object1')]
        self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = robot_utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = [self.sim.data.get_site_xpos('object' + str(i)) for i in range(self.n_object)]
            # rotations
            object_rot = [rotations.mat2euler(self.sim.data.get_site_xmat('object' + str(i))) for i in
                          range(self.n_object)]
            # velocities
            object_velp = [self.sim.data.get_site_xvelp('object' + str(i)) * dt for i in range(self.n_object)]
            object_velr = [self.sim.data.get_site_xvelr('object' + str(i)) * dt for i in range(self.n_object)]
            # gripper state
            object_rel_pos = [pos - grip_pos for pos in object_pos]
            object_velp = [velp - grip_velp for velp in object_velp]

            object_pos = np.concatenate(object_pos)
            object_rot = np.concatenate(object_rot)
            object_velp = np.concatenate(object_velp)
            object_velr = np.concatenate(object_velr)
            object_rel_pos = np.concatenate(object_rel_pos)
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
            # stick_pos = stick_rot = stick_velp = stick_velr = stick_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
            # achieved_goal = self.sim.data.get_site_xpos('object0').copy()
            # achieved_goal = self.sim.data.get_site_xpos('object1').copy()
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])  # dim 40
        # print('grip_pos', grip_pos.shape) # (3,)
        # print('object_pos', object_pos.shape) # (6,)
        # print('object_rel_pos', object_rel_pos.shape) # (6,)
        # print('object_rot', object_rot.shape) # (6,)
        # print('gripper_state', gripper_state.shape) # (2,)
        # print('object_velp', object_velp.shape) # (6,)
        # print('object_velr', object_velr.shape) # (6,)
        # print('grip_velp', grip_velp.shape) # (3,)
        # print('gripper_vel', gripper_vel.shape) # (2,)

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        # TODO: randomize mocap_pos
        if self.random_gripper:
            mocap_pos = np.concatenate([self.np_random.uniform([1.19, 0.6], [1.49, 0.9]), [0.355]])
            self.sim.data.set_mocap_pos('robot0:mocap', mocap_pos)
            for _ in range(10):
                self.sim.step()
            self._step_callback()

        # Randomize start position of object.
        if self.has_object:
            if self.random_box and self.np_random.uniform() < self.random_ratio:
                object_xpos = self.initial_gripper_xpos[:2]
                stick_xpos = object_xpos.copy()
                while (np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1
                       or abs(object_xpos[0] - self.pos_wall[0]) < self.size_wall[0] + self.size_object[0] or abs(
                        stick_xpos[0] - self.pos_wall[0]) < self.size_wall[0] + self.size_obstacle[0]
                       or (abs(object_xpos[0] - stick_xpos[0]) < self.size_object[0] + self.size_obstacle[0] and abs(
                            object_xpos[1] - stick_xpos[1]) < self.size_object[1] + self.size_obstacle[1])):
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                                                                                         self.obj_range, size=2)
                    stick_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                                        size=2)
                    # stick_xpos = np.concatenate(([1.40], self.initial_gripper_xpos[1:2])) + \
                    #              np.array([2 / 3, 1.0]) * self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            else:
                object_xpos = self.initial_gripper_xpos[:2] + np.asarray([self.obj_range * 0.9, self.obj_range / 2])
                stick_xpos = np.asarray(
                    [self.pos_wall[0] + self.size_wall[0] + self.size_obstacle[0], self.initial_gripper_xpos[1]])
                # stick_xpos = self.initial_gripper_xpos[:2] + np.asarray([self.obj_range / 4, 0])
            # Set the position of box. (two slide joints)
            sim_state = self.sim.get_state()
            box_jointx_i = self.sim.model.get_joint_qpos_addr("object0:slidex")
            box_jointy_i = self.sim.model.get_joint_qpos_addr("object0:slidey")
            sim_state.qpos[box_jointx_i] = object_xpos[0]
            sim_state.qpos[box_jointy_i] = object_xpos[1]
            self.sim.set_state(sim_state)
            # Set the position of obstacle. (free joint)
            stick_qpos = self.sim.data.get_joint_qpos('object1:joint')
            assert stick_qpos.shape == (7,)
            stick_qpos[:2] = stick_xpos
            self.sim.data.set_joint_qpos('object1:joint', stick_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = np.concatenate([self.initial_gripper_xpos[:3] + self.target_offset +
                                   self.np_random.uniform(-self.target_range, self.target_range, size=3) for _ in range(self.n_object)])
            goal[2] = self.sim.data.get_site_xpos('object0')[2]
            goal[5] = self.sim.data.get_site_xpos('object1')[2]

            # goal = np.concatenate(([1.40], self.initial_gripper_xpos[1:3])) + np.array(
            #     [2 / 3, 1.0, 1.0]) * self.np_random.uniform(-self.target_range, self.target_range, size=3)
            # goal = self.initial_gripper_xpos[:3] + self.target_offset + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            # goal[2] = self.sim.data.get_site_xpos('object1')[2]
            if not hasattr(self, 'size_wall'):
                self.size_wall = self.sim.model.geom_size[self.sim.model.geom_name2id('wall0')]
            if not hasattr(self, 'size_object'):
                self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]
            if not hasattr(self, 'pos_wall'):
                self.pos_wall = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall0')]
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal[:3] - sites_offset[0]
        site_id = self.sim.model.site_name2id('target1')
        self.sim.model.site_pos[site_id] = self.goal[3:6] - sites_offset[1]
        self.sim.forward()

class FetchPushWallObstacleEnv_curricula(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', penaltize_height=False, heavy_obstacle=False, random_gripper=False,
                 hard_case_ratio=0.0):
        if heavy_obstacle:
            XML_PATH = MODEL_XML_PATH2
        else:
            XML_PATH = MODEL_XML_PATH
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.2, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.35, 0.75, 0.4, 1., 0., 0., 0.],
        }
        self.n_object = sum([('object' in item) for item in initial_qpos.keys()])
        self.penaltize_height = penaltize_height
        self.random_gripper = random_gripper
        self.hard_case_ratio = hard_case_ratio
        fetch_env.FetchEnv.__init__(
            self, XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
        self.pos_wall = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall0')]
        self.size_wall = self.sim.model.geom_size[self.sim.model.geom_name2id('wall0')]
        self.size_obstacle = self.sim.model.geom_size[self.sim.model.geom_name2id('object1')]
        self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = robot_utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = [self.sim.data.get_site_xpos('object' + str(i)) for i in range(self.n_object)]
            # rotations
            object_rot = [rotations.mat2euler(self.sim.data.get_site_xmat('object' + str(i))) for i in
                          range(self.n_object)]
            # velocities
            object_velp = [self.sim.data.get_site_xvelp('object' + str(i)) * dt for i in range(self.n_object)]
            object_velr = [self.sim.data.get_site_xvelr('object' + str(i)) * dt for i in range(self.n_object)]
            # gripper state
            object_rel_pos = [pos - grip_pos for pos in object_pos]
            object_velp = [velp - grip_velp for velp in object_velp]

            object_pos = np.concatenate(object_pos)
            object_rot = np.concatenate(object_rot)
            object_velp = np.concatenate(object_velp)
            object_velr = np.concatenate(object_velr)
            object_rel_pos = np.concatenate(object_rel_pos)
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
            # stick_pos = stick_rot = stick_velp = stick_velr = stick_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = self.sim.data.get_site_xpos('object0').copy()
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])  # dim 40
        # print('grip_pos', grip_pos.shape) # (3,)
        # print('object_pos', object_pos.shape) # (6,)
        # print('object_rel_pos', object_rel_pos.shape) # (6,)
        # print('object_rot', object_rot.shape) # (6,)
        # print('gripper_state', gripper_state.shape) # (2,)
        # print('object_velp', object_velp.shape) # (6,)
        # print('object_velr', object_velr.shape) # (6,)
        # print('grip_velp', grip_velp.shape) # (3,)
        # print('gripper_vel', gripper_vel.shape) # (2,)

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        # TODO: randomize mocap_pos
        if self.random_gripper:
            mocap_pos = np.concatenate([self.np_random.uniform([1.19, 0.6], [1.49, 0.9]), [0.355]])
            self.sim.data.set_mocap_pos('robot0:mocap', mocap_pos)
            for _ in range(10):
                self.sim.step()
            self._step_callback()

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            stick_xpos = object_xpos.copy()
            while (np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1
                   or abs(object_xpos[0] - self.pos_wall[0]) < self.size_wall[0] + self.size_object[0] or abs(
                    stick_xpos[0] - self.pos_wall[0]) < self.size_wall[0] + self.size_obstacle[0]
                   or (abs(object_xpos[0] - stick_xpos[0]) < self.size_object[0] + self.size_obstacle[0] and abs(
                        object_xpos[1] - stick_xpos[1]) < self.size_object[1] + self.size_obstacle[1])):
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                                                                                     self.obj_range, size=2)
                if self.np_random.uniform() > self.hard_case_ratio:
                    # Easy case.
                    stick_xpos[0] = self.initial_gripper_xpos[0] + self.np_random.uniform(-self.obj_range, self.obj_range, size=1)
                    _temp = self.np_random.uniform(0, 1, size=1)
                    if _temp < 0.5:
                        stick_xpos[1] = 0.75 - self.obj_range +_temp / 0.5 * (2 / 3 * self.obj_range)
                    else:
                        stick_xpos[1] = 0.75 + 1/3 * self.obj_range + (_temp - 0.5) / 0.5 * 2/3 * self.obj_range
                else:
                    # Hard case.
                    stick_xpos = np.concatenate([self.initial_gripper_xpos[0:1], [0.75]]) + np.array(
                        [1.0, 1 / 3]) * self.np_random.uniform(-self.obj_range, self.obj_range, size=2)

            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            stick_qpos = self.sim.data.get_joint_qpos('object1:joint')
            assert object_qpos.shape == (7,)
            assert stick_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            stick_qpos[:2] = stick_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)
            self.sim.data.set_joint_qpos('object1:joint', stick_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.target_offset + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal[2] = self.height_offset
            if not hasattr(self, 'size_wall'):
                self.size_wall = self.sim.model.geom_size[self.sim.model.geom_name2id('wall0')]
            if not hasattr(self, 'size_object'):
                self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]
            if not hasattr(self, 'pos_wall'):
                self.pos_wall = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall0')]
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
            'is_blocked': obs['observation'][7] + self.size_obstacle[1] * np.cos(obs['observation'][22]) > 0.85
                          and obs['observation'][7] - self.size_obstacle[1] * np.cos(obs['observation'][22]) < 0.65
                          and abs(obs['observation'][6] - self.pos_wall[0]) < self.size_wall[0] + self.size_obstacle[
                0] + self.size_object[0]
            # and (obs['achieved_goal'][0] - self.pos_wall[0]) * (obs['desired_goal'][0] - self.pos_wall[0]) < 0

        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

class FetchPushWallObstacleEnv_v4(fetch_env.FetchEnv, utils.EzPickle):
    '''
    Universal pushing. Concatenate a one-hot array to the goal space which indicates which object should be moved to the specific location.
    TODO: modidy sample_goal, compute reward. Side remark, in her wrapper, we should also modify observation when resampling goal
    '''
    def __init__(self, reward_type='sparse', penaltize_height=False, heavy_obstacle=False, random_box=True,
                 random_ratio=1.0, hack_obstacle=False, random_gripper=False):
        if heavy_obstacle:
            XML_PATH = MODEL_XML_PATH2
        else:
            XML_PATH = MODEL_XML_PATH
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            # 'object0:joint': [1.2, 0.53, 0.4, 1., 0., 0., 0.],
            'object0:slidex': 0.0,
            'object0:slidey': 0.0,
            'object1:joint': [1.35, 0.75, 0.4, 1., 0., 0., 0.],
        }
        self.n_object = 2
        self.penaltize_height = penaltize_height
        self.random_box = random_box
        self.random_ratio = random_ratio
        self.hack_obstacle = hack_obstacle
        self.random_gripper = random_gripper
        fetch_env.FetchEnv.__init__(
            self, XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
        self.pos_wall = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall0')]
        self.size_wall = self.sim.model.geom_size[self.sim.model.geom_name2id('wall0')]
        self.size_obstacle = self.sim.model.geom_size[self.sim.model.geom_name2id('object1')]
        self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = robot_utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = [self.sim.data.get_site_xpos('object' + str(i)) for i in range(self.n_object)]
            # rotations
            object_rot = [rotations.mat2euler(self.sim.data.get_site_xmat('object' + str(i))) for i in
                          range(self.n_object)]
            # velocities
            object_velp = [self.sim.data.get_site_xvelp('object' + str(i)) * dt for i in range(self.n_object)]
            object_velr = [self.sim.data.get_site_xvelr('object' + str(i)) * dt for i in range(self.n_object)]
            # gripper state
            object_rel_pos = [pos - grip_pos for pos in object_pos]
            object_velp = [velp - grip_velp for velp in object_velp]

            object_pos = np.concatenate(object_pos)
            object_rot = np.concatenate(object_rot)
            object_velp = np.concatenate(object_velp)
            object_velr = np.concatenate(object_velr)
            object_rel_pos = np.concatenate(object_rel_pos)
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
            # stick_pos = stick_rot = stick_velp = stick_velr = stick_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            one_hot = self.goal[3:]
            idx = np.argmax(one_hot)
            achieved_goal = np.concatenate([self.sim.data.get_site_xpos('object' + str(idx)).copy(), one_hot])
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])  # dim 40
        # print('grip_pos', grip_pos.shape) # (3,)
        # print('object_pos', object_pos.shape) # (6,)
        # print('object_rel_pos', object_rel_pos.shape) # (6,)
        # print('object_rot', object_rot.shape) # (6,)
        # print('gripper_state', gripper_state.shape) # (2,)
        # print('object_velp', object_velp.shape) # (6,)
        # print('object_velr', object_velr.shape) # (6,)
        # print('grip_velp', grip_velp.shape) # (3,)
        # print('gripper_vel', gripper_vel.shape) # (2,)

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def get_obs(self):
        return self._get_obs()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        # TODO: randomize mocap_pos
        if self.random_gripper:
            mocap_pos = np.concatenate([self.np_random.uniform([1.19, 0.6], [1.49, 0.9]), [0.355]])
            self.sim.data.set_mocap_pos('robot0:mocap', mocap_pos)
            for _ in range(10):
                self.sim.step()
            self._step_callback()

        # Randomize start position of object.
        if self.has_object:
            if self.random_box and self.np_random.uniform() < self.random_ratio:
                object_xpos = self.initial_gripper_xpos[:2]
                stick_xpos = object_xpos.copy()
                while (np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1
                       or abs(object_xpos[0] - self.pos_wall[0]) < self.size_wall[0] + self.size_object[0] or abs(
                        stick_xpos[0] - self.pos_wall[0]) < self.size_wall[0] + self.size_obstacle[0]
                       or (abs(object_xpos[0] - stick_xpos[0]) < self.size_object[0] + self.size_obstacle[0] and abs(
                            object_xpos[1] - stick_xpos[1]) < self.size_object[1] + self.size_obstacle[1])):
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                                                                                         self.obj_range, size=2)
                    stick_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                                        size=2)
                    # stick_xpos = np.concatenate(([1.40], self.initial_gripper_xpos[1:2])) + \
                    #              np.array([2 / 3, 1.0]) * self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            else:
                object_xpos = self.initial_gripper_xpos[:2] + np.asarray([self.obj_range * 0.9, self.obj_range / 2])
                stick_xpos = np.asarray(
                    [self.pos_wall[0] + self.size_wall[0] + self.size_obstacle[0], self.initial_gripper_xpos[1]])
                # stick_xpos = self.initial_gripper_xpos[:2] + np.asarray([self.obj_range / 4, 0])
            # Set the position of box. (two slide joints)
            sim_state = self.sim.get_state()
            box_jointx_i = self.sim.model.get_joint_qpos_addr("object0:slidex")
            box_jointy_i = self.sim.model.get_joint_qpos_addr("object0:slidey")
            sim_state.qpos[box_jointx_i] = object_xpos[0]
            sim_state.qpos[box_jointy_i] = object_xpos[1]
            self.sim.set_state(sim_state)
            # Set the position of obstacle. (free joint)
            stick_qpos = self.sim.data.get_joint_qpos('object1:joint')
            assert stick_qpos.shape == (7,)
            stick_qpos[:2] = stick_xpos
            self.sim.data.set_joint_qpos('object1:joint', stick_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            # goal = np.concatenate([self.initial_gripper_xpos[:3] + self.target_offset +
            #                    self.np_random.uniform(-self.target_range, self.target_range, size=3) for _ in range(self.n_object)])
            # goal[2] = self.height_offset
            # goal[5] = self.sim.data.get_site_xpos('object1')[2]

            # goal = np.concatenate(([1.40], self.initial_gripper_xpos[1:3])) + np.array(
            #     [2 / 3, 1.0, 1.0]) * self.np_random.uniform(-self.target_range, self.target_range, size=3)
            # which object, 0: box, 1: obstacle
            g_idx = np.random.randint(2)
            one_hot = np.zeros(2)
            one_hot[g_idx] = 1
            goal = self.initial_gripper_xpos[:3] + self.target_offset + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            if g_idx == 0:
                goal[2] = self.sim.data.get_site_xpos('object0')[2]
            else:
                goal[2] = self.sim.data.get_site_xpos('object1')[2]
            goal = np.concatenate([goal, one_hot])
            if not hasattr(self, 'size_wall'):
                self.size_wall = self.sim.model.geom_size[self.sim.model.geom_name2id('wall0')]
            if not hasattr(self, 'size_object'):
                self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]
            if not hasattr(self, 'pos_wall'):
                self.pos_wall = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall0')]
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def compute_reward(self, observation, goal, info):
        # Note: the input is different from other environments.
        one_hot = goal[3:]
        idx = np.argmax(one_hot)
        # HACK: parse the corresponding object position from observation
        achieved_goal = observation[3 + 3 * idx : 3 + 3 * (idx + 1)]
        r = fetch_env.FetchEnv.compute_reward(self, achieved_goal, goal[0:3], info)
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
            # 'is_blocked': obs['observation'][7] + self.size_obstacle[1] * np.cos(obs['observation'][22]) > 0.85
            #               and obs['observation'][7] - self.size_obstacle[1] * np.cos(obs['observation'][22]) < 0.65
            #               and abs(obs['observation'][6] - self.pos_wall[0]) < self.size_wall[0] + self.size_obstacle[
            #     0] + self.size_object[0]
            # # and (obs['achieved_goal'][0] - self.pos_wall[0]) * (obs['desired_goal'][0] - self.pos_wall[0]) < 0

        }
        reward = self.compute_reward(obs['observation'], self.goal, info)
        return obs, reward, done, info

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal[:3] - sites_offset[0]
        self.sim.forward()

class FetchPushWallObstacleEnv_v5(fetch_env.FetchEnv, utils.EzPickle):
    '''
    Universal pushing. Concatenate a one-hot array to the goal space which indicates which object should be moved to the specific location.
    TODO: modidy sample_goal, compute reward. Side remark, in her wrapper, we should also modify observation when resampling goal
    '''
    def __init__(self, reward_type='sparse', penaltize_height=False, heavy_obstacle=False, random_box=True,
                 random_ratio=1.0, hack_obstacle=False, random_gripper=False):
        XML_PATH = MODEL_XML_PATH4
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:slidex': 0.0,
            'object0:slidey': 0.0,
            # 'object1:joint': [1.35, 0.75, 0.4, 1., 0., 0., 0.],
            'object1:slidex': 0.0,
            'object1:slidey': 0.0,
        }
        self.n_object = 2
        self.penaltize_height = penaltize_height
        self.random_box = random_box
        self.random_ratio = random_ratio
        self.hack_obstacle = hack_obstacle
        self.random_gripper = random_gripper
        fetch_env.FetchEnv.__init__(
            self, XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
        self.pos_wall = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall0')]
        self.size_wall = self.sim.model.geom_size[self.sim.model.geom_name2id('wall0')]
        self.size_obstacle = self.sim.model.geom_size[self.sim.model.geom_name2id('object1')]
        self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = robot_utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = [self.sim.data.get_site_xpos('object' + str(i)) for i in range(self.n_object)]
            # rotations
            object_rot = [rotations.mat2euler(self.sim.data.get_site_xmat('object' + str(i))) for i in
                          range(self.n_object)]
            # velocities
            object_velp = [self.sim.data.get_site_xvelp('object' + str(i)) * dt for i in range(self.n_object)]
            object_velr = [self.sim.data.get_site_xvelr('object' + str(i)) * dt for i in range(self.n_object)]
            # gripper state
            object_rel_pos = [pos - grip_pos for pos in object_pos]
            object_velp = [velp - grip_velp for velp in object_velp]

            object_pos = np.concatenate(object_pos)
            object_rot = np.concatenate(object_rot)
            object_velp = np.concatenate(object_velp)
            object_velr = np.concatenate(object_velr)
            object_rel_pos = np.concatenate(object_rel_pos)
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
            # stick_pos = stick_rot = stick_velp = stick_velr = stick_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            one_hot = self.goal[3:]
            idx = np.argmax(one_hot)
            achieved_goal = np.concatenate([self.sim.data.get_site_xpos('object' + str(idx)).copy(), one_hot])
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])  # dim 40
        # print('grip_pos', grip_pos.shape) # (3,)
        # print('object_pos', object_pos.shape) # (6,)
        # print('object_rel_pos', object_rel_pos.shape) # (6,)
        # print('object_rot', object_rot.shape) # (6,)
        # print('gripper_state', gripper_state.shape) # (2,)
        # print('object_velp', object_velp.shape) # (6,)
        # print('object_velr', object_velr.shape) # (6,)
        # print('grip_velp', grip_velp.shape) # (3,)
        # print('gripper_vel', gripper_vel.shape) # (2,)

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def get_obs(self):
        return self._get_obs()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        # TODO: randomize mocap_pos
        if self.random_gripper:
            mocap_pos = np.concatenate([self.np_random.uniform([1.19, 0.6], [1.49, 0.9]), [0.355]])
            self.sim.data.set_mocap_pos('robot0:mocap', mocap_pos)
            for _ in range(10):
                self.sim.step()
            self._step_callback()

        # Randomize start position of object.
        if self.has_object:
            if self.random_box and self.np_random.uniform() < self.random_ratio:
                object_xpos = self.initial_gripper_xpos[:2]
                stick_xpos = object_xpos.copy()
                while (np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1
                       or abs(object_xpos[0] - self.pos_wall[0]) < self.size_wall[0] + self.size_object[0] or abs(
                        stick_xpos[0] - self.pos_wall[0]) < self.size_wall[0] + self.size_obstacle[0]
                       or (abs(object_xpos[0] - stick_xpos[0]) < self.size_object[0] + self.size_obstacle[0] and abs(
                            object_xpos[1] - stick_xpos[1]) < self.size_object[1] + self.size_obstacle[1])):
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                                                                                         self.obj_range, size=2)
                    stick_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                                        size=2)
                    # stick_xpos = np.concatenate(([1.40], self.initial_gripper_xpos[1:2])) + \
                    #              np.array([2 / 3, 1.0]) * self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            else:
                object_xpos = self.initial_gripper_xpos[:2] + np.asarray([self.obj_range * 0.9, self.obj_range / 2])
                stick_xpos = np.asarray(
                    [self.pos_wall[0] + self.size_wall[0] + self.size_obstacle[0], self.initial_gripper_xpos[1]])
                # stick_xpos = self.initial_gripper_xpos[:2] + np.asarray([self.obj_range / 4, 0])
            # Set the position of box and obstacle. (two slide joints)
            sim_state = self.sim.get_state()
            box_jointx_i = self.sim.model.get_joint_qpos_addr("object0:slidex")
            box_jointy_i = self.sim.model.get_joint_qpos_addr("object0:slidey")
            obstacle_jointx_i = self.sim.model.get_joint_qpos_addr("object1:slidex")
            obstacle_jointy_i = self.sim.model.get_joint_qpos_addr("object1:slidey")
            sim_state.qpos[box_jointx_i] = object_xpos[0]
            sim_state.qpos[box_jointy_i] = object_xpos[1]
            sim_state.qpos[obstacle_jointx_i] = stick_xpos[0]
            sim_state.qpos[obstacle_jointy_i] = stick_xpos[1]
            self.sim.set_state(sim_state)
            # # Set the position of obstacle. (free joint)
            # stick_qpos = self.sim.data.get_joint_qpos('object1:joint')
            # assert stick_qpos.shape == (7,)
            # stick_qpos[:2] = stick_xpos
            # self.sim.data.set_joint_qpos('object1:joint', stick_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            # goal = np.concatenate([self.initial_gripper_xpos[:3] + self.target_offset +
            #                    self.np_random.uniform(-self.target_range, self.target_range, size=3) for _ in range(self.n_object)])
            # goal[2] = self.height_offset
            # goal[5] = self.sim.data.get_site_xpos('object1')[2]

            # goal = np.concatenate(([1.40], self.initial_gripper_xpos[1:3])) + np.array(
            #     [2 / 3, 1.0, 1.0]) * self.np_random.uniform(-self.target_range, self.target_range, size=3)
            # which object, 0: box, 1: obstacle
            g_idx = np.random.randint(2)
            one_hot = np.zeros(2)
            one_hot[g_idx] = 1
            goal = self.initial_gripper_xpos[:3] + self.target_offset + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            if g_idx == 0:
                goal[2] = self.sim.data.get_site_xpos('object0')[2]
            else:
                goal[2] = self.sim.data.get_site_xpos('object1')[2]
            goal = np.concatenate([goal, one_hot])
            if not hasattr(self, 'size_wall'):
                self.size_wall = self.sim.model.geom_size[self.sim.model.geom_name2id('wall0')]
            if not hasattr(self, 'size_object'):
                self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]
            if not hasattr(self, 'pos_wall'):
                self.pos_wall = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall0')]
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def compute_reward(self, observation, goal, info):
        # Note: the input is different from other environments.
        one_hot = goal[3:]
        idx = np.argmax(one_hot)
        # HACK: parse the corresponding object position from observation
        achieved_goal = observation[3 + 3 * idx : 3 + 3 * (idx + 1)]
        r = fetch_env.FetchEnv.compute_reward(self, achieved_goal, goal[0:3], info)
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
            # 'is_blocked': obs['observation'][7] + self.size_obstacle[1] * np.cos(obs['observation'][22]) > 0.85
            #               and obs['observation'][7] - self.size_obstacle[1] * np.cos(obs['observation'][22]) < 0.65
            #               and abs(obs['observation'][6] - self.pos_wall[0]) < self.size_wall[0] + self.size_obstacle[
            #     0] + self.size_object[0]
            # # and (obs['achieved_goal'][0] - self.pos_wall[0]) * (obs['desired_goal'][0] - self.pos_wall[0]) < 0

        }
        reward = self.compute_reward(obs['observation'], self.goal, info)
        return obs, reward, done, info

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal[:3] - sites_offset[0]
        self.sim.forward()

import os
from gym import utils
from gym.envs.robotics import fetch_env, rotations
import gym.envs.robotics.utils as robot_utils
import numpy as np

MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'fetch', 'push_wall_heavy_double_obstacle.xml')

class FetchPushWallDoubleObstacleEnv(fetch_env.FetchEnv, utils.EzPickle):
    '''
    Universal pushing. Concatenate a one-hot array to the goal space which indicates which object should be moved to the specific location.
    TODO: modidy sample_goal, compute reward. Side remark, in her wrapper, we should also modify observation when resampling goal
    '''
    def __init__(self, reward_type='sparse', penaltize_height=False, heavy_obstacle=True, random_box=True,
                 random_ratio=1.0, hack_obstacle=False, random_gripper=False):
        XML_PATH = MODEL_XML_PATH
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            # 'object0:joint': [1.2, 0.53, 0.4, 1., 0., 0., 0.],
            'object0:slidex': 0.0,
            'object0:slidey': 0.0,
            'object1:joint': [1.4, 0.47, 0.4, 1., 0., 0., 0.],
            # 'object1:slidex': 0.0,
            # 'object1:slidey': 0.0,
            'object2:joint': [1.4, 0.6, 0.4, 1., 0., 0., 0.],
            # 'object2:slidex': 0.0,
            # 'object2:slidey': 0.0,
        }
        self.n_object = 3
        self.penaltize_height = penaltize_height
        self.random_box = random_box
        self.random_ratio = random_ratio
        self.hack_obstacle = hack_obstacle
        self.random_gripper = random_gripper
        fetch_env.FetchEnv.__init__(
            self, XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.28, target_range=0.28, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
        self.pos_wall0 = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall0')]
        self.pos_wall2 = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall2')]
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

        def config_valid(object_xpos, obstacle1_xpos, obstacle2_xpos):
            if np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) >= 0.1 \
                    and abs(object_xpos[1] - self.pos_wall0[1]) >= self.size_object[1] + self.size_wall[1] \
                    and abs(object_xpos[1] - self.pos_wall2[1]) >= self.size_object[1] + self.size_wall[1] \
                    and abs(obstacle1_xpos[1] - self.pos_wall0[1]) >= self.size_obstacle[1] + self.size_wall[1] \
                    and abs(obstacle1_xpos[1] - self.pos_wall2[1]) >= self.size_obstacle[1] + self.size_wall[1] \
                    and abs(obstacle2_xpos[1] - self.pos_wall0[1]) >= self.size_obstacle[1] + self.size_wall[1] \
                    and abs(obstacle2_xpos[1] - self.pos_wall2[1]) >= self.size_obstacle[1] + self.size_wall[1] \
                    and (abs(object_xpos[0] - obstacle1_xpos[0]) >= self.size_object[0] + self.size_obstacle[0] or abs(object_xpos[1] - obstacle1_xpos[1]) >= self.size_object[1] + self.size_obstacle[1]) \
                    and (abs(object_xpos[0] - obstacle2_xpos[0]) >= self.size_object[0] + self.size_obstacle[0] or abs(object_xpos[1] - obstacle2_xpos[1]) >= self.size_object[1] + self.size_obstacle[1]) \
                    and (abs(obstacle1_xpos[0] - obstacle2_xpos[0]) >= self.size_obstacle[0] * 2 or abs(obstacle1_xpos[1] - obstacle2_xpos[1]) >= self.size_obstacle[1] * 2):
                return True
            else:
                return False
        # Randomize start position of object.
        if self.has_object:
            if self.random_box and self.np_random.uniform() < self.random_ratio:
                self.sample_hard = False
                object_xpos = self.initial_gripper_xpos[:2]
                stick1_xpos = object_xpos.copy()
                stick2_xpos = object_xpos.copy()
                while not config_valid(object_xpos, stick1_xpos, stick2_xpos):
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                    stick1_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                    stick2_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            else:
                self.sample_hard = True
                stick1_xpos = np.array([1.3, self.pos_wall0[1] - self.size_wall[1] - self.size_obstacle[1]])
                stick2_xpos = np.array([1.3, self.pos_wall2[1] - self.size_wall[1] - self.size_obstacle[1]])
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                while object_xpos[1] > stick1_xpos[1]:
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            # Set the position of box. (two slide joints)
            sim_state = self.sim.get_state()
            box_jointx_i = self.sim.model.get_joint_qpos_addr("object0:slidex")
            box_jointy_i = self.sim.model.get_joint_qpos_addr("object0:slidey")
            # obstacle1_jointx_i = self.sim.model.get_joint_qpos_addr("object1:slidex")
            # obstacle1_jointy_i = self.sim.model.get_joint_qpos_addr("object1:slidey")
            # obstacle2_jointx_i = self.sim.model.get_joint_qpos_addr("object2:slidex")
            # obstacle2_jointy_i = self.sim.model.get_joint_qpos_addr("object2:slidey")
            sim_state.qpos[box_jointx_i] = object_xpos[0]
            sim_state.qpos[box_jointy_i] = object_xpos[1]
            # sim_state.qpos[obstacle1_jointx_i] = stick1_xpos[0]
            # sim_state.qpos[obstacle1_jointy_i] = stick1_xpos[1]
            # sim_state.qpos[obstacle2_jointx_i] = stick2_xpos[0]
            # sim_state.qpos[obstacle2_jointy_i] = stick2_xpos[1]
            self.sim.set_state(sim_state)
            # Set the position of obstacle. (free joint)
            stick1_qpos = self.sim.data.get_joint_qpos('object1:joint')
            stick2_qpos = self.sim.data.get_joint_qpos('object2:joint')
            assert stick1_qpos.shape == (7,)
            assert stick2_qpos.shape == (7,)
            stick1_qpos[:2] = stick1_xpos
            stick2_qpos[:2] = stick2_xpos
            self.sim.data.set_joint_qpos('object1:joint', stick1_qpos)
            self.sim.data.set_joint_qpos('object2:joint', stick2_qpos)

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
        if self.has_object:
            # goal = np.concatenate([self.initial_gripper_xpos[:3] + self.target_offset +
            #                    self.np_random.uniform(-self.target_range, self.target_range, size=3) for _ in range(self.n_object)])
            # goal[2] = self.height_offset
            # goal[5] = self.sim.data.get_site_xpos('object1')[2]

            # goal = np.concatenate(([1.40], self.initial_gripper_xpos[1:3])) + np.array(
            #     [2 / 3, 1.0, 1.0]) * self.np_random.uniform(-self.target_range, self.target_range, size=3)
            # which object, 0: box, 1: obstacle
            g_idx = np.random.randint(self.n_object)
            one_hot = np.zeros(self.n_object)
            one_hot[g_idx] = 1
            goal = self.initial_gripper_xpos[:3] + self.target_offset + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            if hasattr(self, 'sample_hard') and self.sample_hard and g_idx == 0:
                while goal[1] < self.pos_wall2[1]:
                    goal = self.initial_gripper_xpos[:3] + self.target_offset + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal[2] = self.sim.data.get_site_xpos('object' + str(g_idx))[2]
            goal = np.concatenate([goal, one_hot])
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

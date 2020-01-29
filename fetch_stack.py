import os
import numpy as np
from gym import utils
from gym.envs.robotics import fetch_env, rotations
import gym.envs.robotics.utils as robotics_utils


MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'fetch', 'pick_and_place_stack.xml')


class FetchStackEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', random_gripper=True, random_box=True, random_ratio=1.0):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.30, 0.53, 0.4, 1., 0., 0., 0.],
            # 'object2:joint': [1.25, 0.58, 0.4, 1., 0., 0., 0.],
            # 'object3:joint': [1.30, 0.58, 0.4, 1., 0., 0., 0.],
        }
        self.random_gripper = random_gripper
        self.random_box = random_box
        self.random_ratio = random_ratio
        self.n_object = 2
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
        self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]
        self.size_obstacle = np.array([0.15, 0.15, 0.15])
        self.task_mode = 0 # 0: pick and place, 1: stack

    def _get_obs(self):
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = robotics_utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = [self.sim.data.get_site_xpos('object' + str(i)) for i in range(self.current_nobject)] \
                         + [np.zeros(3) for _ in range(self.current_nobject, self.n_object)]
            # rotations
            object_rot = [rotations.mat2euler(self.sim.data.get_site_xmat('object' + str(i))) for i in range(self.current_nobject)] \
                         + [np.zeros(3) for _ in range(self.current_nobject, self.n_object)]
            # velocities
            object_velp = [self.sim.data.get_site_xvelp('object' + str(i)) * dt for i in range(self.current_nobject)] \
                          + [np.zeros(3) for _ in range(self.current_nobject, self.n_object)]
            object_velr = [self.sim.data.get_site_xvelr('object' + str(i)) * dt for i in range(self.current_nobject)] \
                          + [np.zeros(3) for _ in range(self.current_nobject, self.n_object)]
            # gripper state
            # object_rel_pos = [pos - grip_pos for pos in object_pos]
            object_rel_pos = [object_pos[i] - grip_pos for i in range(self.current_nobject)] \
                             + [np.zeros(3) for _ in range(self.current_nobject, self.n_object)]
            # object_velp = [velp - grip_velp for velp in object_velp]
            object_velp = [object_velp[i] - grip_velp for i in range(self.current_nobject)] \
                          + [np.zeros(3) for _ in range(self.current_nobject, self.n_object)]

            object_pos = np.concatenate(object_pos)
            object_rot = np.concatenate(object_rot)
            object_velp = np.concatenate(object_velp)
            object_velr = np.concatenate(object_velr)
            object_rel_pos = np.concatenate(object_rel_pos)
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            one_hot = self.goal[3:]
            idx = np.argmax(one_hot)
            achieved_goal = np.concatenate([object_pos[3 * idx: 3 * (idx + 1)], one_hot])
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel, [self.task_mode],
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def get_obs(self):
        # print('in get_obs, goal', self.goal)
        return self._get_obs()

    def set_goal(self, goal):
        self.goal = goal.copy()

    def switch_obs_goal(self, obs, goal):
        obs = obs.copy()
        if isinstance(obs, dict):
            goal_idx = np.argmax(goal[3:])
            obs['achieved_goal'] = np.concatenate([obs['observation'][3 + 3 * goal_idx: 3 + 3 * (goal_idx + 1)], goal[3:]])
            obs['desired_goal'] = goal
        elif isinstance(obs, np.ndarray):
            goal_idx = np.argmax(goal[3:])
            obs_dim = self.observation_space['observation'].shape[0]
            goal_dim = self.observation_space['achieved_goal'].shape[0]
            obs[obs_dim:obs_dim+3] = obs[3 + goal_idx * 3: 3 + (goal_idx + 1) * 3]
            obs[obs_dim+3:obs_dim+goal_dim] = goal[3:]
            obs[obs_dim+goal_dim:obs_dim+goal_dim*2] = goal[:]
        else:
            raise TypeError
        return obs

    def get_state(self):
        return self.sim.get_state()

    def set_state(self, state):
        self.sim.set_state(state)
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        if self.random_gripper:
            mocap_pos = np.concatenate([self.np_random.uniform([1.15, 0.6], [1.45, 0.9]), [0.355]])
            self.sim.data.set_mocap_pos('robot0:mocap', mocap_pos)
            for _ in range(10):
                self.sim.step()
            self._step_callback()

        def is_valid(objects_xpos):
            for id1 in range(len(objects_xpos)):
                for id2 in range(id1 + 1, len(objects_xpos)):
                    if abs(objects_xpos[id1][0] - objects_xpos[id2][0]) < 2 * self.size_object[0] and \
                                    abs(objects_xpos[id1][1] - objects_xpos[id2][1]) < 2 * self.size_object[1]:
                        return False
            return True

        # Randomize start position of object.
        if self.has_object:
            self.current_nobject = np.random.randint(0, self.n_object) + 1

            # if self.random_box and self.np_random.uniform() < self.random_ratio:
            if self.random_box:
                objects_xpos = []
                for i in range(self.current_nobject):
                    objects_xpos.append(self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2))
                while not is_valid(objects_xpos):
                    for i in range(self.current_nobject):
                        objects_xpos[i] = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            else:
                raise NotImplementedError

            # Set the position of obstacle. (free joint)
            for i in range(self.n_object):
                object_qpos = self.sim.data.get_joint_qpos('object%d:joint' % i)
                if i < self.current_nobject:
                    object_qpos[:2] = objects_xpos[i]
                    object_qpos[2] = self.height_offset
                else:
                    object_qpos[:3] = np.array([-1, -1, 0])
                self.sim.data.set_joint_qpos('object%d:joint' % i, object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if not hasattr(self, 'size_object'):
            self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]
        if not hasattr(self, 'current_nobject'):
            self.current_nobject = self.n_object
        if self.np_random.uniform() < 0.3:
            self.task_mode = 1
        else:
            self.task_mode = 0
        # self.task_mode = np.random.randint(2)
        g_idx = np.random.randint(self.current_nobject)
        one_hot = np.zeros(self.n_object)
        one_hot[g_idx] = 1
        goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        goal[2] = self.height_offset
        if self.task_mode == 1:
            level = np.random.randint(0, self.n_object)
            goal[2] += self.size_object[2] * 2 * level
        elif self.np_random.uniform() < 0.5:
            # level = np.random.randint(1, self.n_object)
            goal[2] += self.np_random.uniform(0, 0.45)
            # goal[2] += self.size_object[2] * 2 * level
        goal = np.concatenate([goal, one_hot])
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        raise NotImplementedError

    def compute_reward(self, observation, goal, info):
        task_mode = observation[-1]
        one_hot = goal[3:]
        idx = np.argmax(one_hot)
        # parse the corresponding object position from observation
        achieved_goal = observation[3 + 3 * idx: 3 + 3 * (idx + 1)]
        previous_achieved_goal = info['previous_obs']['observation'][3 + 3 * idx: 3 + 3 * (idx + 1)]
        if task_mode == 0:
            r = fetch_env.FetchEnv.compute_reward(self, achieved_goal, goal[0:3], info)
            if self.reward_type == 'dense':
                r = np.linalg.norm(previous_achieved_goal - goal[0:3]) - np.linalg.norm(achieved_goal - goal[0:3])
            success = np.linalg.norm(achieved_goal - goal[0:3]) < self.distance_threshold
            if self.reward_type == 'dense':
                r += success
        else:
            r_achieve = fetch_env.FetchEnv.compute_reward(self, achieved_goal, goal[0:3], info)
            if self.reward_type == 'dense':
                r_achieve = np.linalg.norm(previous_achieved_goal - goal[0:3]) - np.linalg.norm(achieved_goal - goal[0:3])
                if np.linalg.norm(achieved_goal - goal[0:3]) < self.distance_threshold:
                    gripper_far = np.linalg.norm(observation[0:3] - achieved_goal) > self.distance_threshold
                    r = r_achieve
                    success = gripper_far
                else:
                    r = r_achieve
                    success = 0.0
                r += success
            else:
                if r_achieve < -0.5:
                    r = -1.0
                else:
                    # Check if stacked
                    other_objects_pos = np.concatenate([observation[3: 3 + 3 * idx],
                                                        observation[3 + 3 * (idx + 1) : 3 + 3 * self.n_object]])
                    # print('other_objects_pos', other_objects_pos)
                    # print('achieved_goal', achieved_goal)
                    stack = False
                    if achieved_goal[2] < self.height_offset + self.size_object[2]:
                        stack = True
                    # TODO: if two objects serve together as lower part?
                    else:
                        for i in range(self.n_object - 1):
                            if abs(other_objects_pos[3 * i + 2] - (achieved_goal[2] - self.size_object[2] * 2)) < 0.01 \
                                    and abs(other_objects_pos[3 * i] - achieved_goal[0]) < self.size_object[0] \
                                    and abs(other_objects_pos[3 * i + 1] - achieved_goal[1]) < self.size_object[1]:
                                stack = True
                                break
                    gripper_far = np.linalg.norm(observation[0:3] - achieved_goal) > self.distance_threshold
                    # print('stack', stack, 'gripper_far', gripper_far)
                    if stack and gripper_far:
                        r = 0.0
                    else:
                        r = -1.0
                success = r > -0.5
        return r, success

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        previous_obs = self._get_obs()
        info = {'previous_obs': previous_obs, }
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        reward, is_success = self.compute_reward(obs['observation'], self.goal, info)
        info['is_success'] = is_success
        return obs, reward, done, info

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.0
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -30.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        g_idx = np.argmax(self.goal[3:])
        object_id = self.sim.model.site_name2id('object%d' % g_idx)
        self.sim.model.site_pos[site_id] = self.goal[:3] - sites_offset[0]
        self.sim.model.site_rgba[site_id] = np.concatenate([self.sim.model.site_rgba[object_id][:3], [0.5]])
        self.sim.forward()

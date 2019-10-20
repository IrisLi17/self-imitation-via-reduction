import os
from gym import utils
from gym.envs.robotics import fetch_env, rotations
import gym.envs.robotics.utils as robot_utils
import numpy as np


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'fetch', 'push_wall.xml')


class FetchPushWallEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', penaltize_height=False):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        self.penaltize_height = penaltize_height
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    '''
    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = robot_utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp

            # the stick
            stick_pos = self.sim.data.get_site_xpos('object1')
            stick_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object1'))
            stick_velp = self.sim.data.get_site_xvelp('object1') * dt
            stick_velr = self.sim.data.get_site_xvelr('object1') * dt
            stick_rel_pos = stick_pos - grip_pos
            stick_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
            stick_pos = stick_rot = stick_velp = stick_velr = stick_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(), # [0:14]
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel, # [14:25]
            stick_pos.ravel(), stick_rel_pos.ravel(), stick_rot.ravel(), #[25:34]
            stick_velp.ravel(), stick_velr.ravel(), #[34:40]
        ])
        # print('grip_pos', grip_pos.shape) # (3,)
        # print('object_pos', object_pos.shape) # (3,)
        # print('object_rot', object_rot.shape) # (3,)
        # print('gripper_state', gripper_state.shape) # (2,)
        # print('object_velp', object_velp.shape) # (3,)
        # print('object_velr', object_velr.shape) # (3,)
        # print('grip_velp', grip_velp.shape) # (3,)
        # print('gripper_vel', gripper_vel.shape) # (2,)

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }
    '''
    
    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while (np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1
                   or abs(object_xpos[0] - 1.3) < 0.045):
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True
    
    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            while (abs(goal[0] - 1.3) < 0.045):
                goal = self.initial_gripper_xpos[:3] + self.target_offset + self.np_random.uniform(-self.target_range, self.target_range, size=3)
                goal[2] = self.height_offset
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
        '''
        # Add reward_near. Mimic https://github.com/openai/gym/blob/master/gym/envs/mujoco/pusher.py.
        reward += 0.5 * self.compute_reward(obs['observation'][:3], obs['achieved_goal'], info)
        '''
        # Box penalty.
        if self.penaltize_height:
            gripper_height = obs['observation'][2]
            gripper_x = obs['observation'][0]
            gripper_y = obs['observation'][1]
            height_penalty = gripper_height > 0.5 or gripper_height < 0.3
            x_penalty = gripper_x < 1.05 or gripper_x > 1.55
            y_penalty = gripper_y < 0.4 or gripper_y > 1.1
            reward = reward - height_penalty - x_penalty - y_penalty
        return obs, reward, done, info
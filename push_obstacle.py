import os
from gym import utils
from gym.envs.robotics import fetch_env, rotations
import gym.envs.robotics.utils as robot_utils
import numpy as np


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'fetch', 'push_obstacle.xml')


class FetchPushEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.35, 0.75, 0.4, 1., 0., 0., 0.],
        }
        self.n_object = sum([('object' in item) for item in initial_qpos.keys()])
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

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
            # every object should get involved
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(), 
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel, 
        ]) # dim 40
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

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            stick_xpos = object_xpos.copy()
            while (np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1
                   or abs(object_xpos[0] - 1.3) < 0.045 or abs(stick_xpos[0] - 1.3) < 0.045
                   or (abs(object_xpos[0] - stick_xpos[0]) < 0.05 and abs(object_xpos[1] - stick_xpos[1]) < 0.225)):
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                stick_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
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
            # every object should be goal
            goal = []
            for i in range(self.n_object):
                _goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
                _goal += self.target_offset
                _goal[2] = self.height_offset
                goal.append(_goal)
            goal = np.concatenate(goal)            
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()
    
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
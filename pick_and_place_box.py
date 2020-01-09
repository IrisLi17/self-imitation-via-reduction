import os
import numpy as np
from gym import utils
from gym.envs.robotics import fetch_env, rotations
import gym.envs.robotics.utils as robotics_utils


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'fetch', 'pick_and_place_box.xml')


class FetchPickAndPlaceBoxEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', random_gripper=True, random_box=True, random_ratio=1.0):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'cover:joint': [1.25, 0.60, 0.6, 1., 0., 0., 0.],
        }
        self.random_gripper = random_gripper
        self.random_box = random_box
        self.random_ratio = random_ratio
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
        self.pos_cover = self.sim.data.get_joint_qpos('cover:joint')[:3].copy()
        self.pos_box_bottom = self.sim.data.get_geom_xpos('box_bottom')
        self.size_box_cover = self.sim.model.geom_size[self.sim.model.geom_name2id('cover_top')]
        # print('pos_cover', self.pos_cover)
        # print('pos_box_bottom', self.pos_box_bottom)
        # print('size_box_cover', self.size_box_cover)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = robotics_utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = [self.sim.data.get_site_xpos('object0'), self.sim.data.get_geom_xpos('handle')]
            # rotations
            object_rot = [rotations.mat2euler(self.sim.data.get_site_xmat('object0')), rotations.mat2euler(self.sim.data.get_geom_xmat('handle'))]
            # velocities
            object_velp = [self.sim.data.get_site_xvelp('object0') * dt, self.sim.data.get_geom_xvelp('handle') * dt]
            object_velr = [self.sim.data.get_site_xvelr('object0') * dt, self.sim.data.get_geom_xvelr('handle') * dt]
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
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            # TODO
            one_hot = self.goal[3:]
            idx = np.argmax(one_hot)
            achieved_goal = np.concatenate([object_pos[3 * idx: 3 * (idx + 1)], one_hot])
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        # TODO: randomize mocap_pos
        if self.random_gripper:
            mocap_pos = np.concatenate([self.np_random.uniform([1.15, 0.6], [1.45, 0.9]), [0.355]])
            self.sim.data.set_mocap_pos('robot0:mocap', mocap_pos)
            for _ in range(10):
                self.sim.step()
            self._step_callback()

        # Randomize start position of object.
        if self.has_object:
            if self.random_box and self.np_random.uniform() < self.random_ratio:
                self.sample_hard = False
                object_xpos = self.initial_gripper_xpos[:2]
                cover_xpos = self.pos_cover[:2] + self.np_random.uniform(-0.25, 0.25, size=2)
                # cover_xpos += (self.size_box_cover[0] * 2) * np.sign(cover_xpos)
                while (np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1):
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                                                                                         self.obj_range, size=2)
                while (abs(cover_xpos[0] - self.sim.data.get_mocap_pos('robot0:mocap')[0]) < self.size_box_cover[0]
                       and abs(cover_xpos[1] - self.sim.data.get_mocap_pos('robot0:mocap')[1]) < self.size_box_cover[1]):
                    cover_xpos = self.pos_cover[:2] + self.np_random.uniform(-0.25, 0.25, size=2)
            else:
                self.sample_hard = True
                # object_xpos = self.initial_gripper_xpos[:2] + np.asarray([self.obj_range * 0.9, self.obj_range / 2])
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                cover_xpos = self.pos_cover[:2]
                while (np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1
                       or (abs(object_xpos[0] - self.pos_box_bottom[0]) < self.size_box_cover[0])
                       or (abs(object_xpos[1] - self.pos_box_bottom[1]) < self.size_box_cover[1])):
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                # stick_xpos = self.initial_gripper_xpos[:2] + np.asarray([self.obj_range / 4, 0])
            if abs(object_xpos[0] - self.pos_box_bottom[0]) < self.size_box_cover[0] + 0.025 and abs(object_xpos[1] - self.pos_box_bottom[1]) < self.size_box_cover[1] + 0.025:
                object_height = self.pos_box_bottom[2] + 0.026
            else:
                object_height = 0.425
            if abs(cover_xpos[0] - self.pos_box_bottom[0]) < self.size_box_cover[0] * 2 \
                    and abs(cover_xpos[1] - self.pos_box_bottom[1]) < self.size_box_cover[1] * 2:
                cover_height = 0.51
            else:
                cover_height = 0.41
            # Set the position of obstacle. (free joint)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            cover_qpos = self.sim.data.get_joint_qpos('cover:joint')
            assert object_qpos.shape == (7,)
            assert cover_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            object_qpos[2] = object_height
            cover_qpos[:2] = cover_xpos
            cover_qpos[2] = cover_height
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)
            self.sim.data.set_joint_qpos('cover:joint', cover_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        g_idx = np.random.randint(2)
        one_hot = np.zeros(2)
        one_hot[g_idx] = 1
        goal = self.initial_gripper_xpos[:3] + self.target_offset + self.np_random.uniform(-self.target_range,
                                                                                           self.target_range, size=3)
        if hasattr(self, 'sample_hard') and self.sample_hard and g_idx == 0:
            goal = self.pos_box_bottom[:3] + self.np_random.uniform(-self.size_box_cover, self.size_box_cover, size=3)

        if g_idx == 0:
            goal[2] = self.sim.data.get_site_xpos('object0')[2]
        else:
            goal[2] = 0.501

        goal = np.concatenate([goal, one_hot])
        if self.target_in_the_air and self.np_random.uniform() < 0.5:
            goal[2] += self.np_random.uniform(0, 0.45)
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
        }
        reward = self.compute_reward(obs['observation'], self.goal, info)
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
        self.sim.model.site_pos[site_id] = self.goal[:3] - sites_offset[0]
        self.sim.forward()
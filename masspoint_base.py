import numpy as np

from gym.envs.robotics import rotations, robot_env, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class MasspointPushEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps,
        target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type, n_object,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.n_object = n_object

        super(MasspointPushEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=2,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        pass

    def _set_action(self, action):
        assert action.shape == (2,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        # pos_ctrl, gripper_ctrl = action[:3], action[3]

        action *= 0.05
        # pos_ctrl *= 0.05  # limit maximum change in position
        # rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        # gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        # assert gripper_ctrl.shape == (2,)
        # if self.block_gripper:
        #     gripper_ctrl = np.zeros_like(gripper_ctrl)
        # action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        # utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        # grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        masspoint_pos = self.sim.data.get_site_xpos('masspoint')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        # grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        masspoint_velp = self.sim.data.get_site_xvelp('masspoint') * dt
        # robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        object_pos = [self.sim.data.get_site_xpos('object' + str(i)) for i in range(self.n_object)]
        # rotations
        object_rot = [rotations.mat2euler(self.sim.data.get_site_xmat('object' + str(i))) for i in range(self.n_object)]
        # velocities
        object_velp = [self.sim.data.get_site_xvelp('object' + str(i)) * dt for i in range(self.n_object)]
        object_velr = [self.sim.data.get_site_xvelr('object' + str(i)) * dt for i in range(self.n_object)]
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
        achieved_goal = np.concatenate([self.sim.data.get_site_xpos('object' + str(idx)).copy(), one_hot])
        obs = np.concatenate([
            masspoint_pos, object_pos.ravel(), object_rel_pos.ravel(), object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), masspoint_velp,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        # body_id = self.sim.model.body_name2id('robot0:gripper_link')
        # lookat = self.sim.data.body_xpos[body_id]
        lookat = [1.3, 0.75, 0.0]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.0
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -60.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal[:3] - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        object_xpos = self.initial_masspoint_xpos[:2]
        while np.linalg.norm(object_xpos - self.initial_masspoint_xpos[:2]) < 0.1:
            object_xpos = self.initial_masspoint_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if not hasattr(self, 'size_wall'):
            self.size_wall = self.sim.model.geom_size[self.sim.model.geom_name2id('wall0')]
        if not hasattr(self, 'size_object'):
            self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]
        if not hasattr(self, 'pos_wall0'):
            self.pos_wall0 = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall0')]
        if self.n_object > 2 and not hasattr(self, 'pos_wall2'):
            self.pos_wall2 = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall2')]
        g_idx = np.random.randint(self.n_object)
        one_hot = np.zeros(self.n_object)
        one_hot[g_idx] = 1
        goal = self.initial_masspoint_xpos[:2] + self.target_offset + self.np_random.uniform(-self.target_range, self.target_range, size=2)
        if hasattr(self, 'sample_hard') and self.sample_hard and g_idx == 0:
            while goal[1] < self.pos_wall2[1]:
                goal = self.initial_masspoint_xpos[:2] + self.target_offset + self.np_random.uniform(-self.target_range, self.target_range, size=2)
        goal = np.concatenate([goal, self.sim.data.get_site_xpos('object' + str(g_idx))[2:], one_hot])
        if self.target_in_the_air and self.np_random.uniform() < 0.5:
            goal[2] += self.np_random.uniform(0, 0.45)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        # utils.reset_mocap_welds(self.sim)
        self.sim.forward()
        for _ in range(10):
            self.sim.step()
        # Extract information for sampling goals.
        self.initial_masspoint_xpos = self.sim.data.get_site_xpos('masspoint').copy()

    def render(self, mode='human', width=500, height=500):
        return super(MasspointPushEnv, self).render(mode, width, height)

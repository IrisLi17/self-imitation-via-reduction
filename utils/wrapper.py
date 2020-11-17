import gym.spaces
from gym import Env
from utils.torch.serializable import Serializable
import random
import cv2
import numpy as np
import imutils
from gym.spaces import Box, Dict
import torch
import utils.torch.pytorch_util as ptu
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict
import os.path as osp
from utils.torch.asset_loader import load_local_or_remote_file

from multiworld.core.image_env import ImageEnv
from multiworld.envs.pygame.point2d import Point2DWallEnv

import utils.vae_wrapper_util as vae_wrapper_util

class ProxyEnv(Serializable, Env):
    def __init__(self, wrapped_env):
        Serializable.quick_init(self, locals())
        self._wrapped_env = wrapped_env
        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    def log_diagnostics(self, paths, *args, **kwargs):
        if hasattr(self._wrapped_env, 'log_diagnostics'):
            self._wrapped_env.log_diagnostics(paths, *args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()

    def __getattr__(self, attrname):
        if attrname == '_serializable_initialized':
            return None
        return getattr(self._wrapped_env, attrname)

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.wrapped_env)

class DoneOnSuccessWrapper(gym.Wrapper):
    """
    Reset on success and offsets the reward.
    Useful for GoalEnv.
    """
    def __init__(self, env, reward_offset=1.0):
        super(DoneOnSuccessWrapper, self).__init__(env)
        self.reward_offset = reward_offset

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        done = done or info.get('is_success', False)
        reward += self.reward_offset
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self.env.compute_reward(achieved_goal, desired_goal, info)
        return reward + self.reward_offset


class ScaleRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_scale=1.0):
        super(ScaleRewardWrapper, self).__init__(env)
        self.reward_scale = reward_scale

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward /= self.reward_scale
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self.env.compute_reward(achieved_goal, desired_goal, info)
        return reward / self.reward_scale


class FlexibleTimeLimitWrapper(gym.Wrapper):
    '''
    ONLY applicable to Stacking environment!
    We can set max_episode_steps = None for gym, (so gym.TimeLimitWrapper is not applied),
    then use this class to avoid potential conflict.
    '''
    def __init__(self, env, time_limit=None):
        super(FlexibleTimeLimitWrapper, self).__init__(env)
        self.time_limit = time_limit
        assert 'FetchStack' in env.spec.id
        assert env.spec.max_episode_steps is None
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        self.time_limit = self.env.unwrapped.current_nobject * 50 if self.env.unwrapped.current_nobject > 2 else 100
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self.time_limit:
            info['TimeLimit.truncated'] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

# class LatentWrappedEnv(ProxyEnv,Env):
#     """
#     This is the class wraps an image-based environment with
#      some api that is used for the task-reduction
#     """
#     def __init__(
#         self,
#         wrapped_env,
#         sample_from_true_prior=False,
#         representation_size=16,
#         reward_params=None,
#         imsize = 84,
#         obs_size=None,
#         norm_order = 2,
#         noisy_encoding = False,
#         epsilon = 0.8,
#     ):
#         self.quick_init(locals())
#         if reward_params is None:
#             reward_params = dict()
#         super().__init__(wrapped_env)
#         self.representation_size = representation_size
#         self.imsize = imsize
#         self.obs_size = obs_size
#         self.reward_params = reward_params
#         self.reward_type = self.reward_params.get("type", 'latent_distance')
#         self.norm_order = self.reward_params.get("norm_order", norm_order)
#         self.epsilon = self.reward_params.get("epsilon", epsilon)  # for sparse reward
#         latent_space = Box(
#             -10 * np.ones(obs_size or self.representation_size),
#             10 * np.ones(obs_size or self.representation_size),
#             dtype=np.float32,
#         )
#
#         spaces = self.wrapped_env.observation_space.spaces
#         spaces['observation'] = latent_space
#         spaces['desired_goal'] = latent_space
#         spaces['achieved_goal'] = latent_space
#
#         spaces['latent_observation'] = latent_space
#         spaces['latent_observation_mean'] = latent_space
#         spaces['latent_observation_std'] = latent_space
#
#         spaces['latent_desired_goal'] = latent_space
#         spaces['latent_desired_goal_mean'] = latent_space
#         spaces['latent_desired_goal_std'] = latent_space
#
#         spaces['latent_achieved_goal'] = latent_space
#         spaces['latent_achieved_goal_mean'] = latent_space
#         spaces['latent_achieved_goal_std'] = latent_space
#         self.obs = dict()
#         self.observation_space = Dict(spaces)
class LatentWrappedEnv(ProxyEnv,Env):
    """This class wraps an image-based environment with a VAE.
     Assumes you get flattened (channels,84,84) observations from wrapped_env.
     This class adheres to the "Silent Multitask Env" semantics: on reset,
     it resamples a goal.
     """

    def __init__(
            self,
            wrapped_env,
            #vae,
            vae_input_key_prefix='image',
            use_vae_goals=True,
            sample_from_true_prior=False,
            decode_goals=False,
            render_goals=False,
            render_rollouts=False,
            reward_params=None,
            mode="train",
            imsize=84,
            obs_size=None,
            norm_order=2,
            epsilon=0.4,
            temperature=1.0,
            vis_granularity=50,
            presampled_goals=None,
            train_noisy_encoding=False,
            test_noisy_encoding=False,
            noise_type=None,  # DEPRECATED
            num_samples_for_latent_histogram=10000,
            use_reprojection_network=False,
            reprojection_network=None,
            use_vae_dataset=True,
            vae_dataset_path=None,
            clip_encoding_std=True,
            use_replay_buffer_goals=False,  # DEPRECATED FEATURE
            v_func_heatmap_bounds=(-1.5, 0.0),
            reproj_vae=None,
            disable_annotated_images=False,
            regressor=None
    ):
        self.quick_init(locals())
        if reward_params is None:
            reward_params = dict()
        super().__init__(wrapped_env)
        # if type(vae) is str:
        #     self.vae = load_local_or_remote_file(vae)
        # else:
        #     self.vae = vae
        # ptu.set_device(0)
        # self.vae.cuda()
        # set the blue range for detecting the position of the point
        self.lower_range = np.array([110, 50, 50])
        self.upper_range = np.array([130, 255, 255])
        self.noisy_encoding = False
        self.representation_size = 16
        self.regressor = regressor
        # self.representation_size = self.vae.representation_size
        # if hasattr(self.vae, 'input_channels'):
        #     self.input_channels = self.vae.input_channels
        # else:
        #     self.input_channels = None
        self.input_channels = 3
        self.set_goal_times = 0
        self._reconstr_image_observation = False
        self._use_vae_goals = use_vae_goals
        self.sample_from_true_prior = sample_from_true_prior
        self.decode_goals = decode_goals
        self.render_goals = render_goals
        self.render_rollouts = render_rollouts
        self.default_kwargs = dict(
            decode_goals=decode_goals,
            render_goals=render_goals,
            render_rollouts=render_rollouts,
        )

        self.clip_encoding_std = clip_encoding_std

        self.train_noisy_encoding = train_noisy_encoding
        self.test_noisy_encoding = test_noisy_encoding
        self.noise_type = noise_type

        self.imsize = imsize
        self.reward_params = reward_params
        self.reward_type = self.reward_params.get("type", 'latent_distance')
        self.reward_object = self.reward_params.get('object',1)
        self.norm_order = self.reward_params.get("norm_order", norm_order)
        self.epsilon = self.reward_params.get("epsilon", epsilon)  # for sparse reward
        self.temperature = self.reward_params.get("temperature", temperature)  # for exponential reward
        self.reward_min_variance = self.reward_params.get("min_variance", 0)
        latent_space = Box(
            -10 * np.ones(obs_size or self.representation_size),
            10 * np.ones(obs_size or self.representation_size),
            dtype=np.float32,
        )

        spaces = self.wrapped_env.observation_space.spaces
        spaces['observation'] = latent_space
        spaces['desired_goal'] = latent_space
        spaces['achieved_goal'] = latent_space

        # spaces['latent_observation'] = latent_space
        # spaces['latent_observation_mean'] = latent_space
        # spaces['latent_observation_std'] = latent_space

        # spaces['latent_desired_goal'] = latent_space
        # spaces['latent_desired_goal_mean'] = latent_space
        # spaces['latent_desired_goal_std'] = latent_space

        # spaces['latent_achieved_goal'] = latent_space
        # spaces['latent_achieved_goal_mean'] = latent_space
        # spaces['latent_achieved_goal_std'] = latent_space

        self.observation_space = Dict(spaces)

        self.vae_input_key_prefix = vae_input_key_prefix
        self.vae_input_observation_key = vae_input_key_prefix + '_observation'
        self.vae_input_achieved_goal_key = vae_input_key_prefix + '_achieved_goal'
        self.vae_input_desired_goal_key = vae_input_key_prefix + '_desired_goal'
        self.desired_goal = {}
        self.desired_goal['latent_desired_goal'] = latent_space.sample()
        self._initial_obs = None
        self.latent_subgoals = None
        self.latent_subgoals_reproj = None
        self.subgoal_v_vals = None
        self.image_subgoals = None
        self.updated_image_subgoals = False
        self.wrapped_env.reset()
        self.use_vae_dataset = use_vae_dataset
        self.vae_dataset_path = vae_dataset_path
        self.mu = None
        self.std = None
        self.prior_distr = None
        # add latent_ob_dict for sir code
        self.latent_obs_dict = {}


    def get_obs(self):
        return self.wrapped_env.get_obs()
        # return self._update_obs(self.wrapped_env.get_obs())

    def step(self,action):
        new_obs, reward, done, info = self.wrapped_env.step(action)
        # print('latent_env_done',done)
        # print('reward_lattnet',reward+1.0)
        return new_obs,reward+1.0,done,info

    def compute_reward(self, achieved_goal, desired_goal, temp_info=None, action=None, prev_obs=None):

        reward, _ = self.compute_reward_and_success(achieved_goal=achieved_goal, desired_goal=desired_goal)
        return reward

    def compute_reward_and_success(self, achieved_goal, desired_goal, temp_info=None, action=None, prev_obs=None):
        latents_all = np.concatenate((achieved_goal, desired_goal), axis=0)

        image_all = self._decode(latents_all)

        achieved_state = self.image_goal_to_state(image_all[0])
        desired_state = self.image_goal_to_state(image_all[1])

        obs = dict(state_achieved_goal=achieved_state, state_desired_goal=desired_state)

        reward = self.compute_rewards(actions=[], obs=obs)
        success = np.linalg.norm(achieved_state - desired_state) < 1.0

        return reward, success

    def compute_state_reward_and_success(self,achieved_goal,desired_goal,info=False,temp_info=None,action=None,prev_obs=None):
        size = achieved_goal.shape[0]
        # if self.reward_type in ('state_sparse','sparse'):
        dist = 0
        ## first two element is for the hand_pos and the later two element is for the puck_pos
        hand_dist = achieved_goal[:size//2]-desired_goal[:size//2]
        puck_dist = achieved_goal[-size//2:]-desired_goal[-size//2:]
        # puck_dist = achieved_goal[:size // 2] - desired_goal[:size // 2]
        # hand_dist = achieved_goal[-size // 2:] - desired_goal[-size // 2:]
        puck_dist_2 = np.linalg.norm(puck_dist)
        hand_dist_2 = np.linalg.norm(hand_dist)
        if self.reward_object==2:

            dist = np.linalg.norm(hand_dist)+ np.linalg.norm(puck_dist)

        elif self.reward_object == 1:

            dist = np.linalg.norm(achieved_goal-desired_goal)

        # print('self.epsilon',self.epsilon)
        ## add reward_offset for sparse environment
        if self.reward_type in ('state_sparse') and self.reward_object == 2:
            reward = (not ( (puck_dist_2  < 0.06) and (hand_dist_2 < 0.03) )) * -1.0
            reward = reward + 1.0
            success = (puck_dist_2  < 0.06) and (hand_dist_2 < 0.03)
            # print('reward_type_object', self.reward_type, self.reward_object, type(self.reward_object))

        elif self.reward_type in ('state_sparse','sparse') and self.reward_object !=2 :
            reward = (dist >= self.epsilon) * -1.0
            success = dist < self.epsilon
            reward=reward+1.0

        else:
            reward = -dist
            success = dist < self.epsilon
        if info:
            return reward,success,dict(puck_success=puck_dist_2<0.06,hand_success = hand_dist_2 < 0.06)
        else:
            return reward, success

    #[ 0.00205863  0.53339054  0.02616989  0.59154211]
#  [-0.04464616  0.60091807 -0.09635188  0.57438248]]
# desired_state [[ 0.13523878  0.62652773 -0.07158087  0.51752794]
#  [ 0.02142931  0.5346298  -0.14061854  0.50677612]]


    def compute_img_reward_and_success(self,achieved_goal,desired_goal,temp_info=None,action=None,prev_obs=None):
        achieved_state = self.image_goal_to_state(achieved_goal)
        desired_state = self.image_goal_to_state(desired_goal)
        size = achieved_state.shape[0]
        puck_dist = achieved_state[:size//2]-desired_state[:size//2]
        hand_dist = achieved_state[-size//2:]-desired_state[-size//2:]
        # obs = dict(state_achieved_goal=achieved_state, state_desired_goal=desired_state)

        # reward = self.compute_rewards(actions=[], obs=obs)
        dist = np.linalg.norm(hand_dist)+np.linalg.norm(puck_dist)
        print('epsilon',self.epsilon)
        reward = (dist >= self.epsilon) * -1.0
        success = dist < self.epsilon
        ## add reward_offset for sparse environment
        if self.reward_type in ('state_sparse','sparse'):
            reward=reward+1.0

        return reward, success

    def switch_obs_goal(self, obs, goal):
        obs = obs.copy()
        if isinstance(obs, dict):
            obs['achieved_goal'] = obs['observation']
            obs['desired_goal'] = goal[:]
        elif isinstance(obs, np.ndarray):
            obs_dim = self.observation_space['observation'].shape[0]
            goal_dim = self.observation_space['achieved_goal'].shape[0]
            obs[obs_dim:obs_dim + goal_dim] = obs[:goal_dim]
            obs[obs_dim + goal_dim:obs_dim + goal_dim * 2] = goal[:]
        else:
            raise TypeError
        return obs

    def compute_rewards(self, actions, obs, prev_obs=None):
        if self.reward_type == 'latent_sparse':

            desired_goals = obs['latent_desired_goal']
            achieved_goals = obs['latent_achieved_goal']
            # print('desired_goals',desired_goals)
            # print('achieved_goals',achieved_goals)
            dist = np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order)
            # print('dist',dist)
            return (dist >= self.epsilon) * -1.0

        elif self.reward_type == 'state_distance':
            achieved_goals = obs['state_achieved_goal']
            desired_goals = obs['state_desired_goal']
            diff = achieved_goals - desired_goals
            success = np.linalg.norm(diff) < 1.0
            # if success:
            #     print('reward_success')
            return -0.01 * np.linalg.norm(diff) + success * 1
        elif self.reward_type == 'vectorized_state_distance':
            achieved_goals = obs['state_achieved_goal']
            desired_goals = obs['state_desired_goal']
            return -np.abs(desired_goals - achieved_goals)
        elif self.reward_type == 'state_sparse':
            achieved_goals = obs['state_achieved_goal']
            desired_goals = obs['state_desired_goal']
            dist = np.linalg.norm(desired_goals - achieved_goals)
            return (dist >= 1.0) * -1.0
        elif self.reward_type == 'wrapped_env':
            return self.wrapped_env.compute_rewards(actions, obs)
        else:
            return self.wrapped_env.compute_rewards(actions, obs, reward_type=self.reward_type)

    @property
    def goal_dim(self):
        return self.representation_size

    # def get_goal(self):
    #     return self.desired_goal['latent_desired_goal']
    def set_state(self,state):
        self.wrapped_env.set_state_wrapper(state)

    def set_goal(self,goal):
        self.set_goal_times+=1
        # self.desired_goal['desired_goal'] = goal
        # self.desired_goal['latent_desired_goal'] = goal
        # self.desired_goal['latent_desired_goal_mean'] = goal
        # print('goal_shape',goal)
        # image_goal = self._decode(goal)[0]

        if isinstance(goal,dict):
            self.wrapped_env.set_goal(goal)
        else:
            self.desired_goal['image_desired_goal'] = goal
            # self.save_image_goal(image_goal)
            state_goal = self.image_goal_to_state(goal)
            self.desired_goal['state_desired_goal'] = state_goal
            goal_dict = dict(image_desired_goal=goal,state_desired_goal=state_goal)
            self.wrapped_env.set_goal(goal_dict)



    def save_image_goal(self, image_goal):
        dir = '/home/yilin/sir_img/logs'
        image_np = image_goal.reshape(self.input_channels, self.imsize, self.imsize).transpose(1, 2, 0)
        image_unormalize = np.uint8((image_np * 255))
        image_copy = image_unormalize[:, :, 0].copy()
        image_unormalize[:, :, 0] = image_unormalize[:, :, 2]
        image_unormalize[:, :, 2] = image_copy
        image_filename = osp.join(dir, 'image_set_goal%d.png' % self.set_goal_times)
        cv2.imwrite(image_filename, image_unormalize)
        return 0

    def image_goal_to_state(self, img_goal):
        dir = '/home/yilin/sir_image/logs'
        image_np = img_goal.reshape(self.input_channels, self.imsize, self.imsize).transpose(1, 2, 0)
        image_unormalize = np.uint8((image_np * 255))
        image_copy = image_unormalize[:, :, 0].copy()
        image_unormalize[:, :, 0] = image_unormalize[:, :, 2]
        image_unormalize[:, :, 2] = image_copy
        hsv = cv2.cvtColor(image_unormalize, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_range, self.upper_range)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # loop over the contours
        # image = np.zeros((48, 48, 3))
        areas = np.asarray([cv2.contourArea(c) for c in cnts])
        if cnts is None or areas.size == 0:
            print('cnts',cnts)
            mask_file = osp.join(dir, 'mask_fail.png')
            cv2.imwrite(mask_file, mask)
            image_file = osp.join(dir, 'image_fail.png')
            cv2.imwrite(image_file, image_unormalize)
            cX = 0
            cY = 0
        else:
            max_contours = cnts[np.argmax(areas)]
            # compute the center of the contour
            M = cv2.moments(max_contours)
            if M['m00'] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                print('max_contours',max_contours)
                mask_file = osp.join(dir, 'mask_fail.png')
                cv2.imwrite(mask_file, mask)
                image_file = osp.join(dir, 'image_fail.png')
                cv2.imwrite(image_file, image_unormalize)
                cX = 0
                cY = 0

        pos = np.array([cX, cY]) / 47 * 9 - 4.5
        return pos


    def get_env_update(self):
        """
        For online-parallel. Gets updates to the environment since the last time
        the env was serialized.

        subprocess_env.update_env(**env.get_env_update())
        """
        return dict(
            mode_map=self._mode_map,
            vae_state=self.vae.__getstate__(),
        )

    def update_env(self, mode_map, vae_state):
        self._mode_map = mode_map
        self.vae.__setstate__(vae_state)



class VAEWrappedEnv(ProxyEnv, Env):
    """This class wraps an image-based environment with a VAE.
    Assumes you get flattened (channels,84,84) observations from wrapped_env.
    This class adheres to the "Silent Multitask Env" semantics: on reset,
    it resamples a goal.
    """
    def __init__(
        self,
        wrapped_env,
        vae,
        vae_input_key_prefix='image',
        use_vae_goals=True,
        sample_from_true_prior=False,
        decode_goals=False,
        render_goals=False,
        render_rollouts=False,
        reward_params=None,
        mode="train",
        imsize=84,
        obs_size=None,
        norm_order=2,
        epsilon=0.4,
        temperature=1.0,
        vis_granularity=50,
        presampled_goals=None,
        train_noisy_encoding=False,
        test_noisy_encoding=False,
        noise_type=None, #DEPRECATED
        num_samples_for_latent_histogram=10000,
        use_reprojection_network=False,
        reprojection_network=None,
        use_vae_dataset=True,
        vae_dataset_path=None,
        clip_encoding_std=True,
        use_replay_buffer_goals=False, #DEPRECATED FEATURE
        v_func_heatmap_bounds=(-1.5, 0.0),
        reproj_vae=None,
        disable_annotated_images=False,
    ):
        self.quick_init(locals())
        if reward_params is None:
            reward_params = dict()
        super().__init__(wrapped_env)
        if type(vae) is str:
            self.vae = load_local_or_remote_file(vae)
        else:
            self.vae = vae
        ptu.set_device(0)
        self.vae.cuda()
        # set the blue range for detecting the position of the point
        self.lower_range = np.array([110, 50, 50])
        self.upper_range = np.array([130, 255, 255])
        self.noisy_encoding = False
        # if reproj_vae is not None:
        #     if type(reproj_vae) is str:
        #         self.reproj_vae = load_local_or_remote_file(reproj_vae)
        #     else:
        #         self.reproj_vae = reproj_vae
        #     if ptu.gpu_enabled():
        #         self.reproj_vae.cuda()
        # else:
        #     self.reproj_vae = None

        self.representation_size = self.vae.representation_size
        if hasattr(self.vae, 'input_channels'):
            self.input_channels = self.vae.input_channels
        else:
            self.input_channels = None
        self.set_goal_times = 0
        self._reconstr_image_observation = False
        self._use_vae_goals = use_vae_goals
        self.sample_from_true_prior = sample_from_true_prior
        self.decode_goals = decode_goals
        self.render_goals = render_goals
        self.render_rollouts = render_rollouts
        self.default_kwargs=dict(
            decode_goals=decode_goals,
            render_goals=render_goals,
            render_rollouts=render_rollouts,
        )

        self.clip_encoding_std = clip_encoding_std

        self.train_noisy_encoding = train_noisy_encoding
        self.test_noisy_encoding = test_noisy_encoding
        self.noise_type = noise_type

        self.imsize = imsize
        # self.vis_granularity = vis_granularity # for heatmaps
        self.reward_params = reward_params
        self.reward_type = self.reward_params.get("type", 'latent_distance')
        self.norm_order = self.reward_params.get("norm_order", norm_order)
        self.epsilon = self.reward_params.get("epsilon", epsilon) # for sparse reward
        self.temperature = self.reward_params.get("temperature", temperature) # for exponential reward
        self.reward_min_variance = self.reward_params.get("min_variance", 0)
        latent_space = Box(
            -10 * np.ones(obs_size or self.representation_size),
            10 * np.ones(obs_size or self.representation_size),
            dtype=np.float32,
        )

        spaces = self.wrapped_env.observation_space.spaces
        spaces['observation'] = latent_space
        spaces['desired_goal'] = latent_space
        spaces['achieved_goal'] = latent_space

        spaces['latent_observation'] = latent_space
        spaces['latent_observation_mean'] = latent_space
        spaces['latent_observation_std'] = latent_space

        spaces['latent_desired_goal'] = latent_space
        spaces['latent_desired_goal_mean'] = latent_space
        spaces['latent_desired_goal_std'] = latent_space

        spaces['latent_achieved_goal'] = latent_space
        spaces['latent_achieved_goal_mean'] = latent_space
        spaces['latent_achieved_goal_std'] = latent_space

        self.observation_space = Dict(spaces)
        # self.mode(mode)

        self.vae_input_key_prefix = vae_input_key_prefix
        assert vae_input_key_prefix in set(['image', 'image_proprio', 'state'])
        self.vae_input_observation_key = vae_input_key_prefix + '_observation'
        self.vae_input_achieved_goal_key = vae_input_key_prefix + '_achieved_goal'
        self.vae_input_desired_goal_key = vae_input_key_prefix + '_desired_goal'

        # self._presampled_goals = presampled_goals
        # if self._presampled_goals is None:
        #     self.num_goals_presampled = 0
        # else:
        #     self.num_goals_presampled = presampled_goals[random.choice(list(presampled_goals))].shape[0]
        #     self._presampled_latent_goals, self._presampled_latent_goals_mean, self._presampled_latent_goals_std = \
        #         self._encode(
        #             self._presampled_goals[self.vae_input_desired_goal_key],
        #             noisy=self.noisy_encoding,
        #             batch_size=2500
        #         )

        # self._mode_map = {}
        self.desired_goal = {}
        self.desired_goal['latent_desired_goal'] = latent_space.sample()
        self._initial_obs = None

        self.latent_subgoals = None
        self.latent_subgoals_reproj = None
        self.subgoal_v_vals = None
        self.image_subgoals = None
        # self.image_subgoals_stitched = None
        # self.image_subgoals_reproj_stitched = None
        self.updated_image_subgoals = False

        # self.sweep_goal_mu = None
        self.wrapped_env.reset()
        # self.sweep_goal_latents = None

        # self.use_reprojection_network = use_reprojection_network
        # self.reprojection_network = reprojection_network

        self.use_vae_dataset = use_vae_dataset
        self.vae_dataset_path = vae_dataset_path

        # self.num_samples_for_latent_histogram = num_samples_for_latent_histogram
        # self.latent_histogram = None
        # self.latent_histogram_noisy = None

        # self.num_active_dims = 0
        # for std in self.vae.dist_std:
        #     if std > 0.15:
        #         self.num_active_dims += 1

        # self.active_dims = self.vae.dist_std.argsort()[-self.num_active_dims:][::-1]
        # self.inactive_dims = self.vae.dist_std.argsort()[:-self.num_active_dims][::-1]

        self.mu = None
        self.std = None
        self.prior_distr = None
        # add latent_ob_dict for sir code
        self.latent_obs_dict = {}
        # self.v_func_heatmap_bounds = v_func_heatmap_bounds

        # self.vis_blacklist = []
        # self.disable_annotated_images = disable_annotated_images

    def reset(self):
        self.updated_image_subgoals = False
        obs = self.wrapped_env.reset()
        goal = self.wrapped_env.get_goal()

        # if self.use_vae_goals:
        #     latent_goals = self._sample_vae_prior(1)
        #     latent_goal = latent_goals[0]
        #     latent_goal_mean = latent_goal
        #     latent_goal_std = np.zeros(latent_goal.shape)
        # else:
        latent_goal, latent_goal_mean, latent_goal_std = \
                self._encode_one(obs[self.vae_input_desired_goal_key], noisy=self.noisy_encoding)

        if self.decode_goals:
            decoded_goal = self._decode(latent_goal)[0]
            image_goal, proprio_goal = self._image_and_proprio_from_decoded_one(
                decoded_goal
            )
        else:
            decoded_goal = goal.get(self.vae_input_desired_goal_key, None)
            image_goal = goal.get('image_desired_goal', None)
            proprio_goal = goal.get('proprio_desired_goal', None)

        goal['desired_goal'] = latent_goal
        goal['latent_desired_goal'] = latent_goal
        goal['latent_desired_goal_mean'] = latent_goal_mean
        goal['latent_desired_goal_std'] = latent_goal_std

        goal['image_desired_goal'] = image_goal
        goal['proprio_desired_goal'] = proprio_goal
        goal[self.vae_input_desired_goal_key] = decoded_goal
        self.desired_goal = goal
        self._initial_obs = obs
        return self._update_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)

        self._update_info(info, new_obs)

        keys = [
            'latent_desired_goal',
            'latent_desired_goal_mean',
            'latent_desired_goal_std',
            'latent_achieved_goal',
            'latent_achieved_goal_mean',
            'latent_achieved_goal_std',
            'state_observation',
            'state_achieved_goal',
            'state_desired_goal',
        ]
        # reward = self.compute_reward(
        #     action,
        #     {k: new_obs[k] for k in keys},
        #     prev_obs=None,
        # )
        # print('new_obs',new_obs['latent_achieved_goal'])
        reward,success = self.compute_reward_and_success(
            action=action,achieved_goal=new_obs['latent_achieved_goal'],desired_goal=new_obs['latent_desired_goal']
        )
        info['is_success'] = success
        if success:
            print('episode succeed!')
        # self.try_render(new_obs)
        return new_obs, reward, done, info

    def _update_obs(self, obs):
        latent_obs, latent_obs_mean, latent_obs_std = \
            self._encode_one(obs[self.vae_input_observation_key], noisy=self.noisy_encoding)
        self.latent_obs = latent_obs
        obs['observation'] = latent_obs
        obs['latent_observation'] = latent_obs
        obs['latent_observation_mean'] = latent_obs_mean
        obs['latent_observation_std'] = latent_obs_std

        obs['achieved_goal'] = latent_obs
        obs['latent_achieved_goal'] = latent_obs
        obs['latent_achieved_goal_mean'] = latent_obs_mean
        obs['latent_achieved_goal_std'] = latent_obs_std

        # if "video" in self.cur_mode:
        #     self._update_obs_images(obs)
        # print(self.desired_goal)

        obs = {**obs, **self.desired_goal}
        if not self._reconstr_image_observation and 'image' not in self.vae_input_key_prefix:
            keys = list(obs.keys()).copy()
            for key in keys:
                if 'image' in key:
                    del obs[key]
        return obs

    def get_obs(self):
        return self._update_obs(self.wrapped_env.get_obs())

    def _update_obs_images(self, obs):
        ### if using subgoals ###
        if self.image_subgoals is not None:
            obs['image_desired_subgoal'] = self.image_subgoals[0]
            obs['image_desired_subgoal_reproj'] = self.image_subgoals_reproj[0]

            obs['image_desired_subgoal'] = self._annotate_image(
                obs['image_desired_subgoal'],
                text="V:" + "{:.3f}".format(self.subgoal_v_vals[0])
            )
            obs['image_desired_subgoal_reproj'] = self._annotate_image(
                obs['image_desired_subgoal_reproj'],
                text="V:" + "{:.3f}".format(self.subgoal_v_vals[0])
            )
            if not self.updated_image_subgoals: # do this only once, at beginning
                obs['image_desired_subgoals_reproj'] = self.image_subgoals_reproj
                for i in range(obs['image_desired_subgoals_reproj'].shape[0]):
                    obs['image_desired_subgoals_reproj'][i] = self._annotate_image(
                        obs['image_desired_subgoals_reproj'][i],
                        text="V:" + "{:.3f}".format(self.subgoal_v_vals[i])
                    )
                obs['image_desired_goal_annotated'] = self._annotate_image(
                    obs['image_desired_goal'],
                    text="V:" + "{:.3f}".format(self.subgoal_v_vals[-1])
                )

            self.updated_image_subgoals = True
        elif self.updated_image_subgoals: # at the very end, once we have no remaining subgoals, and just have goal
            obs['image_desired_subgoal'] = self.desired_goal['image_desired_goal']
            obs['image_desired_subgoal_reproj'] = obs['image_desired_subgoal']

            obs['image_desired_subgoal'] = self._annotate_image(
                obs['image_desired_subgoal'],
                text="V:" + "{:.3f}".format(self.subgoal_v_vals[-1])
            )
            obs['image_desired_subgoal_reproj'] = self._annotate_image(
                obs['image_desired_subgoal_reproj'],
                text="V:" + "{:.3f}".format(self.subgoal_v_vals[-1])
            )

        if self.vae_input_key_prefix == 'state':
            state_decoding = self._decode(obs['latent_observation'])[0]
            image_decoding = self.wrapped_env.state_to_image(state_decoding)
        else:
            image_decoding = self._decode(obs['latent_observation'])[0]

        wrapped_env = self.wrapped_env
        if isinstance(wrapped_env, ImageEnv):
            wrapped_env = wrapped_env.wrapped_env

        if isinstance(wrapped_env, Point2DWallEnv):
            goal_image = np.copy(self.desired_goal['image_desired_goal'])
            goal_image = goal_image.reshape(-1, self.imsize, self.imsize)
            goal_image[1], goal_image[2] = goal_image[2], np.copy(goal_image[1])
            goal_image = goal_image.reshape(-1)
            if self.image_subgoals_stitched is None:
                obs['reconstr_image_observation'] = np.min(
                    (image_decoding, goal_image),
                    axis=0)
            else:
                obs['reconstr_image_observation'] = np.min(
                    (image_decoding, goal_image, self.image_subgoals_stitched),
                    axis=0)

            if self.image_subgoals_reproj_stitched is None:
                obs['reconstr_image_reproj_observation'] = np.min(
                    (image_decoding, goal_image),
                    axis=0)
            else:
                obs['reconstr_image_reproj_observation'] = np.min(
                    (image_decoding, goal_image, self.image_subgoals_reproj_stitched),
                    axis=0)

            if self.subgoal_v_vals is not None:
                obs['reconstr_image_observation'] = self._annotate_image(
                    obs['reconstr_image_observation'],
                    text="V:" + "{:.3f}".format(self.subgoal_v_vals[0]),
                    color=(0, 0, 0),
                )
                obs['reconstr_image_reproj_observation'] = self._annotate_image(
                    obs['reconstr_image_reproj_observation'],
                    text="V:" + "{:.3f}".format(self.subgoal_v_vals[0]),
                    color=(0, 0, 0),
                )
        else:
            obs['reconstr_image_observation'] = image_decoding

    # def update_subgoals(self, subgoals, subgoals_reproj, subgoal_v_vals=None):
    #     self.latent_subgoals = subgoals
    #     self.latent_subgoals_reproj = subgoals_reproj
    #     self.subgoal_v_vals = subgoal_v_vals
    #
    #     wrapped_env = self.wrapped_env
    #     if isinstance(wrapped_env, ImageEnv):
    #         wrapped_env = wrapped_env.wrapped_env
    #
    #     if self.vae_input_key_prefix == "state" and self.latent_subgoals is None:
    #         wrapped_env.update_subgoals(None)
    #
    #     if (self.latent_subgoals is None) or ("video" not in self.cur_mode):
    #         self.image_subgoals = None
    #         self.image_subgoals_reproj = None
    #         self.image_subgoals_stitched = None
    #         self.image_subgoals_reproj_stitched = None
    #         return
    #
    #     if self.vae_input_key_prefix == "image":
    #         if self.reproj_vae is not None:
    #             self.image_subgoals = self._decode(self.latent_subgoals, use_reproj_vae=True)
    #         else:
    #             self.image_subgoals = self._decode(self.latent_subgoals, use_reproj_vae=False)
    #         self.image_subgoals_reproj = self._decode(self.latent_subgoals_reproj)
    #     elif self.vae_input_key_prefix == "state":
    #         decoded_subgoals = self._decode(self.latent_subgoals)
    #         wrapped_env.update_subgoals(decoded_subgoals)
    #
    #         self.image_subgoals = self.wrapped_env.states_to_images(decoded_subgoals)
    #         self.image_subgoals_reproj = self.image_subgoals
    #
    #     ### Special case for pointmass: stitch all subgoal images into one image ###
    #     if isinstance(wrapped_env, Point2DWallEnv):
    #         self.image_subgoals_stitched = self._stitch_images(self.image_subgoals)
    #         self.image_subgoals_reproj_stitched = self._stitch_images(self.image_subgoals_reproj)
    #
    # def _stitch_images(self, images):
    #     images = images.reshape(-1, 3, self.imsize, self.imsize)
    #     for (image, factr) in zip(images, np.linspace(0, 0.60, images.shape[0])):
    #         image[2] -= (factr * (image[0] <= 0.4) * (image[1] <= 0.4) * (image[2] >= 0.6))
    #     images_stitched = images.min(axis=0)
    #     images_stitched[0], images_stitched[2] = images_stitched[2], np.copy(images_stitched[0])
    #     return images_stitched.reshape(-1)

    def _update_info(self, info, obs):
        latent_obs, logvar = self.vae.encode(
            ptu.np_to_var(obs[self.vae_input_observation_key])
        )
        latent_obs, logvar = ptu.get_numpy(latent_obs)[0], ptu.get_numpy(logvar)[0]
        if not self.noisy_encoding:
            assert (latent_obs == obs['latent_observation']).all()
        latent_goal = self.desired_goal['latent_desired_goal']
        dist = latent_goal - latent_obs
        info["vae_dist"] = np.linalg.norm(dist, ord=self.norm_order)
        info["vae_dist_l1"] = np.linalg.norm(dist, ord=1)
        info["vae_dist_l2"] = np.linalg.norm(dist, ord=2)
        ## updating info is_success for latent_environment
        # recalculate the success flag\
        # latents_all = np.concatenate((latent_obs, latent_goal), axis=0)
        # # print('latents_all_shape',latents_all.shape)
        # image_all = self._decode(latents_all)
        # # print('image_all',image_all.shape)
        # achieved_state = self.image_goal_to_state(image_all[0])
        # desired_state = self.image_goal_to_state(image_all[1])
        #
        # info['is_success'] = np.linalg.norm(achieved_state-desired_state) < 1.0
        # info['is_success'] =  np.linalg.norm(latent_goal - latent_obs, ord=self.norm_order)< self.epsilon
        # if info['is_success']:
        #     print('episode succeed!')


    # def set_presampled_goals(self, goals):
    #     self._presampled_goals = goals
    #     if self._presampled_goals is None:
    #         self.num_goals_presampled = 0
    #     else:
    #         self.num_goals_presampled = goals[random.choice(list(goals))].shape[0]
    #         self._presampled_latent_goals, self._presampled_latent_goals_mean, self._presampled_latent_goals_std = \
    #             self._encode(
    #                 self._presampled_goals[self.vae_input_desired_goal_key],
    #                 noisy=self.noisy_encoding,
    #                 batch_size=2500
    #             )

    # @property
    # def use_vae_goals(self):
    #     return self._use_vae_goals and not self.reward_type.startswith('state')
    #
    # """
    # Multitask functions
    # """
    # def sample_goals(self, batch_size, keys=None):
    #     if self.use_vae_goals:
    #         goals = {}
    #         latent_goals = self._sample_vae_prior(batch_size)
    #         latent_goals_mean = latent_goals
    #         latent_goals_std = np.zeros(latent_goals.shape)
    #         goals['state_desired_goal'] = None
    #     elif self.num_goals_presampled > 0:
    #         idx = np.random.randint(0, self.num_goals_presampled, batch_size)
    #         if keys is None:
    #             valid_keys = self._presampled_goals.keys()
    #         else:
    #             valid_keys = [k for k in keys if k in self._presampled_goals.keys()]
    #         goals = {
    #             k: self._presampled_goals[k][idx] for k in valid_keys
    #         }
    #         latent_goals = self._presampled_latent_goals[idx]
    #         latent_goals_mean = self._presampled_latent_goals_mean[idx]
    #         latent_goals_std = self._presampled_latent_goals_std[idx]
    #     else:
    #         goals = self.wrapped_env.sample_goals(batch_size)
    #         latent_goals, latent_goals_mean, latent_goals_std = \
    #             self._encode(goals[self.vae_input_desired_goal_key], noisy=self.noisy_encoding)
    #
    #     if self.decode_goals:
    #         decoded_goals = self._decode(latent_goals)
    #     else:
    #         decoded_goals = None
    #     image_goals, proprio_goals = self._image_and_proprio_from_decoded(
    #         decoded_goals
    #     )
    #
    #     goals['desired_goal'] = latent_goals
    #     goals['latent_desired_goal'] = latent_goals
    #     goals['latent_desired_goal_mean'] = latent_goals_mean
    #     goals['latent_desired_goal_std'] = latent_goals_std
    #     goals['proprio_desired_goal'] = proprio_goals
    #     goals['image_desired_goal'] = image_goals
    #     goals[self.vae_input_desired_goal_key] = decoded_goals
    #     return goals

    # def sample_goal(self):
    #     goals = self.sample_goals(1)
    #     return self.unbatchify_dict(goals, 0)
    #
    # def generate_expert_subgoals(self, num_subgoals):
    #     # print(self.use_vae_goals)
    #     # print(getattr(self.wrapped_env, "generate_expert_subgoals", None) is not None)
    #     # import ipdb;ipdb.set_trace()
    #     if self.use_vae_goals and getattr(self.wrapped_env, "generate_expert_subgoals", None) is not None:
    #         env_subgoals = self.wrapped_env.generate_expert_subgoals(num_subgoals)
    #         if self.vae_input_key_prefix == 'state':
    #             return self._encode(env_subgoals, noisy=self.noisy_encoding)[0]
    #         else:
    #             pre_state = self.wrapped_env.get_env_state()
    #             batch_size = env_subgoals.shape[0]
    #             imgs = np.zeros((batch_size, self.wrapped_env.image_length))
    #             for i in range(batch_size):
    #                 self.wrapped_env.set_to_goal({"state_desired_goal": env_subgoals[i]})
    #                 img = self._get_flat_img()
    #                 imgs[i, :] = img
    #             self.wrapped_env.set_env_state(pre_state)
    #             return self._encode(imgs, noisy=self.noisy_encoding)[0]
    #     else:
    #         return None

    # def compute_reward(self, action, obs, prev_obs=None):
    #     actions = action[None]
    #     next_obs = {
    #         k: v[None] for k, v in obs.items()
    #     }
    #     if prev_obs is not None:
    #         prev_obs = {
    #             k: v[None] for k, v in prev_obs.items()
    #         }
    #     return self.compute_rewards(actions, next_obs, prev_obs)[0]
    def compute_reward(self,achieved_goal,desired_goal,temp_info=None,action=None,prev_obs=None):

        reward, _ = self.compute_reward_and_success(achieved_goal=achieved_goal,desired_goal=desired_goal)
        return reward

    def compute_reward_and_success(self,achieved_goal,desired_goal,temp_info=None,action=None,prev_obs=None):
        latents_all = np.concatenate((achieved_goal, desired_goal), axis=0)
        # print('latents_all_shape',latents_all.shape)
        image_all = self._decode(latents_all)
        # print('image_all',image_all.shape)
        achieved_state = self.image_goal_to_state(image_all[0])
        desired_state = self.image_goal_to_state(image_all[1])

        obs = dict(state_achieved_goal=achieved_state, state_desired_goal=desired_state)
        # obs['state_achieved_goal']=
        # obs['state_desired_goal']=
        # obs['latent_achieved_goal']= achieved_goal
        # obs['latent_desired_goal'] = desired_goal
        reward = self.compute_rewards(actions=[], obs=obs)
        success = np.linalg.norm(achieved_state-desired_state)<1.0
        # if success:
        #     print('reward_and_sucess')
        return reward,success

    def switch_obs_goal(self,obs,goal):
        obs = obs.copy()
        if isinstance(obs, dict):
            obs['latent_achieved_goal'] = obs['latent_observation']
            obs['latent_desired_goal'] = goal[:]
        elif isinstance(obs, np.ndarray):
            obs_dim = self.observation_space['latent_observation'].shape[0]
            goal_dim = self.observation_space['latent_achieved_goal'].shape[0]
            obs[obs_dim:obs_dim+goal_dim] = obs[:goal_dim]
            obs[obs_dim+goal_dim:obs_dim+goal_dim*2] = goal[:]
        else:
            raise TypeError
        return obs

    def compute_rewards(self, actions, obs, prev_obs=None):
        # achieved_goals = obs.get('latent_achieved_goal_mean', obs['latent_achieved_goal'])
        # achieved_goals = obs['latent_achieved_goal']
        # desired_goals = obs['latent_desired_goal']
        # desired_goals = obs.get('latent_desired_goal_mean', obs['latent_desired_goal'])
        # print('obs_latent_desired_goal_mean',obs['latent_desired_goal_mean'])
        # print('desired_goals',desired_goals)

        if self.reward_type == 'latent_distance':
            dist = np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order)
            return -dist
        # elif self.reward_type == 'vectorized_latent_distance':
        #     return -np.abs(desired_goals - achieved_goals)
        # elif self.reward_type == 'vectorized_mahalanobis_latent_distance':
        #     if 'latent_achieved_goal_std' in obs:
        #         scaling = 1 / obs['latent_achieved_goal_std']
        #         return -np.abs(desired_goals - achieved_goals) * scaling
        #     else:
        #         return -np.abs(desired_goals - achieved_goals)
        elif self.reward_type == 'latent_sparse':

            desired_goals = obs['latent_desired_goal']
            achieved_goals = obs['latent_achieved_goal']
            # print('desired_goals',desired_goals)
            # print('achieved_goals',achieved_goals)
            dist = np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order)
            # print('dist',dist)
            return (dist >= self.epsilon) * -1.0
        # elif self.reward_type == 'vectorized_latent_sparse':
        #     raise NotImplementedError
        # elif self.reward_type == 'latent_exponential':
        #     dist = np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
        #     return np.exp(-dist/self.temperature) - 1.0
        elif self.reward_type == 'state_distance':
            achieved_goals = obs['state_achieved_goal']
            desired_goals = obs['state_desired_goal']
            diff = achieved_goals-desired_goals
            success = np.linalg.norm(diff) < 1.0
            # if success:
            #     print('reward_success')
            return -0.01*np.linalg.norm(diff)+success*1
        elif self.reward_type == 'vectorized_state_distance':
            achieved_goals = obs['state_achieved_goal']
            desired_goals = obs['state_desired_goal']
            return -np.abs(desired_goals - achieved_goals)
        elif self.reward_type == 'state_sparse':
            achieved_goals = obs['state_achieved_goal']
            desired_goals = obs['state_desired_goal']
            dist = np.linalg.norm(desired_goals - achieved_goals)
            return (dist >= 1.0) * -1.0
        elif self.reward_type == 'wrapped_env':
            return self.wrapped_env.compute_rewards(actions, obs)
        else:
            return self.wrapped_env.compute_rewards(actions, obs, reward_type=self.reward_type)

    @property
    def goal_dim(self):
        return self.representation_size

    def get_goal(self):
        return self.desired_goal['latent_desired_goal']

    # def (self, goal):
    #     """
    #     Assume goal contains both image_desired_goal and any goals required for wrapped envs
    #
    #     :param goal:
    #     :return:
    #     """
    #     self.desired_goal = goal
    #     self.wrapped_env.set_goal(goal)
    def set_goal(self,image_goal):
        self.set_goal_times+=1
        # self.desired_goal['desired_goal'] = goal
        # self.desired_goal['latent_desired_goal'] = goal
        # self.desired_goal['latent_desired_goal_mean'] = goal
        # print('goal_shape',goal)
        # image_goal = self._decode(goal)[0]
        self.desired_goal['image_desired_goal'] = image_goal
        # self.save_image_goal(image_goal)
        state_goal = self.image_goal_to_state(image_goal)
        self.desired_goal['state_desired_goal'] = state_goal
        goal_dict = dict(image_desired_goal=image_goal,state_desired_goal=state_goal)
        self.wrapped_env.set_goal(goal_dict)
        # goal_var = ptu.np_to_var(goal)
        # goal_img = ptu.get_numpy(self.vae.decode(goal_var))
        # self.wrapped_env.set_goal(goal_img)

    def save_image_goal(self,image_goal):
        dir = '/home/yilin/sir_img/logs'
        image_np = image_goal.reshape(self.input_channels,self.imsize,self.imsize).transpose(1,2,0)
        image_unormalize = np.uint8((image_np * 255))
        image_copy = image_unormalize[:, :, 0].copy()
        image_unormalize[:, :, 0] = image_unormalize[:, :, 2]
        image_unormalize[:, :, 2] = image_copy
        image_filename = osp.join(dir,'image_set_goal%d.png'%self.set_goal_times)
        cv2.imwrite(image_filename,image_unormalize)
        return 0

    def image_goal_to_state(self,img_goal):
        dir = '/home/yilin/sir_image/logs'
        image_np = img_goal.reshape(self.input_channels,self.imsize,self.imsize).transpose(1,2,0)
        image_unormalize = np.uint8((image_np * 255))
        image_copy = image_unormalize[:, :, 0].copy()
        image_unormalize[:, :, 0] = image_unormalize[:, :, 2]
        image_unormalize[:, :, 2] = image_copy
        hsv = cv2.cvtColor(image_unormalize, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_range, self.upper_range)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # loop over the contours
        # image = np.zeros((48, 48, 3))
        areas = np.asarray([cv2.contourArea(c) for c in cnts])
        if cnts is None or areas.size==0:
            print(cnts)
            mask_file = osp.join(dir, 'mask_fail.png')
            cv2.imwrite(mask_file, mask)
            image_file = osp.join(dir, 'image_fail.png')
            cv2.imwrite(image_file, image_unormalize)
            cX = 0
            cY = 0
        else:
            max_contours = cnts[np.argmax(areas)]
            # compute the center of the contour
            M = cv2.moments(max_contours)
            if M['m00'] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                print(max_contours)
                mask_file = osp.join(dir,'mask_fail.png')
                cv2.imwrite(mask_file,mask)
                image_file = osp.join(dir,'image_fail.png')
                cv2.imwrite(image_file,image_unormalize)
                cX = 0
                cY = 0
        # print('position x,y', cX, cY)
        # cv2.circle(image, (cX, cY), 2, (255, 255, 255), -1)
        # filename = osp.join(dir, 'mark_image.png')
        # cv2.imwrite(filename, image)
        pos = np.array([cX, cY]) / 47 * 9 - 4.5
        return pos
        # convert the image position to env position


        # goal_image = goal_image.reshape(-1, self.imsize, self.imsize).transpose()
        # goal_image[1], goal_image[2] = goal_image[2], np.copy(goal_image[1])
        # goal_image = goal_image.reshape(-1)

    # def get_diagnostics(self, paths, **kwargs):
    #     statistics = self.wrapped_env.get_diagnostics(paths, **kwargs)
    #     for stat_name_in_paths in ["vae_dist"]:
    #         stats = get_stat_in_paths(paths, 'env_infos', stat_name_in_paths)
    #         statistics.update(create_stats_ordered_dict(
    #             stat_name_in_paths,
    #             stats,
    #             always_show_all_stats=True,
    #         ))
    #         final_stats = [s[-1] for s in stats]
    #         statistics.update(create_stats_ordered_dict(
    #             "Final " + stat_name_in_paths,
    #             final_stats,
    #             always_show_all_stats=True,
    #         ))
    #     return statistics
    # #
    # """
    # Other functions
    # """
    # def mode(self, name):
    #     if name == "train":
    #         self._reconstr_image_observation = False
    #         self._use_vae_goals = True
    #         self.decode_goals = self.default_kwargs['decode_goals']
    #         self.render_goals = self.default_kwargs['render_goals']
    #         self.render_rollouts = self.default_kwargs['render_rollouts']
    #         self.noisy_encoding = self.train_noisy_encoding
    #     elif name == "train_env_goals":
    #         self._reconstr_image_observation = False
    #         self._use_vae_goals = False
    #         self.decode_goals = self.default_kwargs['decode_goals']
    #         self.render_goals = self.default_kwargs['render_goals']
    #         self.render_rollouts = self.default_kwargs['render_rollouts']
    #         self.noisy_encoding = self.train_noisy_encoding
    #     elif name == "test":
    #         self._reconstr_image_observation = False
    #         self._use_vae_goals = False
    #         self.decode_goals = self.default_kwargs['decode_goals']
    #         self.render_goals = self.default_kwargs['render_goals']
    #         self.render_rollouts = self.default_kwargs['render_rollouts']
    #         self.noisy_encoding = self.test_noisy_encoding
    #     elif name == "video_vae":
    #         self._reconstr_image_observation = True
    #         self._use_vae_goals = True
    #         self.decode_goals = True
    #         self.render_goals = False
    #         self.render_rollouts = False
    #         self.noisy_encoding = self.test_noisy_encoding
    #     elif name == "video_env":
    #         self._reconstr_image_observation = True
    #         self._use_vae_goals = False
    #         self.decode_goals = False
    #         self.render_goals = False
    #         self.render_rollouts = False
    #         self.render_decoded = False
    #         self.noisy_encoding = self.test_noisy_encoding
    #     else:
    #         raise ValueError("Invalid mode: {}".format(name))
    #     if hasattr(self.wrapped_env, "mode"):
    #         self.wrapped_env.mode(name)
    #     self.cur_mode = name
    #
    # def add_mode(self, env_type, mode):
    #     assert env_type in ['train',
    #                     'eval',
    #                     'video_vae',
    #                     'video_env',
    #                     'relabeling']
    #     assert mode in ['train',
    #                     'train_env_goals',
    #                     'test',
    #                     'video_vae',
    #                     'video_env']
    #     assert env_type not in self._mode_map
    #     self._mode_map[env_type] = mode
    #
    # def train(self):
    #     self.mode(self._mode_map['train'])
    #
    # def eval(self):
    #     self.mode(self._mode_map['eval'])

    def get_env_update(self):
        """
        For online-parallel. Gets updates to the environment since the last time
        the env was serialized.

        subprocess_env.update_env(**env.get_env_update())
        """
        return dict(
            mode_map=self._mode_map,
            vae_state=self.vae.__getstate__(),
        )

    def update_env(self, mode_map, vae_state):
        self._mode_map = mode_map
        self.vae.__setstate__(vae_state)

    # def enable_render(self):
    #     self._use_vae_goals = False
    #     self.decode_goals = True
    #     self.render_goals = True
    #     self.render_rollouts = True
    #
    # def disable_render(self):
    #     self.decode_goals = False
    #     self.render_goals = False
    #     self.render_rollouts = False
    #
    # def try_render(self, obs):
    #     if self.render_rollouts:
    #         img = obs['image_observation'].reshape(
    #             self.input_channels,
    #             self.imsize,
    #             self.imsize,
    #         ).transpose()
    #         cv2.imshow('env', img)
    #         cv2.waitKey(1)
    #         reconstruction = self._reconstruct_img(obs['image_observation']).transpose()
    #         cv2.imshow('env_reconstruction', reconstruction)
    #         cv2.waitKey(1)
    #         init_img = self._initial_obs['image_observation'].reshape(
    #             self.input_channels,
    #             self.imsize,
    #             self.imsize,
    #         ).transpose()
    #         cv2.imshow('initial_state', init_img)
    #         cv2.waitKey(1)
    #         init_reconstruction = self._reconstruct_img(
    #             self._initial_obs['image_observation']
    #         ).transpose()
    #         cv2.imshow('init_reconstruction', init_reconstruction)
    #         cv2.waitKey(1)
    #
    #     if self.render_goals:
    #         goal = obs['image_desired_goal'].reshape(
    #             self.input_channels,
    #             self.imsize,
    #             self.imsize,
    #         ).transpose()
    #         cv2.imshow('goal', goal)
    #         cv2.waitKey(1)
    #
    # def _sample_vae_prior(self, batch_size):
    #     if self.sample_from_true_prior:
    #         mu, sigma = 0, 1  # sample from prior
    #     else:
    #         mu, sigma = self.vae.dist_mu, self.vae.dist_std
    #     n = np.random.randn(batch_size, self.representation_size)
    #     return sigma * n + mu

    def _decode(self, latents, use_reproj_vae=False):
        latents = ptu.np_to_var(latents)
        latents = latents.view(-1, self.representation_size)
        if use_reproj_vae:
            assert self.reproj_vae is not None
            decoded = self.reproj_vae.decode(latents)
        else:
            decoded = self.vae.decode(latents)
        return ptu.get_numpy(decoded)

    def _encode_one(self, img, noisy):
        sample, mu, std = self._encode(img[None], noisy)
        return sample[0], mu[0], std[0]

    def _encode(self, imgs, noisy, clip_std=None, batch_size=None):
        if batch_size is None:
            mu, logvar = self.vae.encode(ptu.np_to_var(imgs))
        else:
            imgs = imgs.reshape(-1, self.vae.imlength)
            n = imgs.shape[0]
            mu, logvar = None, None
            for i in range(0, n, batch_size):
                batch_mu, batch_logvar = self.vae.encode(ptu.np_to_var(imgs[i:i + batch_size]))
                if mu is None:
                    mu = batch_mu
                    logvar = batch_logvar
                else:
                    mu = torch.cat((mu, batch_mu), dim=0)
                    logvar = torch.cat((logvar, batch_logvar), dim=0)
        std = logvar.mul(0.5).exp_()
        if clip_std is None:
            clip_std = self.clip_encoding_std
        if clip_std:
            vae_std = np.copy(self.vae.dist_std)
            vae_std = ptu.np_to_var(vae_std)
            std = torch.min(std, vae_std)
        if noisy:
            eps = ptu.Variable(std.data.new(std.size()).normal_())
            sample = eps.mul(std).add_(mu)
        else:
            sample = mu
        return ptu.get_numpy(sample), ptu.get_numpy(mu), ptu.get_numpy(std)

    def _reconstruct_img(self, flat_img):
        zs = self.vae.encode(ptu.np_to_var(flat_img[None]))[0]
        imgs = ptu.get_numpy(self.vae.decode(zs))
        imgs = imgs.reshape(
            1, self.input_channels, self.imsize, self.imsize
        )
        return imgs[0]
    #
    # def _image_and_proprio_from_decoded_one(self, decoded):
    #     if len(decoded.shape) == 1:
    #         decoded = np.array([decoded])
    #     images, proprios = self._image_and_proprio_from_decoded(decoded)
    #     image = None
    #     proprio = None
    #     if images is not None:
    #         image = images[0]
    #     if proprios is not None:
    #         proprio = proprios[0]
    #     return image, proprio
    #
    # def _image_and_proprio_from_decoded(self, decoded):
    #     if decoded is None:
    #         return None, None
    #     if self.vae_input_key_prefix == 'image_proprio':
    #         images = decoded[:, :self.image_length]
    #         proprio = decoded[:, self.image_length:]
    #         return images, proprio
    #     elif self.vae_input_key_prefix == 'image':
    #         return decoded, None
    #     elif self.vae_input_key_prefix == 'state':
    #         decoded = self.wrapped_env.states_to_images(decoded)
    #         return decoded, None
    #     else:
    #         raise AssertionError("Bad prefix for the vae input key.")
    #
    # def encode_states(self, states, clip_std=None):
    #     state_dim = len(self.observation_space.spaces['state_observation'].low)
    #     states = states.reshape((-1, state_dim))
    #     if clip_std is None:
    #         clip_std = self.clip_encoding_std
    #     if self.vae_input_key_prefix == 'state':
    #         return self.encode_imgs(states, clip_std=clip_std)
    #
    #     pre_state = self.wrapped_env.get_env_state()
    #     batch_size = states.shape[0]
    #     imgs = np.zeros((batch_size, self.wrapped_env.image_length))
    #     for i in range(batch_size):
    #         self.wrapped_env.set_to_goal({"state_desired_goal": states[i]})
    #         imgs[i, :] = self._get_flat_img()
    #     self.wrapped_env.set_env_state(pre_state)
    #     return self.encode_imgs(imgs, clip_std=clip_std)

    def encode_imgs(self, imgs, clip_std=None):
        mu, logvar = self.vae.encode(ptu.np_to_var(imgs))
        mu, logvar = ptu.get_numpy(mu), ptu.get_numpy(logvar)
        if clip_std is None:
            clip_std = self.clip_encoding_std
        if clip_std:
            vae_std = np.copy(self.vae.dist_std)
            logvar = np.minimum(logvar, np.log(vae_std ** 2))
        return mu, logvar

    # def reproject_encoding(self, encoding):
    #     orig_dims = encoding.shape
    #     encoding = encoding.view(-1, self.representation_size)
    #     if self.use_reprojection_network:
    #         n = encoding.shape[0]
    #         batch_size = 50000
    #         reconstr_encoding = None
    #         for i in range(0, n, batch_size):
    #             batch_encoding = self.reprojection_network(encoding[i:i + batch_size])
    #             if reconstr_encoding is None:
    #                 reconstr_encoding = batch_encoding
    #             else:
    #                 reconstr_encoding = torch.cat((reconstr_encoding, batch_encoding), dim=0)
    #         # reconstr_encoding = self.reprojection_network(encoding)
    #     else:
    #         n = encoding.shape[0]
    #         batch_size = 1000
    #         reconstr_encoding = None
    #         for i in range(0, n, batch_size):
    #             if self.reproj_vae is not None:
    #                 batch_imgs = self.reproj_vae.decode(encoding[i:i + batch_size])
    #             else:
    #                 batch_imgs = self.vae.decode(encoding[i:i + batch_size])
    #             batch_encoding = self.vae.encode(batch_imgs)[0]
    #             if not encoding.requires_grad:
    #                 batch_encoding = batch_encoding.detach()
    #             if reconstr_encoding is None:
    #                 reconstr_encoding = batch_encoding
    #             else:
    #                 reconstr_encoding = torch.cat((reconstr_encoding, batch_encoding), dim=0)
    #
    #     return reconstr_encoding.view(orig_dims)

    def reparameterize(self, mu, logvar, noisy):
        if noisy:
            std = np.exp(0.5*logvar)
            return np.random.normal(mu, std)
        else:
            return mu

#     def _annotate_image(self, image, text, color=(0, 0, 255)):
#         from multiworld.core.image_env import normalize_image
#         from multiworld.core.image_env import unormalize_image
#
#         if self.disable_annotated_images:
#             return image
#
#         img = unormalize_image(image).reshape(3, self.imsize, self.imsize).transpose((1, 2, 0))
#         img = img[::, :, ::-1]
#         img = img.copy()
#
#         if self.imsize == 84:
#             fontScale = 0.30
#         elif self.imsize == 48:
#             fontScale = 0.25
#         else:
#             fontScale = 0.50
#
#         org = (0, self.imsize - 3)
#         fontFace = 0
#
#         cv2.putText(img=img, text=text, org=org, fontFace=fontFace, fontScale=fontScale,
#                     color=color, thickness=1)
#         img = img[::, :, ::-1]
#         return normalize_image(img.transpose((2, 0, 1)).reshape(self.imsize * self.imsize * 3))
#
#     def set_vis_blacklist(self, vis_blacklist):
#         self.vis_blacklist = vis_blacklist
#
#     """
#     Visualization functions
#     """
#     def compute_sampled_latents(self):
#         vae_wrapper_util.compute_sampled_latents(self)
#
#     def compute_latent_histogram(self):
#         vae_wrapper_util.compute_latent_histogram(self)
#
#     def compute_goal_encodings(self):
#         vae_wrapper_util.compute_goal_encodings(self)
#
#     def get_image_v(self, *args, **kwargs):
#         return vae_wrapper_util.get_image_v(self, *args, **kwargs)
#
#     def get_image_rew(self, *args, **kwargs):
#         return vae_wrapper_util.get_image_rew(self, *args, **kwargs)
#
#     def get_image_latent_histogram_2d(self, *args, **kwargs):
#         return vae_wrapper_util.get_image_latent_histogram_2d(self, *args, **kwargs)
#
#     def get_image_latent_histogram(self, *args, **kwargs):
#         return vae_wrapper_util.get_image_latent_histogram(self, *args, **kwargs)
#
#     def get_image_v_latent(self, *args, **kwargs):
#         return vae_wrapper_util.get_image_v_latent(self, *args, **kwargs)
#
#     def get_image_latent_plt(self, *args, **kwargs):
#         return vae_wrapper_util.get_image_latent_plt(self, *args, **kwargs)
#
#     def dump_latent_histogram(self, *args, **kwargs):
#         vae_wrapper_util.dump_latent_histogram(self, *args, **kwargs)
#
#     def dump_samples(self, *args, **kwargs):
#         vae_wrapper_util.dump_samples(self, *args, **kwargs)
#
#     def dump_reconstructions(self, *args, **kwargs):
#         vae_wrapper_util.dump_reconstructions(self, *args, **kwargs)
#
#     def dump_latent_plots(self, *args, **kwargs):
#         vae_wrapper_util.dump_latent_plots(self, *args, **kwargs)
#
# def temporary_mode(env, mode, func, args=None, kwargs=None):
#     if args is None:
#         args = []
#     if kwargs is None:
#         kwargs = {}
#     cur_mode = env.cur_mode
#     env.mode(env._mode_map[mode])
#     if 'vis_blacklist' in kwargs:
#         env.set_vis_blacklist(kwargs['vis_blacklist'])
#     return_val = func(*args, **kwargs)
#     env.mode(cur_mode)
#     env.set_vis_blacklist([])
#     return return_val

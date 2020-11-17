import multiprocessing
from collections import OrderedDict

import gym
import numpy as np
import torch
from stable_baselines.common.vec_env import VecEnv, CloudpickleWrapper
from stable_baselines.common.tile_images import tile_images
import pickle
import os.path as osp
import utils.torch.pytorch_util as ptu
import cv2,imutils
import time
import csv
import json
import os
VAE_LOAD_PATH = {
    'Image84SawyerPushAndReachArenaTrainEnvBig-v0':'/home/yilin/vae_data/pnr//vae.pkl',
    'Image84SawyerPushAndReachArenaTrainEnvBigUnlimit-v0': '/home/yilin/vae_data/pnr/vae.pkl',

    'Image48PointmassUWallTrainEnvBig-v0':'/home/yilin/vae_data/pm//vae.pkl',
    'Image48PointmassUWallTrainEnvBigUnlimit-v0': '/home/yilin/vae_data/pm/vae.pkl',

}

def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, done, info = env.step(data)
                ## recalculate the done with vae, cannot use the done from the env
                # if done:
                # #     # save final observation where user can get it, then reset
                #     info['terminal_observation'] = observation
                #     info['terminal_state'] = env.get_state()
                ##     observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == 'reset':
                observation = env.reset()
                remote.send(observation)
            elif cmd == 'render':
                remote.send(env.render(*data[0], **data[1]))
            elif cmd == 'get_goal':
                goal = env.get_goal()
                remote.send(goal)
            elif cmd == 'get_obs':
                obs = env.get_obs()
                remote.send(obs)
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError
        except EOFError:
            break


class ParallelVAESubprocVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: ([Gym Environment]) Environments to run in subprocesses
    :param start_method: (str) method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, env_fns, env_id=None,start_method=None,log_dir=None,regressor=None):
        self.waiting = False
        self.closed = False
        self.env_id = env_id
        n_envs = len(env_fns)
        self.total_step_time=0.0
        self.process_obs_time = 0.0
        self.compute_rewsuc_time = 0.0
        self.process_infos_time = 0.0
        self.process_dones_time = 0.0
        self.rewards = [[] for _ in range(n_envs)]
        self.regressor = regressor
        ## adding some logging info here
        self.t_start = time.time()
        filenames = [os.path.join(log_dir, str(rank) + ".update_monitor_aug.csv") for rank in range(n_envs)]
        self.loggers = []
        self.file_handlers = []
        for i in range(n_envs):
            file_handler = open(filenames[i], "wt")
            file_handler.write('#%s\n' % json.dumps({"t_start": self.t_start}))
            logger = csv.DictWriter(file_handler,
                                    fieldnames=('r', 'l', 't') + ('is_success','puck_success','hand_success'))
            logger.writeheader()
            file_handler.flush()
            self.loggers.append(logger)
            self.file_handlers.append(file_handler)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = 'forkserver' in multiprocessing.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'
        ctx = multiprocessing.get_context(start_method)
        ## adding vae model in this part
        if env_id is not None:
            vae_file = open(VAE_LOAD_PATH[env_id], 'rb')
            self.vae_model = pickle.load(vae_file)
            ptu.set_gpu_mode(True)
            self.vae_model.cuda()


        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    # TODO update the step function
    def step_wait(self):
        total_step_time0 = time.time()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        # process the img_obs to latent_obs and process the rewards
        # for info in infos:
        #     info['terminal_observation'] = self.process_obs(info['terminal_observation'])

        if self.env_id is not None:
            # print('aug_env_obs',obs)
            process_obs_time0 = time.time()
            obs_latent = self.process_obs(obs)
            self.process_obs_time += time.time()-process_obs_time0

            # achieved_latent_obs = np.asarray([obs['achieved_goal'] for obs in obs_dicts])
            # desired_latent_obs = np.asarray([obs['desired_goal'] for obs in obs_dicts])
            achieved_latent_obs = obs_latent[:,:self.vae_model.representation_size]
            desired_latent_obs = obs_latent[:,-self.vae_model.representation_size:]
            # print('achieved_latent_obs',achieved_latent_obs.shape)
            # print('desired_latent_obs',desired_latent_obs.shape)
            compute_rewsuc_time0 = time.time()
            rews_and_successes = self.compute_reward_and_success(achieved_goal= achieved_latent_obs,desired_goal=desired_latent_obs,info=True)
            rews = [rew_and_suc[0] for rew_and_suc in rews_and_successes]
            successes = [rew_and_suc[1] for rew_and_suc in rews_and_successes]
            state_infos = [rew_and_suc[2] for rew_and_suc in rews_and_successes]

            for i,rew_and_suc in enumerate(rews_and_successes):
                self.rewards[i].append(rew_and_suc[0])
            for i, info in enumerate(infos):
                infos[i]['is_success'] = successes[i]
            self.compute_rewsuc_time += time.time()-compute_rewsuc_time0
            ## done once succeed wrapper and add terminal_observation in this part
            process_dones_time0 = time.time()
            dones = list(dones)
            for i,done in enumerate(dones):
            # dones = [done or successes[i] for i,done in enumerate(dones)]
                dones[i] = len(self.rewards[i])==100 or successes[i]
                if dones[i]:
                    infos[i]['terminal_observation']=obs[i]
                    infos[i]['terminal_state'] = self.env_method('get_state',indices=i)[0]
                    state_dict = state_infos[i]
                    ## adding info[episode] for loggin the episode mean reward
                    sum_reward = round(sum(self.rewards[i]),6)
                    len_reward = len(self.rewards[i])
                    ep_info = {'r': sum_reward, 'l': len_reward,
                               't': round(time.time() - self.t_start, 6),
                               'is_success': infos[i]['is_success'],
                               'puck_success': state_dict['puck_success'],
                               'hand_success': state_dict['hand_success']}
                    # infos[i]['episode']['r']= round(sum(self.rewards[i]),6)
                    # infos[i]['episode']['l']= len(self.rewards[i])
                    # infos[i]['episode']['is_success']= infos[i]['is_success']
                    infos[i]['episode'] = ep_info

                    self.loggers[i].writerow(infos[i]['episode'])
                    self.file_handlers[i].flush()
                    self.rewards[i] = []
            self.process_dones_time += time.time()-process_dones_time0
        process_infos_time0 = time.time()
        real_infos = self.process_infos(infos)
        self.process_infos_time += time.time() -process_infos_time0
        self.total_step_time += time.time() - total_step_time0
        #    return flatten_latent_obs,np.stack(rews),np.stack(dones),infos
        return obs_latent, np.stack(rews), np.stack(dones), real_infos

    def process_infos_dict(self,infos_dicts):
        img_obs = []
        for infos_dict in infos_dicts:
            if 'terminal_observation' in infos_dict.keys():
                obs_ter = infos_dict['terminal_observation']
                img_stack = np.stack([obs_ter['image_observation'],obs_ter['image_achieved_goal'],obs_ter['image_desired_goal']])
                img_obs.append(img_stack)
        if len(img_obs):
            img_obs_stack= np.concatenate(img_obs)
            latent_obs = self.img_to_latent(img_obs_stack)
            index=0
            for i,infos_dict in enumerate(infos_dicts):
                if 'terminal_observation' in infos_dict.keys():
                    infos_dicts[i]['terminal_observation']['observation']=latent_obs[3*index]
                    infos_dicts[i]['terminal_observation']['achieved_goal']=latent_obs[3*index+1]
                    infos_dicts[i]['terminal_observation']['desired_goal']=latent_obs[3*index+2]
                    index += 1
        return infos_dicts

    def process_infos(self,infos_dicts):
        img_obs = []
        for infos_dict in infos_dicts:
            if 'terminal_observation' in infos_dict.keys():
                obs_ter = infos_dict['terminal_observation']
                # img_stack = np.stack([obs_ter['image_observation'],obs_ter['image_achieved_goal'],obs_ter['image_desired_goal']])
                img_obs.append(obs_ter)
        if len(img_obs):
            img_obs_stack= np.concatenate(img_obs)
            latent_obs = self.img_to_latent(img_obs_stack)
            index=0
            for i,infos_dict in enumerate(infos_dicts):
                if 'terminal_observation' in infos_dict.keys():
                    infos_dicts[i]['terminal_observation'] = np.concatenate(latent_obs[3*index:3*index+3])
            #         infos_dict['terminal_observation']['observation']=latent_obs[3*index]
            #         infos_dict['terminal_observation']['achieved_goal']=latent_obs[3*index+1]
            #         infos_dict['terminal_observation']['desired_goal']=latent_obs[3*index+2]
                    index += 1
        return infos_dicts

    def process_obs_dict(self,obs_dicts):
        img_obs = []
        for obs_dict in obs_dicts:
            img_stack = np.stack([obs_dict['image_observation'],obs_dict['image_achieved_goal'],obs_dict['image_desired_goal']])
            img_obs.append(img_stack)
        img_obs_stack= np.concatenate(img_obs)
        latent_obs = self.img_to_latent(img_obs_stack)
        for i,obs_dict in enumerate(obs_dicts):
            obs_dicts[i]['observation']=latent_obs[3*i]
            obs_dicts[i]['achieved_goal']=latent_obs[3*i+1]
            obs_dicts[i]['desired_goal']=latent_obs[3*i+2]
        return obs_dicts

    def process_obs(self, obs_tuple):
        img_obs_stack = np.stack(obs_tuple)
        latent_obs = self.img_to_latent(img_obs_stack)
        latent_obs = latent_obs.reshape(-1,self.vae_model.representation_size*3)
        return latent_obs


    def img_to_latent(self,img_obs,batch_size=None,noisy=False):

        imgs = img_obs
        # print('torch use the GPU for computation',torch.cuda.is_available())
        if batch_size is None:
            mu, logvar = self.vae_model.encode(ptu.np_to_var(imgs))
        else:
            imgs = imgs.reshape(-1, self.vae_model.imlength)
            n = imgs.shape[0]
            mu, logvar = None, None
            for i in range(0, n, batch_size):
                batch_mu, batch_logvar = self.vae_model.encode(ptu.np_to_var(imgs[i:i + batch_size]))
                if mu is None:
                    mu = batch_mu
                    logvar = batch_logvar
                else:
                    mu = torch.cat((mu, batch_mu), dim=0)
                    logvar = torch.cat((logvar, batch_logvar), dim=0)
        std = logvar.mul(0.5).exp_()
        if noisy:
            eps = ptu.Variable(std.data.new(std.size()).normal_())
            sample = eps.mul(std).add_(mu)
        else:
            sample = mu
        return ptu.get_numpy(sample)

    ## TODO update the reset function
    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        self.rewards = [[] for _ in range(len(self.rewards))]
        if self.env_id is not None:
            # flatten_latent_obs = self.img_to_latent(obs)
            obs = self.process_obs(obs)
        return _flatten_obs(obs, self.observation_space)

    ## TODO update goal-related functions

    def decode_goal(self, latents):
        latents = ptu.np_to_var(latents)
        latents = latents.view(-1, self.vae_model.representation_size)
        decoded = self.vae_model.decode(latents)
        return ptu.get_numpy(decoded)

    def get_goal(self):
        for remote in self.remotes:
            remote.send(('get_goal',None))
        img_goals = [remote.recv() for remote in self.remotes]
        flatten_decode_goals = self.img_to_latent(img_goals)
        return flatten_decode_goals



    def set_goal(self,goals,indices=None):
        img_desired_goals = self.decode_goal(goals)

        if self.regressor is not None:
            state_goals = self.regressor.predict(goals)
            goal_dict = [dict(image_desired_goal=img_desired_goals[i],state_desired_goal=state_goals[i]) for i in range(img_desired_goals.shape[0]) ]
            self.env_method('set_goal',goal_dict,indices=indices)
        else:

            self.env_method('set_goal',img_desired_goals,indices=indices)



    ## TODO update compute_reward function
    ## for the api in sac_augment
    def compute_reward_and_success(self,achieved_goal,desired_goal,info=False,temp_info=None,prev_obs=None,indices=None):
        if isinstance(achieved_goal,list) or isinstance(desired_goal,list):
            achieved_goal= np.concatenate(achieved_goal)
            desired_goal = np.concatenate(desired_goal)
        achieved_goal= achieved_goal.reshape(-1,self.vae_model.representation_size)
        desired_goal = desired_goal.reshape(-1,self.vae_model.representation_size)
        latents_all = np.concatenate((achieved_goal, desired_goal), axis=0)
        # print(achieved_goal.shape,desired_goal.shape,latents_all.shape)
        if self.regressor is not None:
            states_all = self.regressor.predict(latents_all)
            size = states_all.shape[0] // 2
            achieved_state = states_all[:size]
            desired_state = states_all[-size:]
            rews_and_sucesses = self.env_method('compute_state_reward_and_success', achieved_state, desired_state,info=info,
                                                indices=indices)
        else:
            image_all = self.decode_goal(latents_all)
            size = image_all.shape[0]//2
            achieved_state_img = image_all[:size]
            desired_state_img = image_all[-size:]
            rews_and_sucesses= self.env_method('compute_img_reward_and_success',achieved_state_img,desired_state_img,indices=indices)
        # for remote in self.remotes:
        #     remote.send(('compute_reward_and_sucess',achieved_state_img,desired_state_img))
        # for remote in self.remotes:
        #     reward,success = remote.recv()
        #     rews.append(reward)
        #     successes.append(success)

        return rews_and_sucesses

    def compute_reward(self,achieved_goal,desired_goal,temp_info=None,prev_obs=None,indices=None):
        rews_and_successes= self.compute_reward_and_success(achieved_goal=achieved_goal,desired_goal=desired_goal,temp_info=temp_info,prev_obs=prev_obs,indices=indices)
        rews = [rew_and_suc[0] for rew_and_suc in rews_and_successes]
        return rews

    ## TODO update obs and info
    def get_obs(self,indices=None):
        if indices is not None:
            target_remotes = self._get_target_remotes(indices)
        else :
            target_remotes = self.remotes
        for remote in target_remotes:
            remote.send(('get_obs',None))
        obs = [remote.recv() for remote in target_remotes]
        if self.env_id is not None:
            # flatten_latent_obs = self.img_to_latent(obs)
            # return flatten_latent_obs
            obs = self.process_obs_dict(obs)
        return _flatten_obs(obs, self.observation_space)

    # def update_info(self):
    #     # def _update_info(self, info, obs):
    #     latent_obs, logvar = self.vae.encode(
    #         ptu.np_to_var(obs[self.vae_input_observation_key])
    #     )
    #     latent_obs, logvar = ptu.get_numpy(latent_obs)[0], ptu.get_numpy(logvar)[0]
    #     if not self.noisy_encoding:
    #         assert (latent_obs == obs['latent_observation']).all()
    #     latent_goal = self.desired_goal['latent_desired_goal']
    #     dist = latent_goal - latent_obs
    #     info["vae_dist"] = np.linalg.norm(dist, ord=self.norm_order)
    #     info["vae_dist_l1"] = np.linalg.norm(dist, ord=1)
    #     info["vae_dist_l2"] = np.linalg.norm(dist, ord=2)


    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def render(self, mode='human', *args, **kwargs):
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(('render', (args, {'mode': 'rgb_array', **kwargs})))
        imgs = [pipe.recv() for pipe in self.remotes]
        # Create a big image by tiling images from subprocesses
        bigimg = tile_images(imgs)
        if mode == 'human':
            import cv2
            cv2.imshow('vecenv', bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        for pipe in self.remotes:
            pipe.send(('render', {"mode": 'rgb_array'}))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        assert isinstance(value, list) or isinstance(value, tuple) or isinstance(value, np.ndarray)
        target_remotes = self._get_target_remotes(indices)
        for i, remote in enumerate(target_remotes):
            remote.send(('set_attr', (attr_name, value[i])))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        # TODO: dispatch method_kwargs. Now method_kwargs is the same for every remote
        dispatched_args = [[] for _ in range(len(target_remotes))]
        for args in method_args:
            assert isinstance(args, list) or isinstance(args, tuple) or isinstance(args, np.ndarray), type(args)
            for i in range(len(target_remotes)):
                dispatched_args[i].append(args[i])
        for i, remote in enumerate(target_remotes):
            remote.send(('env_method', (method_name, dispatched_args[i], method_kwargs)))
        return [remote.recv() for remote in target_remotes]


    # def set_attr(self, attr_name, value, indices=None):
    #     """Set attribute inside vectorized environments (see base class)."""
    #     target_remotes = self._get_target_remotes(indices)
    #     for remote in target_remotes:
    #         remote.send(('set_attr', (attr_name, value)))
    #     for remote in target_remotes:
    #         remote.recv()
    #
    #
    #
    #
    # def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
    #     """Call instance methods of vectorized environments."""
    #     target_remotes = self._get_target_remotes(indices)
    #     for remote in target_remotes:
    #         remote.send(('env_method', (method_name, method_args, method_kwargs)))
    #     return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices):
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: ([multiprocessing.Connection]) Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


def _flatten_obs(obs, space):
    """
    Flatten observations, depending on the observation space.

    :param obs: (list<X> or tuple<X> where X is dict<ndarray>, tuple<ndarray> or ndarray) observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return (OrderedDict<ndarray>, tuple<ndarray> or ndarray) flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple((np.stack([o[i] for o in obs]) for i in range(obs_len)))
    else:
        return np.stack(obs)

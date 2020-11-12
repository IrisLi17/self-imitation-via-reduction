from baselines import PPO2_augment, PPO2_augment_sil
from stable_baselines import logger
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from utils.parallel_subproc_vec_env import ParallelSubprocVecEnv
from gym.wrappers import FlattenDictWrapper
from stable_baselines.common.policies import register_policy

from push_wall_obstacle import FetchPushWallObstacleEnv_v4
from masspoint_env import MasspointPushSingleObstacleEnv_v2, MasspointPushDoubleObstacleEnv
from masspoint_env import MasspointMazeEnv, MasspointSMazeEnv
from fetch_stack import FetchStackEnv
# from push_wall import FetchPushWallEnv
# from push_box import FetchPushBoxEnv
import gym
from utils.wrapper import DoneOnSuccessWrapper,VAEWrappedEnv
import pickle
import numpy as np
import csv, pickle
import multiworld
import os, time, argparse, imageio
import matplotlib.pyplot as plt


ENTRY_POINT = {'FetchPushWallObstacle-v4': FetchPushWallObstacleEnv_v4,
               'FetchPushWallObstacleUnlimit-v4': FetchPushWallObstacleEnv_v4,
               # 'FetchPushWall-v1': FetchPushWallEnv,
               # 'FetchPushBox-v1': FetchPushBoxEnv,
               }
MASS_ENTRY_POINT = {
    'MasspointPushSingleObstacle-v2': MasspointPushSingleObstacleEnv_v2,
    'MasspointPushSingleObstacleUnlimit-v2': MasspointPushSingleObstacleEnv_v2,
    'MasspointPushDoubleObstacle-v1': MasspointPushDoubleObstacleEnv,
    'MasspointPushDoubleObstacleUnlimit-v1': MasspointPushDoubleObstacleEnv,
    'MasspointMaze-v1': MasspointMazeEnv,
    'MasspointMazeUnlimit-v1': MasspointMazeEnv,
    'MasspointMaze-v2': MasspointSMazeEnv,
    'MasspointMazeUnlimit-v2': MasspointSMazeEnv,
}

PICK_ENTRY_POINT = {
    'FetchStack-v1': FetchStackEnv,
    'FetchStackUnlimit-v1': FetchStackEnv,
}
IMAGE_ENTRY_POINT = {
    'Image84SawyerPushAndReachArenaTrainEnvBig-v0':  'ImagePushAndReach',
    'Image84SawyerPushAndReachArenaTrainEnvBigUnlimit-v0':  'ImagePushAndReach',
    'Image48PointmassUWallTrainEnvBig-v0':'ImageUWall',
    'Image48PointmassUWallTrainEnvBigUnlimit-v0': 'ImageUWall',

}
VAE_LOAD_PATH = {
    'Image84SawyerPushAndReachArenaTrainEnvBig-v0':'/home/yilin/leap/data/pnr/09-20-train-vae-local/09-20-train-vae-local_2020_09_20_16_10_33_id000--s85192/vae.pkl',
    'Image84SawyerPushAndReachArenaTrainEnvBigUnlimit-v0': '/home/yilin/leap/data/pnr/09-20-train-vae-local/09-20-train-vae-local_2020_09_20_16_10_33_id000--s85192/vae.pkl',

    'Image48PointmassUWallTrainEnvBig-v0':'/home/yilin/leap/data/pm/09-20-train-vae-local/09-20-train-vae-local_2020_09_20_22_23_14_id000--s4047/vae.pkl',
    'Image48PointmassUWallTrainEnvBigUnlimit-v0': '/home/yilin/leap/data/pm/09-20-train-vae-local/09-20-train-vae-local_2020_09_20_22_23_14_id000--s4047/vae.pkl',

}
def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)

def create_image_84_sawyer_pnr_arena_train_env_big_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_tdm_v4

    wrapped_env = gym.make('SawyerPushAndReachArenaTrainEnvBig-v0')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=sawyer_pusher_camera_tdm_v4,
        transpose=True,
        normalize=True,
        reward_type='sparse'
    )
def create_image_48_pointmass_uwall_train_env_big_v0():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassUWallTrainEnvBig-v0')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )

def make_env(env_id, seed, rank, log_dir=None, allow_early_resets=True, kwargs=None):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param allow_early_resets: (bool) allows early reset of the environment
    :return: (Gym Environment) The mujoco environment
    """
    if env_id in ENTRY_POINT.keys() or env_id in MASS_ENTRY_POINT.keys() or env_id in PICK_ENTRY_POINT.keys() or env_id in IMAGE_ENTRY_POINT.keys():
        # env = ENTRY_POINT[env_id](**kwargs)
        # print(env)
        # from gym.wrappers.time_limit import TimeLimit
        kwargs = kwargs.copy()
        max_episode_steps = None
        if env_id in IMAGE_ENTRY_POINT.keys():
            if IMAGE_ENTRY_POINT[env_id] == 'ImagePushAndReach':
                gym.register(
                    id=env_id,
                    entry_point=create_image_84_sawyer_pnr_arena_train_env_big_v0,
                    max_episode_steps=max_episode_steps,
                    tags={
                        'git-commit-hash': 'e5c11ac',
                        'author': 'Soroush'
                    },
                )
            elif IMAGE_ENTRY_POINT[env_id] =='ImageUWall':
                gym.register(
                    id=env_id,
                    entry_point=create_image_48_pointmass_uwall_train_env_big_v0,
                    tags={
                        'git-commit-hash': 'e5c11ac',
                        'author': 'Soroush'
                    },
                )
            env = gym.make(env_id)
            # env.wrapped_env.reward_type='sparse'
            vae_file = open(VAE_LOAD_PATH[env_id],'rb')
            import utils.torch.pytorch_util as ptu
            ptu.set_gpu_mode(True)
            vae_model = pickle.load(vae_file)
            env = VAEWrappedEnv(env,vae_model,epsilon=1.0,imsize=48,reward_params=dict(type='latent_distance2'))
            print(env.observation_space.spaces.keys())
            # env.wrapped_env.reward_type='wrapped_env'
            # env.reward_type='latent_sparse'
        else:
            if 'max_episode_steps' in kwargs:
                max_episode_steps = kwargs['max_episode_steps']
                del kwargs['max_episode_steps']
            if env_id in ENTRY_POINT.keys():
                gym.register(env_id, entry_point=ENTRY_POINT[env_id], max_episode_steps=max_episode_steps, kwargs=kwargs)
            elif env_id in MASS_ENTRY_POINT.keys():
                gym.register(env_id, entry_point=MASS_ENTRY_POINT[env_id], max_episode_steps=max_episode_steps, kwargs=kwargs)
            elif env_id in PICK_ENTRY_POINT.keys():
                gym.register(env_id, entry_point=PICK_ENTRY_POINT[env_id], max_episode_steps=max_episode_steps,
                             kwargs=kwargs)
            env = gym.make(env_id)


        # env = TimeLimit(env, max_episode_steps=50)
    else:
        env = gym.make(env_id, reward_type='sparse')
    # env = FlattenDictWrapper(env, ['observation', 'achieved_goal', 'desired_goal'])
    env = FlattenDictWrapper(env,['latent_observation','latent_achieved_goal','latent_desired_goal'])
    if env_id in PICK_ENTRY_POINT.keys() and kwargs['reward_type'] == 'dense':
        env = DoneOnSuccessWrapper(env, reward_offset=0.0)
    else:
        env = DoneOnSuccessWrapper(env)
    if log_dir is not None:
        env = Monitor(env, os.path.join(log_dir, str(rank) + ".monitor.csv"), allow_early_resets=allow_early_resets,
                      info_keywords=('is_success',))
    # env.seed(seed + 10000 * rank)
    return env


def main(env_name, seed, num_timesteps, log_path, load_path, play, export_gif, random_ratio, aug_clip, n_subgoal,
         parallel, start_augment, reuse_times, aug_adv_weight, reward_type, n_object, curriculum, self_imitate, sil_clip, policy):
    log_dir = log_path if (log_path is not None) else "/tmp/stable_baselines_" + time.strftime('%Y-%m-%d-%H-%M-%S')
    configure_logger(log_dir)

    set_global_seeds(seed)

    n_cpu = 1 if not play else 1
    if 'MasspointPushDoubleObstacle' in env_name:
        n_cpu = 64 if not play else 1
    elif 'FetchStack' in env_name:
        n_cpu = 128 if not play else 1
    elif 'MasspointMaze' in env_name:
        n_cpu = 8 if not play else 1
    if env_name in ['FetchReach-v1', 'FetchPush-v1', 'CartPole-v1']:
        env_kwargs = dict(reward_type='dense')
        # pass
    elif env_name in ENTRY_POINT.keys():
        env_kwargs = dict(random_box=True,
                          heavy_obstacle=True,
                          random_ratio=random_ratio,
                          random_gripper=True,
                          max_episode_steps=100,)
    elif env_name in MASS_ENTRY_POINT.keys():
        env_kwargs = dict(random_box=True,
                          random_ratio=random_ratio,
                          random_pusher=True,
                          max_episode_steps=100,)
        if 'MasspointPushSingleObstacle' in env_name:
            env_kwargs['max_episode_steps']=200
        if 'MasspointPushDoubleObstacle' in env_name:
            env_kwargs['max_episode_steps']=150
    elif env_name in PICK_ENTRY_POINT.keys():
        env_kwargs = dict(random_box=True,
                          random_ratio=random_ratio,
                          random_gripper=True,
                          max_episode_steps=100,
                          reward_type=reward_type,
                          n_object=n_object, )
    elif env_name in IMAGE_ENTRY_POINT.keys():
        env_kwargs = {}
    else:
        raise NotImplementedError("%s not implemented" % env_name)

    def make_thunk(rank):
        return lambda: make_env(env_id=env_name, seed=seed, rank=rank, log_dir=log_dir, kwargs=env_kwargs)

    # if not parallel:
    env = SubprocVecEnv([make_thunk(i) for i in range(n_cpu)])
    # else:
    #     env = ParallelSubprocVecEnv([make_thunk(i) for i in range(n_cpu)])

    aug_env_name = env_name.split('-')[0] + 'Unlimit-' + env_name.split('-')[1]
    aug_env_kwargs = env_kwargs.copy()
    aug_env_kwargs['max_episode_steps'] = None
    def make_thunk_aug(rank):
        return lambda: make_env(env_id=aug_env_name, seed=seed, rank=rank, kwargs=aug_env_kwargs)
    if not parallel:
        aug_env = make_env(env_id=aug_env_name, seed=seed, rank=0, kwargs=aug_env_kwargs)
    else:
        # aug_env = ParallelSubprocVecEnv([make_thunk_aug(i) for i in range(n_subgoal)])
        if 'FetchStack' in env_name:
            aug_env = ParallelSubprocVecEnv([make_thunk_aug(i) for i in range(32)])
        else:
            aug_env = ParallelSubprocVecEnv([make_thunk_aug(i) for i in range(min(32, n_cpu))])
    print(aug_env)
    if os.path.exists(os.path.join(logger.get_dir(), 'eval.csv')):
        os.remove(os.path.join(logger.get_dir(), 'eval.csv'))
        print('Remove existing eval.csv')
    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs['random_ratio'] = 0.0
    eval_env = make_env(env_id=env_name, seed=seed, rank=0, kwargs=eval_env_kwargs)
    # print(eval_env)
    # print(eval_env.unwrapped)
    # print(eval_env.reward_type)
    # print(eval_env.env.env.wrapped_env.reward_type)
    # print(eval_env.env.env.reward_type)
    # print(eval_env.env.env.wrapped_env.wrapped_env.reward_type)
    aug_env.reset()
    obs,reward,done,info = aug_env.step(aug_env.action_space.sample())
    print(obs.shape,obs)
    aug_env.env_method('set_goal',obs[:,:16])
    print('aug_env_reward:',reward)
    # print('aug_env_reward_type',env.get_attr('reward_type'))
    # eval_env.reset()
    # obs,reward,done,info = eval_env.step(eval_env.action_space.sample())
    # print('eval_env_reward:',reward)
    # print(obs)
    # print(obs.shape)
    # print(env.env_method('get_goal')[0]['latent_desired_goal'].shape[0])
    # print(env.observation_space.shape[0])
    # print(env.get_attr('goal_dim')[0])
    # # print(env.get_attr('desired_goal')[0])
    # # print(env.env_method('set_goal',np.ones((16,))))
    # print(env.get_attr('desired_goal')[0])
    # print(env.env_method('get_state')[0])
    # print(env.env_method('get_obs')[0])
    # print(env.get_attr('latent_obs')[0])
    # print(env.env_method('get_env_state'))
    # state,goal = env.env_method('get_env_state')
    # env.env_method('set_env_state',state,goal)
    # draw the plot
    # assert load_path is not None
    # model = PPO2_augment.load(load_path)
    # env_render = eval_env
    # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # obs = env_render.reset()
    # # while (np.argmax(obs[0][-2:]) != 0):
    # #     obs = env.reset()
    # # img = env.render(mode='rgb_array')
    # load_path = logger.get_dir()
    # episode_reward = 0.0
    # num_episode = 0
    # frame_idx = 0
    # images = []
    # for i in range(500):
    #     print(env.unwrapped)
    #     print(env_render.env.env.wrapped_env.wrapped_env)
    #     # img = env_render.env.env.wrapped_env.wrapped_env.render()
    #     img = env_render.get_image()
    #     images.append(img)
    #     ax.cla()
    #     ax.imshow(img)
    #     # ax.set_title('episode ' + str(num_episode) + ', frame ' + str(frame_idx) +
    #     #              ', goal idx ' + str(np.argmax(obs[0][-2:])))
    #     ax.set_title('episode'+str(num_episode)+', frame'+str(frame_idx))
    #     # assert np.argmax(obs[0][-2:]) == 0
    #     # action, _ = model.predict(obs)
    #     action = env_render.action_space.sample()
    #     print('action', action)
    #     obs, reward, done, _ = env_render.step(action)
    #     episode_reward += reward
    #     frame_idx += 1
    #     if not export_gif:
    #         plt.pause(0.1)
    #     else:
    #         plt.savefig(os.path.join(os.path.dirname(load_path), 'tempimg%d.png' % i))
    #     if done:
    #         obs = env_render.reset()
    #         # while (np.argmax(obs[0][-2:]) != 0):
    #         #     obs = env_render.reset()
    #         print('episode_reward', episode_reward)
    #         episode_reward = 0.0
    #         frame_idx = 0
    #         num_episode += 1
    #         if num_episode >= 5:
    #             break
    # # imageio.mimsave(env_name + '.gif', images)
    # if export_gif:
    #     os.system('ffmpeg -r 5 -start_number 0 -i ' + os.path.dirname(
    #         load_path) + '/tempimg%d.png -c:v libx264 -pix_fmt yuv420p ' +
    #               os.path.join(os.path.dirname(load_path), env_name + '.mp4'))
    #     for i in range(500):
    #         # images.append(plt.imread('tempimg' + str(i) + '.png'))
    #         try:
    #             os.remove(os.path.join(os.path.dirname(load_path), 'tempimg' + str(i) + '.png'))
    #         except:
    #             pass

#    return
    # print(eval_env.goal.shape[0], eval_env.n_object)
    # if not play:
    #     os.makedirs(log_dir, exist_ok=True)
    #     policy_kwargs = dict(layers=[256, 256])
    #     # policy_kwargs = {}
    #     # TODO: vectorize env
    #     if 'MasspointPushDoubleObstacle' in env_name or 'FetchStack' in env_name:
    #         n_steps = 8192
    #     elif 'MasspointMaze' in env_name:
    #         n_steps = 1024
    #     else:
    #         n_steps = 2048
    #     # policy = 'MlpPolicy'
    #     from utils.attention_policy import AttentionPolicy
    #     register_policy('AttentionPolicy', AttentionPolicy)
    #     if 'FetchStack' in env_name:
    #         policy = 'AttentionPolicy' # Force attention policy for fetchstack env
    #         policy_kwargs["n_object"] = n_object
    #         policy_kwargs["feature_extraction"] = "attention_mlp"
    #     elif 'MasspointPushDoubleObstacle' in env_name:
    #         if policy == "AttentionPolicy":
    #             policy_kwargs["feature_extraction"] = "attention_mlp_particle"
    #     if 'FetchStack' in env_name:
    #         dim_candidate = 3
    #     else:
    #         dim_candidate = 2


def arg_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', default='FetchPushWallObstacle-v4')
    parser.add_argument('--policy', type=str, default='MlpPolicy')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_timesteps', type=float, default=1e8)
    parser.add_argument('--log_path', default=None, type=str)
    parser.add_argument('--load_path', default=None, type=str)
    parser.add_argument('--random_ratio', default=1.0, type=float)
    parser.add_argument('--aug_clip', default=0.1, type=float)
    parser.add_argument('--aug_adv_weight', default=1.0, type=float)
    parser.add_argument('--n_subgoal', default=4, type=int)
    parser.add_argument('--parallel', action="store_true", default=False)
    parser.add_argument('--self_imitate', action="store_true", default=False)
    parser.add_argument('--sil_clip', default=0.2, type=float)
    parser.add_argument('--start_augment', type=float, default=0)
    parser.add_argument('--reuse_times', default=1, type=int)
    parser.add_argument('--reward_type', default="sparse", type=str)
    parser.add_argument('--n_object', default=2, type=int)
    parser.add_argument('--curriculum', action="store_true", default=False)
    parser.add_argument('--play', action="store_true", default=False)
    parser.add_argument('--export_gif', action="store_true", default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    print('arg parsed')
    main(env_name=args.env, seed=args.seed, num_timesteps=int(args.num_timesteps),
         log_path=args.log_path, load_path=args.load_path, play=args.play, export_gif=args.export_gif,
         random_ratio=args.random_ratio, aug_clip=args.aug_clip, n_subgoal=args.n_subgoal,
         parallel=args.parallel, start_augment=int(args.start_augment), reuse_times=args.reuse_times,
         aug_adv_weight=args.aug_adv_weight, reward_type=args.reward_type, n_object=args.n_object,
        curriculum=args.curriculum, self_imitate=args.self_imitate,
         policy=args.policy, sil_clip=args.sil_clip
    )





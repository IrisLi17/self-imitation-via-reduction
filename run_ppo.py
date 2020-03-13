from baselines import PPO2
from stable_baselines import logger
from stable_baselines.bench import Monitor
from stable_baselines.common.policies import register_policy
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from gym.wrappers import FlattenDictWrapper
from run_ppo_augment import eval_model, log_eval, stack_eval_model

from push_wall_obstacle import FetchPushWallObstacleEnv_v4
# from push_wall_double_obstacle import FetchPushWallDoubleObstacleEnv
from masspoint_env import MasspointPushDoubleObstacleEnv, MasspointPushSingleObstacleEnv, MasspointPushSingleObstacleEnv_v2
from masspoint_env import MasspointMazeEnv, MasspointSMazeEnv
from fetch_stack import FetchStackEnv, FetchPureStackEnv
# from push_wall import FetchPushWallEnv
# from push_box import FetchPushBoxEnv
import gym
from utils.wrapper import DoneOnSuccessWrapper, ScaleRewardWrapper
import numpy as np

import os, time, argparse, imageio
import matplotlib.pyplot as plt

ENTRY_POINT = {'FetchPushWallObstacle-v4': FetchPushWallObstacleEnv_v4,
               'FetchPushWallObstacleUnlimit-v4': FetchPushWallObstacleEnv_v4,
               # 'FetchPushWallDoubleObstacle-v1': FetchPushWallDoubleObstacleEnv,
               # 'FetchPushWall-v1': FetchPushWallEnv,
               # 'FetchPushBox-v1': FetchPushBoxEnv,
               }

MASS_ENTRY_POINT = {
    'MasspointPushSingleObstacle-v1': MasspointPushSingleObstacleEnv,
    'MasspointPushSingleObstacle-v2': MasspointPushSingleObstacleEnv_v2,
    'MasspointPushDoubleObstacle-v1': MasspointPushDoubleObstacleEnv,
    'MasspointMaze-v1': MasspointMazeEnv,
    'MasspointMaze-v2': MasspointSMazeEnv,
}

PICK_ENTRY_POINT = {
    'FetchStack-v1': FetchStackEnv,
    'FetchStack-v0': FetchPureStackEnv,
}

def arg_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', default='FetchPushWallObstacle-v4')
    parser.add_argument('--policy', type=str, default='MlpPolicy')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_timesteps', type=float, default=1e8)
    parser.add_argument('--reward_type', type=str, default='sparse')
    parser.add_argument('--n_object', type=int, default=2) # Only used for stacking
    parser.add_argument('--log_path', default=None, type=str)
    parser.add_argument('--load_path', default=None, type=str)
    parser.add_argument('--random_ratio', default=1.0, type=float)
    parser.add_argument('--curriculum', action="store_true", default=False)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--play', action="store_true", default=False)
    parser.add_argument('--export_gif', action="store_true", default=False)
    args = parser.parse_args()
    return args

def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)

def make_env(env_id, seed, rank, log_dir=None, allow_early_resets=True, kwargs=None):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param allow_early_resets: (bool) allows early reset of the environment
    :return: (Gym Environment) The mujoco environment
    """
    if env_id in ENTRY_POINT.keys() or env_id in MASS_ENTRY_POINT.keys() or env_id in PICK_ENTRY_POINT.keys():
        # env = ENTRY_POINT[env_id](**kwargs)
        # print(env)
        # from gym.wrappers.time_limit import TimeLimit
        kwargs = kwargs.copy()
        max_episode_steps = None
        if 'max_episode_steps' in kwargs:
            max_episode_steps = kwargs['max_episode_steps']
            del kwargs['max_episode_steps']
        if env_id in ENTRY_POINT.keys():
            gym.register(env_id, entry_point=ENTRY_POINT[env_id], max_episode_steps=max_episode_steps, kwargs=kwargs)
        elif env_id in MASS_ENTRY_POINT.keys():
            gym.register(env_id, entry_point=MASS_ENTRY_POINT[env_id], max_episode_steps=max_episode_steps, kwargs=kwargs)
        elif env_id in PICK_ENTRY_POINT.keys():
            gym.register(env_id, entry_point=PICK_ENTRY_POINT[env_id], max_episode_steps=max_episode_steps, kwargs=kwargs)
        env = gym.make(env_id)
        # env = TimeLimit(env, max_episode_steps=50)
    else:
        env = gym.make(env_id, reward_type='sparse')
    env = FlattenDictWrapper(env, ['observation', 'achieved_goal', 'desired_goal'])
    if env_id in PICK_ENTRY_POINT.keys() and (kwargs['reward_type'] == 'dense' or kwargs['reward_type'] == 'incremental'):
        env = DoneOnSuccessWrapper(env, reward_offset=0.0)
        env = ScaleRewardWrapper(env, reward_scale=100.0)
    else:
        env = DoneOnSuccessWrapper(env)
    if log_dir is not None:
        env = Monitor(env, os.path.join(log_dir, str(rank) + ".monitor.csv"), allow_early_resets=allow_early_resets, info_keywords=('is_success',))
    # env.seed(seed + 10000 * rank)
    return env

def main(env_name, seed, num_timesteps, log_path, load_path, play, export_gif, random_ratio, reward_type, n_object,
         curriculum, gamma, policy):
    log_dir = log_path if (log_path is not None) else "/tmp/stable_baselines_" + time.strftime('%Y-%m-%d-%H-%M-%S')
    configure_logger(log_dir) 
    
    set_global_seeds(seed)

    n_cpu = 32 if not play else 1
    if 'MasspointPushDoubleObstacle' in env_name:
        n_cpu = 64 if not play else 1
    elif 'FetchStack' in env_name:
        n_cpu = 128 if not play else 1
    elif 'MasspointMaze' in env_name:
        n_cpu = 8 if not play else 1
    if env_name in ['FetchReach-v1', 'FetchPush-v1', 'CartPole-v1', 'FetchPickAndPlace-v1']:
        env_kwargs = {}
        # pass
    elif env_name in ENTRY_POINT.keys():
        env_kwargs = dict(random_box=True,
                          heavy_obstacle=True,
                          random_ratio=random_ratio,
                          random_gripper=True,
                          max_episode_steps=100, )
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
    else:
        raise NotImplementedError("%s not implemented" % env_name)
    def make_thunk(rank):
        return lambda: make_env(env_id=env_name, seed=seed, rank=rank, log_dir=log_dir, kwargs=env_kwargs)
    env = SubprocVecEnv([make_thunk(i) for i in range(n_cpu)])
    if env_name in ENTRY_POINT.keys():
        eval_env_kwargs = dict(random_box=True,
                               heavy_obstacle=True,
                               random_ratio=0.0,
                               random_gripper=True,
                               max_episode_steps=100, )
    elif env_name in MASS_ENTRY_POINT.keys():
        eval_env_kwargs = dict(random_box=True,
                               random_ratio=0.0,
                               random_pusher=True,
                               max_episode_steps=100,)
        if 'MasspointPushDoubleObstacle' in env_name:
            env_kwargs['max_episode_steps']=150
    elif env_name in PICK_ENTRY_POINT.keys():
        eval_env_kwargs = dict(random_box=True,
                               random_ratio=0.0,
                               random_gripper=True,
                               max_episode_steps=100,
                               reward_type=reward_type,
                               n_object=n_object, )
    elif env_name in ['FetchPickAndPlace-v1']:
        eval_env_kwargs = {}
    eval_env = make_env(env_id=env_name, seed=seed, rank=0, kwargs=eval_env_kwargs)
    print(eval_env)
    if not play:
        os.makedirs(log_dir, exist_ok=True)
        policy_kwargs = dict(layers=[256, 256])
        # if 'FetchStack' in env_name:
        #     policy_kwargs = dict(layers=[512, 512])
        # policy_kwargs = {}
        # TODO: vectorize env
        n_steps = 2048
        if 'MasspointPushDoubleObstacle' in env_name or 'FetchStack' in env_name:
            n_steps = 8192
        elif 'MasspointMaze' in env_name:
            n_steps = 1024

        # policy = 'MlpPolicy'
        from utils.attention_policy import AttentionPolicy
        register_policy('AttentionPolicy', AttentionPolicy)
        if 'FetchStack' in env_name:
            # from utils.attention_policy import AttentionPolicy
            # policy = AttentionPolicy
            policy = "AttentionPolicy" # Force attention policy for fetchstack env
            policy_kwargs["n_object"] = n_object
            policy_kwargs["feature_extraction"] = "attention_mlp"
        elif 'MasspointPushDoubleObstacle' in env_name:
            # from utils.attention_policy import AttentionPolicy
            # policy = AttentionPolicy
            if policy == "AttentionPolicy":
                policy_kwargs["feature_extraction"] = "attention_mlp_particle"
        print(policy_kwargs)

        model = PPO2(policy, env, eval_env, verbose=1, n_steps=n_steps, nminibatches=32, lam=0.95, gamma=gamma, noptepochs=10,
                     ent_coef=0.01, learning_rate=3e-4, cliprange=0.2, policy_kwargs=policy_kwargs,
                     curriculum=curriculum,
                     )
        print(model.get_parameter_list())
        def callback(_locals, _globals):
            num_update = _locals["update"]
            if 'FetchStack' in env_name:
                mean_eval_reward = stack_eval_model(eval_env, _locals["self"])
            else:
                mean_eval_reward = eval_model(eval_env, _locals["self"])
            log_eval(num_update, mean_eval_reward)
            if num_update % 10 == 0:
                model_path = os.path.join(log_dir, 'model_' + str(num_update // 10))
                model.save(model_path)
                print('model saved to', model_path)
            return True
        model.learn(total_timesteps=num_timesteps, callback=callback, seed=seed, log_interval=1)
        model.save(os.path.join(log_dir, 'final'))
    
    else:
        assert load_path is not None
        model = PPO2.load(load_path)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        obs = env.reset()
        goal_dim = env.get_attr('goal')[0].shape[0]
        if 'FetchStack' in env_name:
            while env.get_attr('current_nobject')[0] != env.get_attr('n_object')[0] or env.get_attr('task_mode')[0] != 1:
                obs = env.reset()
        else:
            if 'FetchPush' in env_name:
                obs = env.reset()
                while not (obs[0][6] > 1.25 and obs[0][6] < 1.33 and obs[0][7] < 0.61 and obs[0][4] > 0.7 and obs[0][4] < 0.8):
                    obs = env.reset()
                env.env_method('set_goal', np.array([1.2, 0.75, 0.425, 1, 0]))
                obs = env.env_method('get_obs')
                obs[0] = np.concatenate([obs[0][key] for key in ['observation', 'achieved_goal', 'desired_goal']])
            if 'MasspointPush' in env_name:
                obs = env.reset()
                while not (obs[0][3] < 1.5 and obs[0][0] < 2.8 and np.argmax(obs[0][-goal_dim+3:] == 0)):
                    obs = env.reset()
                obs = env.env_method('get_obs')
                obs[0] = np.concatenate([obs[0][key] for key in ['observation', 'achieved_goal', 'desired_goal']])
            while np.argmax(obs[0][-goal_dim+3:]) != 0:
                obs = env.reset()
        print('goal', obs[0][-goal_dim:])
        # while (obs[0][3] - 1.25) * (obs[0][6] - 1.25) < 0:
        #     obs = env.reset()
        # img = env.render(mode='rgb_array')
        episode_reward = 0.0
        num_episode = 0
        frame_idx = 0
        images = []
        if not 'max_episode_steps' in env_kwargs.keys():
            env_kwargs['max_episode_steps'] = 100
        for i in range(env_kwargs['max_episode_steps'] * 10):
            img = env.render(mode='rgb_array')
            ax.cla()
            ax.imshow(img)
            if env.get_attr('goal')[0].shape[0] <= 3:
                ax.set_title('episode ' + str(num_episode) + ', frame ' + str(frame_idx))
            else:
                ax.set_title('episode ' + str(num_episode) + ', frame ' + str(frame_idx) +
                             ', goal idx ' + str(np.argmax(env.get_attr('goal')[0][3:])))
                if 'FetchStack' in env_name:
                    tasks = ['pick and place', 'stack']
                    ax.set_title('episode ' + str(num_episode) + ', frame ' + str(frame_idx)
                            + ', task: ' + tasks[np.argmax(obs[0][-2*goal_dim-2:-2*goal_dim])])
            images.append(img)
            action, _ = model.predict(obs)
            # print('action', action)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            frame_idx += 1
            if not export_gif:
                plt.pause(0.1)
            else:
                plt.imsave(os.path.join(os.path.dirname(load_path), 'tempimg%d.png' % i), img)
                # plt.savefig(os.path.join(os.path.dirname(load_path), 'tempimg%d.png' % i))
            if done:
                # obs = env.reset()
                # while (obs[0][3] - 1.25) * (obs[0][6] - 1.25) < 0:
                #     obs = env.reset()
                print('episode_reward', episode_reward)
                if 'FetchStack' in env_name:
                    while env.get_attr('current_nobject')[0] != env.get_attr('n_object')[0] or env.get_attr('task_mode')[0] != 1:
                        obs = env.reset()
                else:
                    while np.argmax(obs[0][-goal_dim + 3:]) != 0:
                        obs = env.reset()
                print('goal', obs[0][-goal_dim:])
                episode_reward = 0.0
                frame_idx = 0
                num_episode += 1
                if num_episode >= 10:
                    break
        # imageio.mimsave(env_name + '.gif', images)
        # exit()
        if export_gif:
            os.system('ffmpeg -r 5 -start_number 0 -i ' + os.path.dirname(load_path) + '/tempimg%d.png -c:v libx264 -pix_fmt yuv420p ' +
                      os.path.join(os.path.dirname(load_path), env_name + '.mp4'))
            for i in range(env_kwargs['max_episode_steps'] * 10):
                # images.append(plt.imread('tempimg' + str(i) + '.png'))
                try:
                    os.remove(os.path.join(os.path.dirname(load_path), 'tempimg' + str(i) + '.png'))
                except:
                    pass

if __name__ == '__main__':
    args = arg_parse()
    print('arg parsed')
    main(env_name=args.env, seed=args.seed, num_timesteps=int(args.num_timesteps), 
         log_path=args.log_path, load_path=args.load_path, play=args.play, export_gif=args.export_gif,
         random_ratio=args.random_ratio, reward_type=args.reward_type, n_object=args.n_object,
         curriculum=args.curriculum, gamma=args.gamma, policy=args.policy)

from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy
from stable_baselines.common.policies import register_policy

from utils.make_env_utils import configure_logger, make_env, get_env_kwargs, get_train_kwargs, get_policy_kwargs
from utils.parallel_subproc_vec_env import ParallelSubprocVecEnv
from baselines import HER_HACK, SAC_parallel
from utils.log_utils import eval_model, log_eval, stack_eval_model, egonav_eval_model
from gym.wrappers import FlattenDictWrapper
import matplotlib.pyplot as plt
from stable_baselines.common import set_global_seeds
from stable_baselines import logger
import os, time
import argparse
import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def arg_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', default='FetchPushWallObstacle-v4')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--policy', type=str, default='CustomSACPolicy')
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--action_noise', type=str, default='none')
    parser.add_argument('--num_timesteps', type=float, default=3e6)
    parser.add_argument('--log_path', default=None, type=str)
    parser.add_argument('--load_path', default=None, type=str)
    parser.add_argument('--play', action="store_true", default=False)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--random_ratio', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--reward_type', type=str, default='sparse')
    parser.add_argument('--n_object', type=int, default=2)
    parser.add_argument('--priority', action="store_true", default=False)
    parser.add_argument('--curriculum', action="store_true", default=False)
    parser.add_argument('--sequential', action="store_true", default=False)
    parser.add_argument('--sil', action="store_true", default=False)
    parser.add_argument('--sil_coef', type=float, default=1.0)
    parser.add_argument('--export_gif', action="store_true", default=False)
    args = parser.parse_args()
    return args


def main(args):
    log_dir = args.log_path if (args.log_path is not None) else "/tmp/stable_baselines_" + time.strftime('%Y-%m-%d-%H-%M-%S')
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(log_dir)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(log_dir, format_strs=[])

    set_global_seeds(args.seed)

    model_class = SAC_parallel

    n_workers = args.num_workers if not args.play else 1
    env_kwargs = get_env_kwargs(args.env, random_ratio=args.random_ratio, sequential=args.sequential,
                                reward_type=args.reward_type, n_object=args.n_object)
    def make_thunk(rank):
        return lambda: make_env(env_id=args.env, rank=rank, log_dir=log_dir, kwargs=env_kwargs)
    env = ParallelSubprocVecEnv([make_thunk(i) for i in range(n_workers)], reset_when_done=True)

    if os.path.exists(os.path.join(logger.get_dir(), 'eval.csv')):
        os.remove(os.path.join(logger.get_dir(), 'eval.csv'))
        print('Remove existing eval.csv')
    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs['random_ratio'] = 0.0
    eval_env = make_env(env_id=args.env, rank=0, kwargs=eval_env_kwargs)
    eval_env = FlattenDictWrapper(eval_env, ['observation', 'achieved_goal', 'desired_goal'])

    if not args.play:
        os.makedirs(log_dir, exist_ok=True)

    # Available strategies (cf paper): future, final, episode, random
    goal_selection_strategy = 'future'  # equivalent to GoalSelectionStrategy.FUTURE

    if not args.play:
        from stable_baselines.ddpg.noise import NormalActionNoise
        noise_type = args.action_noise.split('_')[0]
        if noise_type == 'none':
            parsed_action_noise = None
        elif noise_type == 'normal':
            sigma = float(args.action_noise.split('_')[1])
            parsed_action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape),
                                                    sigma=sigma * np.ones(env.action_space.shape))
        else:
            raise NotImplementedError
        train_kwargs = get_train_kwargs("sac", args, parsed_action_noise, eval_env)

        def callback(_locals, _globals):
            if _locals['step'] % int(1e3) == 0:
                if 'FetchStack' in args.env:
                    mean_eval_reward = stack_eval_model(eval_env, _locals["self"],
                                                        init_on_table=(args.env == 'FetchStack-v2'))
                elif 'MasspointPushDoubleObstacle-v2' in args.env:
                    mean_eval_reward = egonav_eval_model(eval_env, _locals["self"], env_kwargs["random_ratio"],
                                                         fixed_goal=np.array([4., 4., 0.15, 0., 0., 0., 1.]))
                    mean_eval_reward2 = egonav_eval_model(eval_env, _locals["self"], env_kwargs["random_ratio"],
                                                          goal_idx=0,
                                                          fixed_goal=np.array([4., 4., 0.15, 1., 0., 0., 0.]))
                    log_eval(_locals['self'].num_timesteps, mean_eval_reward2, file_name="eval_box.csv")
                else:
                    mean_eval_reward = eval_model(eval_env, _locals["self"])
                log_eval(_locals['self'].num_timesteps, mean_eval_reward)
            if _locals['step'] % int(2e4) == 0:
                model_path = os.path.join(log_dir, 'model_' + str(_locals['step'] // int(2e4)))
                model.save(model_path)
                print('model saved to', model_path)
            return True

        class CustomSACPolicy(SACPolicy):
            def __init__(self, *model_args, **model_kwargs):
                super(CustomSACPolicy, self).__init__(*model_args, **model_kwargs,
                                                      layers=[256, 256] if 'MasspointPushDoubleObstacle' in args.env else [256, 256, 256, 256],
                                                      feature_extraction="mlp")
        register_policy('CustomSACPolicy', CustomSACPolicy)
        from utils.sac_attention_policy import AttentionPolicy
        register_policy('AttentionPolicy', AttentionPolicy)
        policy_kwargs = get_policy_kwargs("sac", args)

        if rank == 0:
            print('train_kwargs', train_kwargs)
            print('policy_kwargs', policy_kwargs)
        # Wrap the model
        model = HER_HACK(args.policy, env, model_class, n_sampled_goal=4,
                         goal_selection_strategy=goal_selection_strategy,
                         num_workers=args.num_workers,
                         policy_kwargs=policy_kwargs,
                         verbose=1,
                         **train_kwargs)
        print(model.get_parameter_list())

        # Train the model
        model.learn(int(args.num_timesteps), seed=args.seed, callback=callback, log_interval=100 if not ('MasspointMaze-v3' in args.env) else 10)

        if rank == 0:
            model.save(os.path.join(log_dir, 'final'))

    # WARNING: you must pass an env
    # or wrap your environment with HERGoalEnvWrapper to use the predict method
    if args.play and rank == 0:
        assert args.load_path is not None
        model = HER_HACK.load(args.load_path, env=env)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        obs = env.reset()
        if 'FetchStack' in args.env:
            env.env_method('set_task_array', [[(env.get_attr('n_object')[0], 0)]])
            obs = env.reset()
            while env.get_attr('current_nobject')[0] != env.get_attr('n_object')[0] or env.get_attr('task_mode')[0] != 1:
                obs = env.reset()
        elif 'FetchPushWallObstacle' in args.env:
            while not (obs['observation'][0][4] > 0.7 and obs['observation'][0][4] < 0.8):
                obs = env.reset()
            env.env_method('set_goal', [np.array([1.18, 0.8, 0.425, 1, 0])])
            obs = env.env_method('get_obs')
            obs = {'observation': obs[0]['observation'][None],
                    'achieved_goal': obs[0]['achieved_goal'][None],
                    'desired_goal': obs[0]['desired_goal'][None]}
            # obs[0] = np.concatenate([obs[0][key] for key in ['observation', 'achieved_goal', 'desired_goal']])
        elif 'MasspointPushDoubleObstacle' in args.env or 'FetchPushWallObstacle' in args.env:
            while np.argmax(obs['desired_goal'][0][3:]) != 0:
                obs = env.reset()
        elif 'MasspointMaze-v2' in args.env:
            while obs['observation'][0][0] < 3 or obs['observation'][0][1] < 3:
                obs = env.reset()
            env.env_method('set_goal', [np.array([1., 1., 0.15])])
            obs = env.env_method('get_obs')
            obs = {'observation': obs[0]['observation'][None],
                    'achieved_goal': obs[0]['achieved_goal'][None],
                    'desired_goal': obs[0]['desired_goal'][None]}

        print('goal', obs['desired_goal'][0], 'obs', obs['observation'][0])
        episode_reward = 0.0
        images = []
        frame_idx = 0
        num_episode = 0
        for i in range(env_kwargs['max_episode_steps'] * 10):
            img = env.render(mode='rgb_array')
            ax.cla()
            ax.imshow(img)
            tasks = ['pick and place', 'stack']
            ax.set_title('episode ' + str(num_episode) + ', frame ' + str(frame_idx)
                         + ', task: ' + tasks[np.argmax(obs['observation'][0][-2:])])
            images.append(img)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            frame_idx += 1
            if args.export_gif:
                plt.imsave(os.path.join(os.path.dirname(args.load_path), 'tempimg%d.png' % i), img)
            else:
                plt.pause(0.02)
            if done:
                print('episode_reward', episode_reward)
                obs = env.reset()
                if 'FetchStack' in args.env:
                    while env.get_attr('current_nobject')[0] != env.get_attr('n_object')[0] or \
                                    env.get_attr('task_mode')[0] != 1:
                        obs = env.reset()
                elif 'MasspointPushDoubleObstacle' in args.env or 'FetchPushWallObstacle' in args.env:
                    while np.argmax(obs['desired_goal'][0][3:]) != 0:
                        obs = env.reset()
                print('goal', obs['desired_goal'][0])
                episode_reward = 0.0
                frame_idx = 0
                num_episode += 1
                if num_episode >= 1:
                    break
        exit()
        if args.export_gif:
            os.system('ffmpeg -r 5 -start_number 0 -i ' + os.path.dirname(args.load_path) + '/tempimg%d.png -c:v libx264 -pix_fmt yuv420p ' +
                      os.path.join(os.path.dirname(args.load_path), args.env + '.mp4'))
            for i in range(env_kwargs['max_episode_steps'] * 10):
                # images.append(plt.imread('tempimg' + str(i) + '.png'))
                try:
                    os.remove(os.path.join(os.path.dirname(args.load_path), 'tempimg' + str(i) + '.png'))
                except:
                    pass


if __name__ == '__main__':
    args = arg_parse()
    main(args)

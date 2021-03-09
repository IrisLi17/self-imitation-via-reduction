from baselines import PPO2
from stable_baselines.common.policies import register_policy
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from utils.log_utils import eval_model, log_eval, stack_eval_model

from utils.make_env_utils import make_env, configure_logger, get_env_kwargs, get_policy_kwargs, get_train_kwargs, \
    get_num_workers
import numpy as np

import os, time, argparse
import matplotlib.pyplot as plt


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
    parser.add_argument('--sequential', action="store_true", default=False)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--play', action="store_true", default=False)
    parser.add_argument('--export_video', action="store_true", default=False)
    args = parser.parse_args()
    return args


def main(args):
    log_dir = args.log_path if (args.log_path is not None) else \
        "/tmp/stable_baselines_" + time.strftime('%Y-%m-%d-%H-%M-%S')
    configure_logger(log_dir) 
    
    set_global_seeds(args.seed)

    n_cpu = get_num_workers(args.env) if not args.play else 1
    env_kwargs = get_env_kwargs(args.env, args.random_ratio, args.sequential, args.reward_type,
                                args.n_object, args.curriculum)
    def make_thunk(rank):
        return lambda: make_env(env_id=args.env, rank=rank, log_dir=log_dir, flatten_dict=True, kwargs=env_kwargs)
    env = SubprocVecEnv([make_thunk(i) for i in range(n_cpu)])

    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs['random_ratio'] = 0.0
    if "use_cu" in eval_env_kwargs:
        eval_env_kwargs['use_cu'] = False
    eval_env = make_env(env_id=args.env, rank=0, flatten_dict=True, kwargs=eval_env_kwargs)
    print(eval_env)
    if not args.play:
        os.makedirs(log_dir, exist_ok=True)
        train_kwargs = get_train_kwargs("ppo", args, parsed_action_noise=None, eval_env=eval_env)

        # policy = 'MlpPolicy'
        from utils.attention_policy import AttentionPolicy
        register_policy('AttentionPolicy', AttentionPolicy)
        policy_kwargs = get_policy_kwargs("ppo", args)
        print(policy_kwargs)

        model = PPO2(args.policy, env, verbose=1, nminibatches=32, lam=0.95, noptepochs=10,
                     ent_coef=0.01, learning_rate=3e-4, cliprange=0.2, policy_kwargs=policy_kwargs, **train_kwargs)
        print(model.get_parameter_list())

        def callback(_locals, _globals):
            num_update = _locals["update"]
            if 'FetchStack' in args.env:
                mean_eval_reward = stack_eval_model(eval_env, _locals["self"])
            else:
                mean_eval_reward = eval_model(eval_env, _locals["self"])
            log_eval(num_update, mean_eval_reward)
            if num_update % 10 == 0:
                model_path = os.path.join(log_dir, 'model_' + str(num_update // 10))
                model.save(model_path)
                print('model saved to', model_path)
            return True

        model.learn(total_timesteps=int(args.num_timesteps), callback=callback, seed=args.seed, log_interval=1)
        model.save(os.path.join(log_dir, 'final'))
    
    else:
        assert args.load_path is not None
        model = PPO2.load(args.load_path)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        obs = env.reset()
        goal_dim = env.get_attr('goal')[0].shape[0]
        if 'FetchStack' in args.env:
            while env.get_attr('current_nobject')[0] != env.get_attr('n_object')[0] or \
                    env.get_attr('task_mode')[0] != 1:
                obs = env.reset()
        elif 'FetchPush' in args.env:
            while not (1.25 < obs[0][6] < 1.33 and obs[0][7] < 0.61 and 0.7 < obs[0][4] < 0.8):
                obs = env.reset()
            env.env_method('set_goal', np.array([1.2, 0.75, 0.425, 1, 0]))
            obs = env.env_method('get_obs')
            obs[0] = np.concatenate([obs[0][key] for key in ['observation', 'achieved_goal', 'desired_goal']])
        else:
            while np.argmax(obs[0][-goal_dim+3:]) != 0:
                obs = env.reset()
        print('achieved_goal', obs[0][-2*goal_dim: -goal_dim], 'goal', obs[0][-goal_dim:])
        episode_reward = 0.0
        num_episode = 0
        frame_idx = 0
        images = []
        if 'max_episode_steps' not in env_kwargs.keys():
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
                if 'FetchStack' in args.env:
                    tasks = ['pick and place', 'stack']
                    ax.set_title('episode ' + str(num_episode) + ', frame ' + str(frame_idx)
                                 + ', task: ' + tasks[np.argmax(obs[0][-2*goal_dim-2:-2*goal_dim])])
            images.append(img)
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            frame_idx += 1
            if not args.export_video:
                plt.pause(0.1)
            else:
                plt.imsave(os.path.join(os.path.dirname(args.load_path), 'tempimg%d.png' % i), img)
            if done:
                print('episode_reward', episode_reward)
                if 'FetchStack' in args.env:
                    while env.get_attr('current_nobject')[0] != env.get_attr('n_object')[0] or \
                            env.get_attr('task_mode')[0] != 1:
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
        if args.export_video:
            os.system('ffmpeg -r 5 -start_number 0 -i ' + os.path.dirname(args.load_path) +
                      '/tempimg%d.png -c:v libx264 -pix_fmt yuv420p ' +
                      os.path.join(os.path.dirname(args.load_path), args.env + '.mp4'))
            for i in range(env_kwargs['max_episode_steps'] * 10):
                try:
                    os.remove(os.path.join(os.path.dirname(args.load_path), 'tempimg' + str(i) + '.png'))
                except:
                    pass


if __name__ == '__main__':
    args = arg_parse()
    main(args)

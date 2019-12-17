from baselines import SAC_sanity
from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy
from stable_baselines.common.policies import register_policy
from baselines import HER_HACK
from test_ensemble import make_env
import gym
import matplotlib.pyplot as plt
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines import logger
import os, time, pickle, pandas
import imageio
import argparse
import numpy as np
from stable_baselines.her.utils import KEY_ORDER

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

hard_test = False


def arg_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', default='FetchPushWallObstacle-v4')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--policy', type=str, default='MlpPolicy')
    parser.add_argument('--goal_selection_strategy', type=str, default='future')
    parser.add_argument('--n_sampled_goal', type=int, default=4)
    parser.add_argument('--num_timesteps', type=float, default=3e6)
    parser.add_argument('--trained_model', default=None, type=str)
    parser.add_argument('--n_subgoal', type=int, default=4)
    # parser.add_argument('--augment_when_success', action="store_true", default=False)
    # parser.add_argument('--hack_augment_time', action="store_true", default=False)
    # parser.add_argument('--hack_augment_policy', action="store_true", default=False)
    # parser.add_argument('--double_buffer', action="store_true", default=False)
    parser.add_argument('--log_path', default=None, type=str)
    parser.add_argument('--load_path', default=None, type=str)
    parser.add_argument('--play', action="store_true", default=False)
    parser.add_argument('--buffer_size', type=float, default=1e6)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--random_ratio', type=float, default=1.0)
    parser.add_argument('--heavy_obstacle', action="store_true", default=False)
    parser.add_argument('--random_gripper', action="store_true", default=False)
    parser.add_argument('--export_gif', action="store_true", default=False)
    parser.add_argument('--pretrain', action="store_true", default=False)
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def main(seed, policy, num_timesteps, batch_size, log_path, load_path, play, heavy_obstacle, random_gripper,
         export_gif, **args):
    log_dir = log_path if (log_path is not None) else "/tmp/stable_baselines_" + time.strftime('%Y-%m-%d-%H-%M-%S')
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(log_dir)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(log_dir, format_strs=[])

    set_global_seeds(seed)

    model_class = SAC_sanity  # works also with SAC, DDPG and TD3

    # if env_name in ENTRY_POINT.keys():
    #     kwargs = dict(penaltize_height=False, heavy_obstacle=heavy_obstacle, random_gripper=random_gripper)
    #     print(kwargs)
    #     max_episode_steps = 100 if env_name == 'FetchPushWallObstacle-v4' else 50
    #     gym.register(env_name, entry_point=ENTRY_POINT[env_name], max_episode_steps=max_episode_steps, kwargs=kwargs)
    #     env = gym.make(env_name)
    # else:
    #     raise NotImplementedError("%s not implemented" % env_name)
    env_name = args['env']
    env_kwargs = dict(random_box=True,
                      heavy_obstacle=True,
                      random_ratio=args['random_ratio'],
                      random_gripper=True,
                      max_episode_steps=100, )
    env = make_env(env_name, **env_kwargs)

    if not play:
        os.makedirs(log_dir, exist_ok=True)

    # Available strategies (cf paper): future, final, episode, random
    # I add another strategy: future_and_final
    goal_selection_strategy = args['goal_selection_strategy']  # equivalent to GoalSelectionStrategy.FUTURE

    if not play:
        if model_class is SAC_sanity:
            # wrap env
            from utils.wrapper import DoneOnSuccessWrapper
            env = DoneOnSuccessWrapper(env)
            # TODO: make monitor work with our hacky step
            # env = Monitor(env, os.path.join(log_dir, str(rank) + ".monitor.csv"), allow_early_resets=True)
            sac_model = HER_HACK.load(args['trained_model']).model
            train_kwargs = dict(buffer_size=int(args['buffer_size']),
                                ent_coef="auto",
                                gamma=0.95,
                                learning_starts=1000,
                                train_freq=1,
                                batch_size=batch_size,
                                trained_sac_model=sac_model,
                                # n_subgoal=args['n_subgoal'],
                                # augment_when_success=args['augment_when_success'],
                                # hack_augment_time=args['hack_augment_time'],
                                # hack_augment_policy=args['hack_augment_policy'],
                                # double_buffer=args['double_buffer'],
                                )
            policy_kwargs = {}

            def callback(_locals, _globals):
                if _locals['step'] % int(1e5) == 0:
                    model_path = os.path.join(log_dir, 'model_' + str(_locals['step'] // int(1e5)))
                    model.save(model_path)
                return True
        else:
            train_kwargs = {}
            policy_kwargs = {}
            callback = None
        class CustomSACPolicy(SACPolicy):
            def __init__(self, *args, **kwargs):
                super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                                    layers=[256, 256],
                                                    feature_extraction="mlp")
        register_policy('CustomSACPolicy', CustomSACPolicy)
        # if layer_norm:
        #     policy = 'LnMlpPolicy'
        # else:
        #     policy = 'MlpPolicy'
        if rank == 0:
            print('train_kwargs', train_kwargs)
            print('policy_kwargs', policy_kwargs)
        # Wrap the model
        model = HER_HACK(policy, env, model_class, n_sampled_goal=args['n_sampled_goal'],
                         goal_selection_strategy=goal_selection_strategy,
                         policy_kwargs=policy_kwargs,
                         verbose=1,
                         **train_kwargs)
        # TODO: this is a hack
        if load_path is not None:
            print('Start training from', load_path)
            model.model.load_parameters(load_path)

        # Pretrain with expert data.
        if args['pretrain'] and os.path.exists('./logs/sanity_data/demo1.npz'):
            from stable_baselines.gail import ExpertDataset
            dataset = ExpertDataset(expert_path='./logs/sanity_data/demo1.npz', train_fraction=0.9)
            model.model.pretrain(dataset, n_epochs=3000, learning_rate=1e-4)

        # Train the model
        model.learn(int(num_timesteps), seed=seed, callback=callback, log_interval=20)

        if rank == 0:
            model.save(os.path.join(log_dir, 'final'))

    # WARNING: you must pass an env
    # or wrap your environment with HERGoalEnvWrapper to use the predict method
    if play and rank == 0:
        assert load_path is not None
        model = HER_HACK.load(load_path, env=env)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        with open(os.path.join(os.path.dirname(load_path), 'start_states.pkl'), 'rb') as f:
            start_states = pickle.load(f)
        if os.path.exists(os.path.join(os.path.dirname(load_path), 'goals.pkl')):
            with open(os.path.join(os.path.dirname(load_path), 'goals.pkl'), 'rb') as f:
                parsed_goals = pickle.load(f)
        else:
            # Parse csv
            def get_item(log_file, label):
                data = pandas.read_csv(log_file, index_col=None, comment='#')
                return data[label].values
            fname = os.path.join(os.path.dirname(load_path), 'success_traj.csv')
            goals = []
            for i in range(5):
                goals.append(get_item(fname, 'goal_' + str(i)))
            goals = np.asarray(goals)
            goals = np.swapaxes(goals, 0, 1)
            dones = get_item(fname, 'done')
            end_points = np.where(dones > 0.5)[0]
            parsed_goals = [goals[i] for i in end_points]
        print(parsed_goals)
        # demo_data = np.load('./logs/sanity_data/demo1.npz')
        # obs = env.reset()
        # while not (obs['desired_goal'][0] < env.pos_wall[0] < obs['achieved_goal'][0] or
        #             obs['desired_goal'][0] > env.pos_wall[0] > obs['achieved_goal'][0]):
        #     if not hard_test:
        #         break
        #     obs = env.reset()
        # print('gripper_pos', obs['observation'][0:3])
        # img = env.render(mode='rgb_array')
        env.reset()
        # print(env)
        print(env.unwrapped.goal)
        episode_reward = 0.0
        images = []
        states = []
        values = []
        frame_idx = 0
        episode_idx = 0
        env.unwrapped.sim.set_state(start_states[0])
        env.unwrapped.goal = parsed_goals[0]
        print(env.unwrapped.goal, env.goal)
        env.unwrapped.sim.forward()
        # Set goal.
        obs = env.get_obs()
        print(obs, env.goal)
        for i in range(env.spec.max_episode_steps * 5):
            # images.append(img)
            states.append(env.unwrapped.sim.get_state())
            feed_dict = {model.model.observations_ph: np.expand_dims(np.concatenate([obs[key] for key in KEY_ORDER]), axis=0)}
            value = model.model.sess.run(model.model.step_ops[6], feed_dict)
            values.append(np.squeeze(value))
            action, _ = model.predict(obs)
            # print('step', i, 'obs dist', np.mean((demo_data['obs'][i][:9] - obs['observation'][:9])**2),
            #       'act dist', np.mean((demo_data['actions'][i] - action) ** 2))
            # print('obstacle euler', obs['observation'][20:23])
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            frame_idx += 1
            ax[0].cla()
            img = env.render(mode='rgb_array')
            ax[0].imshow(img)
            ax[0].set_title('episode ' + str(episode_idx) + ', frame ' + str(frame_idx) +
                         ', goal idx ' + str(np.argmax(obs['desired_goal'][3:])))
            ax[1].cla()
            ax[1].plot(values, 'tab:blue')
            ax[1].set_xlim(0, 100)
            ax[1].set_ylim(0, 10)
            if export_gif:
                plt.savefig(os.path.join(os.path.dirname(load_path), 'tempimg' + str(i) + '.png'))
            else:
                plt.pause(0.02)
            if done and episode_idx < 4:
                values = []
                env.reset()
                env.unwrapped.sim.set_state(start_states[episode_idx + 1])
                env.unwrapped.goal = parsed_goals[episode_idx + 1]
                env.unwrapped.sim.forward()
                obs = env.get_obs()
                # obs = env.reset()
                # while not (obs['desired_goal'][0] < env.pos_wall[0] < obs['achieved_goal'][0] or
                #             obs['desired_goal'][0] > env.pos_wall[0] > obs['achieved_goal'][0]):
                #     if not hard_test:
                #         break
                #     obs = env.reset()
                print('gripper_pos', obs['observation'][0:3])
                print('episode_reward', episode_reward)
                episode_reward = 0.0
                frame_idx = 0
                episode_idx += 1
        with open(os.path.join(os.path.dirname(load_path), 'states.pkl'), 'wb') as f:
            pickle.dump(states, f)
        if export_gif:
            os.system('ffmpeg -r 5 -start_number 0 -i ' + os.path.dirname(load_path) + '/tempimg%d.png -c:v libx264 -pix_fmt yuv420p ' +
                      os.path.join(os.path.dirname(load_path), env_name + '.mp4'))
            for i in range(env.spec.max_episode_steps * 5):
                # images.append(plt.imread('tempimg' + str(i) + '.png'))
                os.remove(os.path.join(os.path.dirname(load_path), 'tempimg' + str(i) + '.png'))
            # imageio.mimsave(env_name + '.gif', images)


if __name__ == '__main__':
    args = arg_parse()
    main(**args)

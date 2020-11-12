import sys, os
import numpy as np
from run_her import make_env, get_env_kwargs
from baselines import HER_HACK
from gym.wrappers import FlattenDictWrapper
from utils.parallel_subproc_vec_env2 import ParallelSubprocVecEnv as ParallelSubprocVecEnv2
import matplotlib.pyplot as plt
from baselines import SAC_augment


if __name__ == '__main__':
    env_id = sys.argv[1]
    model_path = sys.argv[2]
    env_kwargs = get_env_kwargs(env_id, random_ratio=0.0)

    def make_thunk(rank):
        return lambda: make_env(env_id=env_id, seed=0, rank=rank, kwargs=env_kwargs)

    env = ParallelSubprocVecEnv2([make_thunk(i) for i in range(1)])

    aug_env_id = env_id.split('-')[0] + 'Unlimit-' + env_id.split('-')[1]
    aug_env_kwargs = env_kwargs.copy()
    aug_env_kwargs['max_episode_steps'] = None

    # def make_thunk_aug(rank):
    #     return lambda: FlattenDictWrapper(make_env(env_id=aug_env_id, seed=0, rank=rank, kwargs=aug_env_kwargs),
    #                                       ['observation', 'achieved_goal', 'desired_goal'])
    #
    # aug_env = ParallelSubprocVecEnv([make_thunk_aug(i) for i in range(1)])
    aug_env = make_env(aug_env_id, seed=0, rank=0, kwargs=aug_env_kwargs)
    aug_env = FlattenDictWrapper(aug_env, ['observation', 'achieved_goal', 'desired_goal'])

    # goal_dim = aug_env.get_attr('goal')[0].shape[0]
    # obs_dim = aug_env.observation_space.shape[0] - 2 * goal_dim
    # noise_mag = aug_env.get_attr('size_obstacle')[0][1]
    # n_object = aug_env.get_attr('n_object')[0]

    goal_dim = aug_env.goal.shape[0]
    obs_dim = aug_env.observation_space.shape[0] - 2 * goal_dim
    noise_mag = aug_env.size_obstacle[1]
    n_object = aug_env.n_object
    model = HER_HACK.load(model_path, env=env)
    model.model.env_id = env_id
    model.model.goal_dim = goal_dim
    model.model.obs_dim = obs_dim
    model.model.noise_mag = noise_mag
    model.model.n_object = n_object

    obs_buf, state_buf, tower_height_buf = [], [], []
    transition_buf = []
    img_buf = []
    obs = env.reset()
    while np.argmax(obs['desired_goal'][0][3:]) != 0:
        obs = env.reset()
    done = False
    while not done:
        # tower_height = env.tower_height
        state = env.env_method('get_state')[0]
        obs_buf.append(np.concatenate([obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']], axis=-1)[0])
        state_buf.append(state)
        # tower_height_buf.append(tower_height)
        img = env.env_method('render', ['rgb_array'])[0]
        img_buf.append(img)
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        transition_buf.append((obs_buf[-1], action[0]))  # dummy items after obs
    if info[0]['is_success']:
        print('No need to do reduction')
    else:
        restart_step, subgoal = SAC_augment.select_subgoal(model.model, transition_buf, 1, tower_height_buf if 'FetchStack' in env_id else None)
        # restart_step, subgoal = model.model.select_subgoal(transition_buf, 1, tower_height_buf if 'FetchStack' in env_id else None)
        print(restart_step, subgoal)
        img_buf = img_buf[:restart_step[0]]
        aug_env.reset()
        aug_env.set_state(state_buf[restart_step[0]])
        # aug_env.set_task_mode(0)
        aug_env.set_goal(subgoal[0])
        step_so_far = restart_step[0]
        done = False
        obs = aug_env.get_obs()
        obs = np.concatenate([obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']])
        fig, ax = plt.subplots(1, 1)
        while not done:
            if step_so_far > env_kwargs['max_episode_steps']:
                break
            action, _ = model.predict(obs)
            obs, reward, done, info = aug_env.step(action)
            img = aug_env.render(mode='rgb_array')
            ax.cla()
            ax.set_title('frame %d' % step_so_far)
            ax.imshow(img)
            plt.pause(0.2)
            img_buf.append(img)
            step_so_far += 1
        # aug_env.set_task_mode(1)
        aug_env.set_goal(obs_buf[0][-goal_dim:])
        print('Switch to ultimate goal', obs_buf[0][-goal_dim:])
        done = False
        while not done:
            if step_so_far > env_kwargs['max_episode_steps']:
                break
            action, _ = model.predict(obs)
            obs, reward, done, info = aug_env.step(action)
            img = aug_env.render(mode='rgb_array')
            ax.cla()
            ax.set_title('frame %d' % step_so_far)
            ax.imshow(img)
            plt.pause(0.2)
            img_buf.append(img)
            step_so_far += 1
    for i in range(len(img_buf)):
        plt.imsave(os.path.join(os.path.dirname(model_path), 'tempimg%d.png' % i), img_buf[i])
    os.system('ffmpeg -r 5 -start_number 0 -i ' + os.path.dirname(
        model_path) + '/tempimg%d.png -c:v libx264 -pix_fmt yuv420p ' +
              os.path.join(os.path.dirname(model_path), 'test_time.mp4'))
    for i in range(len(img_buf)):
        os.remove(os.path.join(os.path.dirname(model_path), 'tempimg%d.png' % i))

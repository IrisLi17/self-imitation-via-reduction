import sys, os
import numpy as np
from run_her import make_env
from baselines import HER_HACK
from gym.wrappers import FlattenDictWrapper
import matplotlib.pyplot as plt


def select_subgoal(obs_buf, k, tower_height):
    # self.ep_transition_buf, self.model.value
    obs_buf = np.asarray(obs_buf)
    if tower_height[-1] + 0.05 - obs_buf[-1][obs_dim + goal_dim + 2] > 0.01:
        # Tower height is equal to (or higher than) goal height.
        # print('towerheight exceed goalheight')
        return np.array([]), np.array([])
    sample_t = np.random.randint(0, len(obs_buf), 4096)
    sample_obs = obs_buf[sample_t]
    ultimate_idx = np.argmax(sample_obs[0][obs_dim + goal_dim + 3:])
    noise = np.random.uniform(low=-noise_mag, high=noise_mag, size=(len(sample_t), 2))
    # TODO: if there are more than one obstacle
    sample_obs_buf = []
    subgoal_obs_buf = []
    sample_height = np.array(tower_height)[sample_t]
    for object_idx in range(0, n_object):
        if abs(sample_height[0] + 0.05 - sample_obs[0][obs_dim + goal_dim + 2]) > 0.01 \
                and object_idx == np.argmax(sample_obs[0][obs_dim + goal_dim + 3:]):
            # If the goal is not 1 floor above towerheight, we don't perturb self position
            continue
        if np.linalg.norm(sample_obs[0][3 + object_idx * 3: 3 + (object_idx + 1) * 3]) < 1e-3:
            # This object is masked
            continue
        if np.linalg.norm(sample_obs[0][3 + object_idx * 3: 3 + object_idx * 3 + 2] -
                          sample_obs[0][
                          obs_dim + goal_dim: obs_dim + goal_dim + 2]) < 1e-3:
            # This object is part of tower
            continue
        obstacle_xy = sample_obs[:, 3 * (object_idx + 1):3 * (object_idx + 1) + 2] + noise
        # Find how many objects have been stacked
        obstacle_height = np.expand_dims(sample_height + 0.05, axis=1)
        # obstacle_height = max(sample_obs[0][self.obs_dim + self.goal_dim + 2] - 0.05, 0.425) * np.ones((len(sample_t), 1))
        obstacle_xy = np.concatenate([obstacle_xy, obstacle_height], axis=-1)
        sample_obs[:, 3 * (object_idx + 1):3 * (object_idx + 1) + 3] = obstacle_xy
        sample_obs[:, 3 * (object_idx + 1 + n_object):3 * (object_idx + 1 + n_object) + 3] \
            = sample_obs[:, 3 * (object_idx + 1):3 * (object_idx + 1) + 3] - sample_obs[:, 0:3]
        sample_obs[:, obs_dim: obs_dim + 3] = sample_obs[:,
                                                       3 * (ultimate_idx + 1):3 * (ultimate_idx + 1) + 3]
        sample_obs_buf.append(sample_obs.copy())

        subgoal_obs = obs_buf[sample_t]
        # if debug:
        #     subgoal_obs = np.tile(subgoal_obs, (2, 1))
        subgoal_obs[:, obs_dim - 2: obs_dim] = np.array([1, 0])  # Pick and place
        subgoal_obs[:, obs_dim:obs_dim + 3] = subgoal_obs[:,
                                                        3 * (object_idx + 1):3 * (object_idx + 1) + 3]
        one_hot = np.zeros(n_object)
        one_hot[object_idx] = 1
        subgoal_obs[:, obs_dim + 3: obs_dim + goal_dim] = one_hot
        subgoal_obs[:, obs_dim + goal_dim: obs_dim + goal_dim + 3] = obstacle_xy
        # subgoal_obs[:, self.obs_dim + self.goal_dim + 2:self.obs_dim + self.goal_dim + 3] = subgoal_obs[:, 3 * (
        # object_idx + 1) + 2:3 * (object_idx + 1) + 3]
        subgoal_obs[:, obs_dim + goal_dim + 3: obs_dim + goal_dim * 2] = one_hot
        subgoal_obs_buf.append(subgoal_obs)
    # print(len(sample_obs_buf))
    if len(sample_obs_buf) == 0:
        return np.array([]), np.array([])
    sample_obs_buf = np.concatenate(sample_obs_buf, axis=0)
    # value2 = self.model.value(sample_obs_buf)
    subgoal_obs_buf = np.concatenate(subgoal_obs_buf)
    # value1 = self.model.value(subgoal_obs_buf)

    # _values = self.model.value(np.concatenate([sample_obs_buf, subgoal_obs_buf], axis=0))
    feed_dict = {model.model.observations_ph: np.concatenate([sample_obs_buf, subgoal_obs_buf], axis=0)}
    _values = np.squeeze(model.model.sess.run(model.model.step_ops[6], feed_dict), axis=-1)
    value2 = _values[:sample_obs_buf.shape[0]]
    value1 = _values[sample_obs_buf.shape[0]:]
    normalize_value1 = (value1 - np.min(value1)) / (np.max(value1) - np.min(value1))
    normalize_value2 = (value2 - np.min(value2)) / (np.max(value2) - np.min(value2))
    # best_idx = np.argmax(normalize_value1 * normalize_value2)
    ind = np.argsort(normalize_value1 * normalize_value2)
    good_ind = ind[-k:]
    # if debug:
    #     print('original value1', 'mean', np.mean(origin_value1), 'std', np.std(origin_value1))
    #     print('original value2', 'mean', np.mean(origin_value2), 'std', np.std(origin_value2))
    #     print(value1[good_ind])
    #     print(value2[good_ind])
    # restart_step = sample_t[best_idx]
    # subgoal = subgoal_obs[best_idx, 45:50]
    mean_values = (value1[good_ind] + value2[good_ind]) / 2
    assert mean_values.shape[0] == k
    # for i in range(k):
    #     self.mean_value_buf.append(mean_values[i])
    # filtered_idx = np.where(mean_values >= np.mean(self.mean_value_buf))[0]
    # good_ind = good_ind[filtered_idx]

    restart_step = sample_t[good_ind % len(sample_t)]
    subgoal = subgoal_obs_buf[good_ind, obs_dim + goal_dim: obs_dim + goal_dim * 2]

    # print('subgoal', subgoal, 'with value1', normalize_value1[best_idx], 'value2', normalize_value2[best_idx])
    # print('restart step', restart_step)
    return restart_step, subgoal


if __name__ == '__main__':
    model_path = sys.argv[1]
    n_object = 3
    env_kwargs = dict(random_box=True,
                      random_ratio=0.0,
                      random_gripper=True,
                      max_episode_steps=(50 * n_object if n_object > 3 else 100),
                      # max_episode_steps=100,
                      reward_type='sparse',
                      n_object=n_object,
                      )
    env = make_env(env_id='FetchStack-v1', seed=0, rank=0, kwargs=env_kwargs)
    env = FlattenDictWrapper(env, ['observation', 'achieved_goal', 'desired_goal'])
    aug_env_kwargs = env_kwargs.copy()
    aug_env_kwargs['max_episode_steps'] = None
    aug_env = make_env(env_id='FetchStackUnlimit-v1', seed=0, rank=0, kwargs=aug_env_kwargs)
    aug_env = FlattenDictWrapper(aug_env, ['observation', 'achieved_goal', 'desired_goal'])
    goal_dim = env.goal.shape[0]
    obs_dim = env.observation_space.shape[0] - 2 * goal_dim
    noise_mag = env.size_obstacle[0]
    model = HER_HACK.load(model_path)

    obs_buf, state_buf, tower_height_buf = [], [], []
    img_buf = []
    value_buf = []
    env.unwrapped.set_task_array([(n_object, 1)])
    obs = env.reset()
    done = False
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    while not done:
        value, attention_weight = model.model.sess.run([model.model.step_ops[6], model.model.policy_tf.attention_weight],
                                                       {model.model.observations_ph: np.expand_dims(obs, axis=0)})
        value_buf.append(np.squeeze(value))
        print(np.squeeze(attention_weight))
        # tower_height = env.tower_height
        # state = env.get_state()
        obs_buf.append(obs)
        # state_buf.append(state)
        # tower_height_buf.append(tower_height)
        img = env.render(mode='rgb_array')
        img_buf.append(img)
        ax[0].cla()
        ax[0].imshow(img)
        ax[0].set_title('value ' + str(value))
        ax[1].cla()
        ax[1].plot(value_buf)
        plt.pause(0.1)
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
    if info['is_success']:
        print('No need to do reduction')
        print('values', value_buf)
    else:
        raise AssertionError
        restart_step, subgoal = select_subgoal(obs_buf, 1, tower_height_buf)
        print(restart_step, subgoal)
        img_buf = img_buf[:restart_step[0]]
        aug_env.reset()
        aug_env.set_state(state_buf[restart_step[0]])
        aug_env.set_task_mode(0)
        aug_env.set_goal(subgoal[0])
        step_so_far = restart_step
        done = False
        obs = aug_env.get_obs()
        obs = np.concatenate([obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']])
        fig, ax = plt.subplots(1, 1)
        while not done:
            if step_so_far > 100:
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
        aug_env.set_task_mode(1)
        aug_env.set_goal(obs_buf[0][-goal_dim:])
        print('Switch to ultimate goal', obs_buf[0][-goal_dim:])
        done = False
        while not done:
            if step_so_far > 100:
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

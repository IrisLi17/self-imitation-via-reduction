import seaborn as sns
import pandas
import numpy as np
import sys, os
from scipy import interpolate
import matplotlib.pyplot as plt


def get_item(log_file, label):
    data = pandas.read_csv(log_file, index_col=None, comment='#', error_bad_lines=True)
    return data[label].values


def smooth(array, window):
    out = np.zeros(array.shape[0] - window)
    for i in range(out.shape[0]):
        out[i] = np.mean(array[i:i + window])
    return out


if __name__ == '__main__':
    folder_name = sys.argv[1]
    env_name = sys.argv[2]
    assert env_name in ['push', 'particle', 'maze', 'stacking']
    # assert mode in ['train', 'hard', 'iteration']
    max_timesteps = {'push': 4.99e7,
                     'particle': 2.8e8,
                     'maze': 1.5e6,
                     'stacking': 2e8,}
    max_iterationss = {'push': 750,
                       'particle': 510,
                       'maze': 245,}
    # df_timesteps, df_sr, df_eval, df_legend, df_iteration, df_eval_iteration, df_legend_iteration = [], [], [], [], [], [], []
    df_iteration, df_len_mean, df_legend_iteration = [], [], []
    subfolders = ['ppo', 'sir_re8']
    if 'particle_random0.7' in folder_name:
        subfolders = ['ppo', 'sir_re1-8']
    elif 'particle_random1.0' in folder_name:
        subfolders = ['ppo', 'sir_re1-8']
    elif 'maze' in folder_name:
        subfolders = ['ppo', 'sir_re2']
    for subfolder in subfolders:
        for i in range(3):
            if not os.path.exists(os.path.join(folder_name, subfolder, str(i), 'progress.csv')):
                continue
            progress_file = os.path.join(folder_name, subfolder, str(i), 'progress.csv')
            # eval_file = os.path.join(folder_name, subfolder, str(i), 'eval.csv')
            # raw_success_rate = get_item(progress_file, 'ep_reward_mean')
            # raw_total_timesteps = get_item(progress_file, 'total_timesteps')
            raw_len_mean = get_item(progress_file, 'ep_len_mean')
            raw_iterations = get_item(progress_file, 'n_updates')
            # raw_eval_reward = get_item(eval_file, 'mean_eval_reward')
            # sr_f = interpolate.interp1d(raw_total_timesteps, raw_success_rate, fill_value="extrapolate")
            # eval_f = interpolate.interp1d(raw_total_timesteps, raw_eval_reward, fill_value="extrapolate")
            # timesteps = np.arange(0, max_timesteps[env_name], max_timesteps[env_name] // 500)
            # print(timesteps[0], timesteps[-1], raw_total_timesteps[0], raw_total_timesteps[-1])
            # success_rate = sr_f(timesteps)
            # eval_reward = eval_f(timesteps)
            L = max_iterationss[env_name]
            iterations = smooth(raw_iterations[:L], 20)
            len_mean = smooth(raw_len_mean[:L], 20)
            # timesteps = smooth(timesteps, 20)
            # success_rate = smooth(success_rate, 20)
            # eval_reward = smooth(eval_reward, 20)
            df_iteration.append(iterations)
            df_len_mean.append(len_mean)
            # df_timesteps.append(timesteps)
            # df_sr.append(success_rate)
            # df_eval.append(eval_reward)
            # df_legend.append(np.array([subfolder.upper()] * len(timesteps)))


            # eval_iteration = smooth(raw_eval_reward[:L], 20)
            # df_iteration.append(iterations)
            # df_eval_iteration.append(eval_iteration)
            df_legend_iteration.append(np.array([subfolder.upper()] * len(iterations)))
    # df_timesteps = np.concatenate(df_timesteps, axis=0).tolist()
    # df_sr = np.concatenate(df_sr, axis=0).tolist()
    # df_eval = np.concatenate(df_eval, axis=0).tolist()
    # df_legend = np.concatenate(df_legend, axis=0).tolist()
    df_iteration = np.concatenate(df_iteration, axis=0).tolist()
    df_len_mean = np.concatenate(df_len_mean, axis=0).tolist()
    # df_eval_iteration = np.concatenate(df_eval_iteration, axis=0).tolist()
    df_legend_iteration = np.concatenate(df_legend_iteration, axis=0).tolist()
    # data = {'samples': df_timesteps, 'success_rate': df_sr, 'algo': df_legend}
    # sr_timesteps = pandas.DataFrame(data)
    # data = {'samples': df_timesteps, 'eval': df_eval, 'algo': df_legend}
    # eval_timesteps = pandas.DataFrame(data)
    data = {'iterations': df_iteration, 'len_mean': df_len_mean, 'algo': df_legend_iteration}
    len_mean_iteration = pandas.DataFrame(data)

    wspace = .3
    bottom = .3
    margin = .1
    left = .18
    width = 1.2 / ((1. - left) / (2. + wspace + margin / 2))
    height = 1.5 / ((1. - bottom) / (1 + margin / 2))

    plt.style.use("ggplot")
    # plt.rcParams.update({'legend.fontsize': 14})
    p = sns.color_palette()
    sns.set_palette([p[0], p[1]])
    f, axes = plt.subplots(1, 1, figsize=(width, height))
    # sns.lineplot(x='samples', y='success_rate', hue='algo', ax=axes[0], data=sr_timesteps)
    # axes[0].set_xlabel('samples')
    # axes[0].set_ylabel('success_rate')
    # axes[0].get_legend().remove()
    # sns.lineplot(x='samples', y='eval', hue='algo', ax=axes[1], data=eval_timesteps)
    # axes[1].set_xlabel('samples')
    # axes[1].set_ylabel('')
    # axes[1].get_legend().remove()
    sns.lineplot(x='iterations', y='len_mean', hue='algo', ax=axes, data=len_mean_iteration)
    axes.set_xlabel('iterations')
    axes.set_ylabel('episode length')
    axes.get_legend().remove()
    handles, labels = axes.get_legend_handles_labels()
    # if mode == 'train':
    #     sns.lineplot(x='samples', y='success_rate', hue='algo', data=sr_timesteps)
    #     axes.set_xlabel('samples')
    # elif mode == 'hard':
    #     sns.lineplot(x='samples', y='eval', hue='algo', data=eval_timesteps)
    #     axes.set_xlabel('samples')
    # elif mode == 'iteration':
    #     sns.lineplot(x='iterations', y='eval', hue='algo', ax=axes, data=eval_iteration)
    #     axes.set_xlabel('iterations')
    # axes.set_ylabel('success rate')
    # axes.get_legend().remove()
    # handles, labels = axes.get_legend_handles_labels()
    f.legend(handles[1:], ['PPO', 'SIR'], loc="upper right", ncol=1, title='')
    f.subplots_adjust(top=1. - margin / height, bottom=0.2, wspace=wspace, left=left, right=1. - margin / width)
    plt.savefig(os.path.join(folder_name, '../', os.path.basename(folder_name) + '_len.pdf'))
    # plt.show()

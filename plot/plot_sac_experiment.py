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
    assert env_name in ['push', 'stack2', 'stack3']
    # assert mode in ['train', 'hard', 'iteration']
    max_timesteps = {'push': 1.45e7,
                     'stack2': 2.8e7,
                     'stack3': 1e8,
                     }
    max_iterationss = {'push': 440000,
                       'stack2': 8.9e5,
                       'stack3': 2.5e6,
                       }
    df_timesteps, df_sr, df_eval, df_legend, df_iteration, df_eval_iteration, df_legend_iteration = [], [], [], [], [], [], []
    subfolders = ['sac', 'sir', 'sil']
    if 'particle_random0.7' in folder_name:
        subfolders = ['ppo', 'sir_re1-8']
    elif 'particle_random1.0' in folder_name:
        subfolders = ['ppo', 'sir_re1-8']
    elif 'push_random0.7' in folder_name:
        subfolders = ['sac', 'sir', 'sil', 'ds2']
    elif 'stack_2obj' in folder_name or 'stack_3obj' in folder_name:
        subfolders = ['sac', 'sir', 'sil', 'ds']
    for subfolder in subfolders:
        last_sr = []
        last_eval = []
        for i in range(3):
            if not os.path.exists(os.path.join(folder_name, subfolder, str(i), 'progress.csv')):
                continue
            progress_file = os.path.join(folder_name, subfolder, str(i), 'progress.csv')
            eval_file = os.path.join(folder_name, subfolder, str(i), 'eval.csv')
            if subfolder is 'ds' or subfolder is 'ds2':
                raw_success_rate = get_item(progress_file, 'success rate')
            else:
                raw_success_rate = get_item(progress_file, 'ep_rewmean')
            raw_total_timesteps = get_item(progress_file, 'total timesteps')
            try:
                raw_original_timesteps = get_item(progress_file, 'original_timesteps')
            except KeyError:
                raw_original_timesteps = raw_total_timesteps
            raw_eval_timesteps = get_item(eval_file, 'n_updates')
            raw_eval_reward = get_item(eval_file, 'mean_eval_reward')
            print(raw_total_timesteps.shape, raw_eval_reward.shape)
            sr_f = interpolate.interp1d(raw_total_timesteps, raw_success_rate, fill_value="extrapolate")
            eval_f = interpolate.interp1d(raw_eval_timesteps, raw_eval_reward, fill_value="extrapolate")
            step_shrink_fn = interpolate.interp1d(raw_total_timesteps, raw_original_timesteps, fill_value="extrapolate")

            timesteps = np.arange(0, max_timesteps[env_name], max_timesteps[env_name] // 500)
            print(timesteps[0], timesteps[-1], raw_total_timesteps[0], raw_total_timesteps[-1])
            success_rate = sr_f(timesteps)
            eval_reward = eval_f(step_shrink_fn(timesteps))
            timesteps = smooth(timesteps, 50)
            success_rate = smooth(success_rate, 50)
            eval_reward = smooth(eval_reward, 50)
            df_timesteps.append(timesteps)
            df_sr.append(success_rate)
            df_eval.append(eval_reward)
            last_sr.append(success_rate[-1])
            last_eval.append(eval_reward[-1])
            df_legend.append(np.array([subfolder.upper()] * len(timesteps)))

            raw_iterations = get_item(progress_file, 'n_updates')
            iter_step_convert_fn = interpolate.interp1d(raw_iterations, raw_original_timesteps, fill_value="extrapolate")
            iterations = np.arange(0, max_iterationss[env_name], max_iterationss[env_name] // 500)
            eval_iteration = eval_f(iter_step_convert_fn(iterations))
            iterations = smooth(iterations, 50)
            eval_iteration = smooth(eval_iteration, 50)
            df_iteration.append(iterations)
            df_eval_iteration.append(eval_iteration)
            df_legend_iteration.append(np.array([subfolder.upper()] * len(iterations)))
        print(subfolder, 'sr', np.mean(last_sr), 'eval', np.mean(last_eval))
    df_timesteps = np.concatenate(df_timesteps, axis=0).tolist()
    df_sr = np.concatenate(df_sr, axis=0).tolist()
    df_eval = np.concatenate(df_eval, axis=0).tolist()
    df_legend = np.concatenate(df_legend, axis=0).tolist()
    df_iteration = np.concatenate(df_iteration, axis=0).tolist()
    df_eval_iteration = np.concatenate(df_eval_iteration, axis=0).tolist()
    df_legend_iteration = np.concatenate(df_legend_iteration, axis=0).tolist()
    data = {'samples': df_timesteps, 'success_rate': df_sr, 'algo': df_legend}
    sr_timesteps = pandas.DataFrame(data)
    data = {'samples': df_timesteps, 'eval': df_eval, 'algo': df_legend}
    eval_timesteps = pandas.DataFrame(data)
    data = {'iterations': df_iteration, 'eval': df_eval_iteration, 'algo': df_legend_iteration}
    eval_iteration = pandas.DataFrame(data)

    wspace = .3
    bottom = .3
    margin = .1
    # left = .08
    left = .1
    # width = 3.5 / ((1. - left) / (2 + wspace + margin / 2))
    width = 2.15 / ((1. - left) / (2 + wspace + margin / 2))
    height = 1.5 / ((1. - bottom) / (1 + margin / 2))

    plt.style.use("ggplot")
    # plt.rcParams.update({'legend.fontsize': 14})
    p = sns.color_palette()
    sns.set_palette([p[i] for i in range(len(subfolders))])
    # f, axes = plt.subplots(1, 3, figsize=(width, height))
    f, axes = plt.subplots(1, 2, figsize=(width, height))
    sns.lineplot(x='samples', y='success_rate', hue='algo', ax=axes[0], data=sr_timesteps)
    axes[0].set_xlabel('samples')
    axes[0].set_ylabel('success_rate')
    axes[0].get_legend().remove()
    sns.lineplot(x='samples', y='eval', hue='algo', ax=axes[1], data=eval_timesteps)
    axes[1].set_xlabel('samples')
    axes[1].set_ylabel('')
    axes[1].get_legend().remove()
    # sns.lineplot(x='iterations', y='eval', hue='algo', ax=axes[2], data=eval_iteration)
    # axes[2].xaxis.get_major_formatter().set_powerlimits((0, 1))
    # axes[2].set_xlabel('iterations')
    # axes[2].set_ylabel('')
    # axes[2].get_legend().remove()
    handles, labels = axes[1].get_legend_handles_labels()

    f.legend(handles[1:], ['SAC', 'SIR', 'SIL', 'DS'][:len(subfolders)], loc="lower right", ncol=1, bbox_to_anchor=(0.99, 0.18), title='')
    f.subplots_adjust(top=1. - margin / height, bottom=0.21, wspace=wspace, left=left, right=1. - margin / width)
    plt.savefig(os.path.join(folder_name, '../', os.path.basename(folder_name) + '.pdf'))
    plt.show()

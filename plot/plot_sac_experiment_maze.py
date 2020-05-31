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
    assert env_name in ['smaze', 'maze_ego', 'maze_box']
    max_timesteps = {'smaze': 1e6, 'maze_ego': 2.5e7, 'maze_box': 4.9e7}
    df_timesteps, df_sr, df_eval, df_legend, df_iteration, df_success_rate_iteration, df_legend_iteration = [], [], [], [], [], [], []
    subfolders = ['sir', 'hiro']
    if env_name == 'smaze':
        for subfolder in subfolders:
            last_sr = []
            for i in range(3):
                if not os.path.exists(os.path.join(folder_name, subfolder, str(i), 'progress.csv')):
                    continue
                progress_file = os.path.join(folder_name, subfolder, str(i), 'progress.csv')
                if subfolder == 'hiro':
                    raw_success_rate = get_item(progress_file, 'Value')
                    raw_total_timesteps = get_item(progress_file, 'Step')
                else:
                    raw_success_rate = get_item(progress_file, 'ep_rewmean')
                    raw_total_timesteps = get_item(progress_file, 'total timesteps')
                sr_f = interpolate.interp1d(raw_total_timesteps, raw_success_rate, fill_value="extrapolate")
                timesteps = np.arange(0, max_timesteps[env_name], max_timesteps[env_name] // 250)
                print(timesteps[0], timesteps[-1], raw_total_timesteps[0], raw_total_timesteps[-1])
                success_rate = sr_f(timesteps)
                timesteps = smooth(timesteps, 20)
                success_rate = smooth(success_rate, 20)
                # eval_reward = smooth(eval_reward, 20)
                df_timesteps.append(timesteps)
                df_sr.append(success_rate)
                last_sr.append(success_rate[-1])
                # df_eval.append(eval_reward)
                df_legend.append(np.array([subfolder.upper()] * len(timesteps)))

            print(subfolder, np.mean(last_sr))
    else:
        for subfolder in subfolders:
            last_sr = []
            for i in range(3):
                if not os.path.exists(os.path.join(folder_name, subfolder, str(i), 'eval.csv')):
                    continue
                eval_file = os.path.join(folder_name, subfolder, str(i), 'eval.csv' if env_name == 'maze_ego' else 'eval_box.csv')
                if subfolder == 'hiro':
                    raw_success_rate = get_item(eval_file, 'Value')
                    raw_total_timesteps = get_item(eval_file, 'Step')
                    sr_f = interpolate.interp1d(raw_total_timesteps, raw_success_rate, fill_value="extrapolate")
                else:
                    progress_file = os.path.join(folder_name, subfolder, str(i), 'progress.csv')
                    raw_total_timesteps = get_item(progress_file, 'total timesteps')
                    if subfolder == 'sir':
                        original_timesteps = get_item(progress_file, 'original_timesteps')
                    else:
                        original_timesteps = raw_total_timesteps
                    expand_fn = interpolate.interp1d(original_timesteps, raw_total_timesteps, fill_value="extrapolate")
                    success_rate = get_item(eval_file, 'mean_eval_reward')
                    eval_steps = get_item(eval_file, 'n_updates')
                    eval_steps = expand_fn(eval_steps)
                    sr_f = interpolate.interp1d(eval_steps, success_rate, fill_value="extrapolate")
                timesteps = np.arange(0, max_timesteps[env_name], max_timesteps[env_name] // 250)
                print(timesteps[0], timesteps[-1], raw_total_timesteps[0], raw_total_timesteps[-1])
                success_rate = sr_f(timesteps)
                timesteps = smooth(timesteps, 20)
                success_rate = smooth(success_rate, 20)
                # eval_reward = smooth(eval_reward, 20)
                df_timesteps.append(timesteps)
                df_sr.append(success_rate)
                last_sr.append(success_rate[-1])
                # df_eval.append(eval_reward)
                df_legend.append(np.array([subfolder.upper()] * len(timesteps)))

            print(subfolder, np.mean(last_sr))

    df_timesteps = np.concatenate(df_timesteps, axis=0).tolist()
    df_sr = np.concatenate(df_sr, axis=0).tolist()
    # df_eval = np.concatenate(df_eval, axis=0).tolist()
    df_legend = np.concatenate(df_legend, axis=0).tolist()
    data = {'samples': df_timesteps, 'success_rate': df_sr, 'algo': df_legend}
    sr_timesteps = pandas.DataFrame(data)

    wspace = .3
    bottom = .3
    margin = .1
    left = .1
    width = 1.25 / ((1. - left) / (2 + wspace + margin / 2))
    height = 1.5 / ((1. - bottom) / (1 + margin / 2))

    plt.style.use("ggplot")
    # plt.rcParams.update({'legend.fontsize': 14})
    p = sns.color_palette()
    sns.set_palette([p[i] for i in range(len(subfolders))])
    f, axes = plt.subplots(1, 1, figsize=(width, height))
    sns.lineplot(x='samples', y='success_rate', hue='algo', ax=axes, data=sr_timesteps)
    axes.set_xlabel('samples')
    axes.set_ylabel('success_rate')
    axes.xaxis.get_major_formatter().set_powerlimits((0, 1))
    axes.get_legend().remove()

    handles, labels = axes.get_legend_handles_labels()

    f.legend(handles[1:], ['SIR', 'HIRO'], loc="lower right", ncol=1, bbox_to_anchor=(0.99, 0.18), title='')
    f.subplots_adjust(top=1. - margin / height, bottom=0.21, wspace=wspace, left=left, right=1. - margin / width)
    plt.savefig(os.path.join(folder_name, '../', os.path.basename(folder_name) + env_name + '.pdf'))
    plt.show()

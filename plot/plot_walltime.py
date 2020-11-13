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
        out[i] = np.nanmean(array[i:i + window])
    return out


if __name__ == '__main__':
    folder_name = sys.argv[1]
    env_name = sys.argv[2]
    assert env_name in ['particle']
    max_timesteps = {'umaze': 1e5, 'maze_ego': 2.5e7, 'maze_box': 4.9e7}
    df_walltime, df_sr, df_eval, df_legend = [], [], [], []
    # df_timesteps, df_sr, df_eval, df_legend, df_iteration, df_success_rate_iteration, df_legend_iteration = [], [], [], [], [], [], []
    subfolders = ['sac', 'ppo']
    if env_name == "particle":
        for subfolder in subfolders:
            last_sr = []
            for i in range(3):
                if not os.path.exists(os.path.join(folder_name, subfolder, str(i), 'progress.csv')):
                    continue
                progress_file = os.path.join(folder_name, subfolder, str(i), 'progress.csv')
                eval_file = os.path.join(folder_name, subfolder, str(i), 'eval.csv')
                raw_walltime = get_item(progress_file, 'time_elapsed')
                raw_success_rate = get_item(progress_file, 'ep_success_rate') if subfolder == "ppo" else get_item(progress_file, 'success rate')
                if subfolder == "ppo":
                    raw_walltime = raw_walltime[:560]
                    raw_success_rate = raw_success_rate[:560]
                sr_f = interpolate.interp1d(raw_walltime, raw_success_rate, bounds_error=False)
                wall_time = np.arange(0, 3.3e5, 3.3e5 // 250)
                success_rate = sr_f(wall_time)
                print(wall_time[0], wall_time[-1], raw_walltime[0], raw_walltime[-1])

                if subfolder == "ppo":
                    print(len(wall_time))
                    print(success_rate[-10:])
                wall_time = smooth(wall_time, 10)
                success_rate = smooth(success_rate, 10)
                # eval_reward = smooth(eval_reward, 20)
                df_walltime.append(wall_time)
                df_sr.append(success_rate)
                last_sr.append(success_rate[-1])
                # df_eval.append(eval_reward)
                df_legend.append(np.array([subfolder.upper()] * len(wall_time)))

            print(subfolder, np.mean(last_sr))

    # df_timesteps = np.concatenate(df_timesteps, axis=0).tolist()
    df_walltime = np.concatenate(df_walltime, axis=0).tolist()
    df_sr = np.concatenate(df_sr, axis=0).tolist()
    # df_eval = np.concatenate(df_eval, axis=0).tolist()
    df_legend = np.concatenate(df_legend, axis=0).tolist()
    data = {'wall time': df_walltime, 'success_rate': df_sr, 'algo': df_legend}
    sr_walltime = pandas.DataFrame(data)

    wspace = .3
    bottom = .3
    margin = .1
    left = .15
    width = 1.5 / ((1. - left) / (2 + wspace + margin / 2))
    height = 1.5 / ((1. - bottom) / (1 + margin / 2))

    plt.style.use("ggplot")
    # plt.rcParams.update({'legend.fontsize': 14})
    p = sns.color_palette()
    sns.set_palette([p[i] for i in range(len(subfolders))])
    f, axes = plt.subplots(1, 1, figsize=(width, height))
    sns.lineplot(x='wall time', y='success_rate', hue='algo', ax=axes, data=sr_walltime)
    axes.set_xlabel('wall time')
    axes.set_ylabel('success_rate')
    axes.xaxis.get_major_formatter().set_powerlimits((0, 1))
    axes.get_legend().remove()

    handles, labels = axes.get_legend_handles_labels()

    f.legend(handles[:], ['SAC', 'PPO'], loc="lower right", ncol=1, bbox_to_anchor=(0.99, 0.18), title='')
    f.subplots_adjust(top=1. - margin / height, bottom=0.21, wspace=wspace, left=left, right=1. - margin / width)
    plt.savefig(os.path.join(folder_name, '../', os.path.basename(folder_name) + env_name + 'walltime.pdf'))
    plt.show()

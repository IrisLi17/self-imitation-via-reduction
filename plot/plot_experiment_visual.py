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
    assert env_name in ['uwall']
    # assert mode in ['train', 'hard', 'iteration']
    max_timesteps = {'uwall': 2.1e6}
    df_timesteps, df_sr, df_legend= [], [], []
    subfolders = ['ppo', 'sir', 'sil']
    # subfolders = ['ppo_attention_new', 'ppo_attention_sir_new', 'ppo_attention_sil_new']

    for subfolder in subfolders:
        last_sr = []
        for i in range(3):
            if not os.path.exists(os.path.join(folder_name, subfolder, str(i), 'progress.csv')):
                continue
            progress_file = os.path.join(folder_name, subfolder, str(i), 'progress.csv')
            raw_success_rate = get_item(progress_file, 'ep_success_rate')
            raw_total_timesteps = get_item(progress_file, 'total_timesteps')
            print(raw_total_timesteps.shape)
            sr_f = interpolate.interp1d(raw_total_timesteps, raw_success_rate, fill_value="extrapolate")
            timesteps = np.arange(0, max_timesteps[env_name], max_timesteps[env_name] // 70)
            print(timesteps[0], timesteps[-1], raw_total_timesteps[0], raw_total_timesteps[-1])
            success_rate = sr_f(timesteps)
            timesteps = smooth(timesteps, 5)
            success_rate = smooth(success_rate, 5)
            df_timesteps.append(timesteps)
            df_sr.append(success_rate)
            last_sr.append(success_rate[-1])
            df_legend.append(np.array([subfolder.upper()] * len(timesteps)))

        print(subfolder, 'sr', np.mean(last_sr))
    df_timesteps = np.concatenate(df_timesteps, axis=0).tolist()
    df_sr = np.concatenate(df_sr, axis=0).tolist()
    df_legend = np.concatenate(df_legend, axis=0).tolist()
    data = {'samples': df_timesteps, 'success_rate': df_sr, 'algo': df_legend}
    sr_timesteps = pandas.DataFrame(data)

    wspace = .3
    bottom = .3
    margin = .1
    # left = .08
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
    axes.set_ylabel('avg. succ. rate')
    axes.get_legend().remove()

    handles, labels = axes.get_legend_handles_labels()
    f.legend(handles[:], ['PPO', 'SIR', 'SIL'], loc="lower right", ncol=1, bbox_to_anchor=(0.99, 0.18), title='')
    f.subplots_adjust(top=1. - margin / height, bottom=0.21, wspace=wspace, left=left, right=1. - margin / width)
    plt.savefig(os.path.join(folder_name, '../', os.path.basename(folder_name) + '.pdf'))
    print(os.path.join(folder_name, '../', os.path.basename(folder_name) + '.pdf'))
    plt.show()

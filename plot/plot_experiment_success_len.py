import seaborn as sns
import pandas
import numpy as np
import sys, os
from scipy import interpolate
import matplotlib.pyplot as plt
from stable_baselines.bench.monitor import load_results

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
    alg = sys.argv[3]
    assert alg in ['ppo', 'sac']
    # assert mode in ['train', 'hard', 'iteration']
    if alg == 'ppo':
        max_timesteps = {'push': 4.99e7,
                         'particle': 2.5e8,
                         'maze': 1.5e6,
                         'stacking': 2e8,}
    elif alg == 'sac':
        max_timesteps = {'push': 1.36e7,}
    # max_iterationss = {'push': 750,
    #                    'particle': 510,
    #                    'maze': 245,}
    # df_timesteps, df_sr, df_eval, df_legend, df_iteration, df_eval_iteration, df_legend_iteration = [], [], [], [], [], [], []
    df_iteration, df_len_mean, df_legend_iteration = [], [], []
    subfolders = [alg, 'sir', 'sil']
    if 'particle_random0.7' in folder_name:
        subfolders = ['ppo', 'sir', 'sil']
    elif 'particle_random1.0' in folder_name:
        subfolders = ['ppo', 'sir', 'sil']
    elif 'maze' in folder_name:
        subfolders = ['ppo', 'sir_re2']
    for subfolder in subfolders:
        last_success_len = []
        for i in range(3):
            if not os.path.exists(os.path.join(folder_name, subfolder, str(i), '0.monitor.csv')):
                continue
            monitor_df = load_results(os.path.join(folder_name, subfolder, str(i)))
            raw_len = monitor_df.l
            raw_success = monitor_df.is_success
            cum_len = raw_len.cumsum()
            masked_len = smooth(raw_len[raw_success > 0.5].values, 100)
            masked_cum_len = smooth(cum_len[raw_success > 0.5].values, 100)
            success_len_f = interpolate.interp1d(masked_cum_len, masked_len, fill_value="extrapolate")
            print(masked_cum_len[-1], max_timesteps[env_name])
            timesteps = np.arange(0, max_timesteps[env_name], max_timesteps[env_name] // 500)
            success_len = success_len_f(timesteps)
            # iterations = timesteps / timesteps[-1] * max_iterationss[env_name]
            # iterations = smooth(iterations, 20)
            timesteps = smooth(timesteps, 20)
            success_len = smooth(success_len, 20)

            last_success_len.append(success_len[-1])

            df_iteration.append(timesteps)
            df_len_mean.append(success_len)
            df_legend_iteration.append(np.array([subfolder.upper()] * len(timesteps)))
            assert len(timesteps) == len(success_len)
        print(subfolder, np.mean(last_success_len))
    df_iteration = np.concatenate(df_iteration, axis=0).tolist()
    df_len_mean = np.concatenate(df_len_mean, axis=0).tolist()
    df_legend_iteration = np.concatenate(df_legend_iteration, axis=0).tolist()
    data = {'timesteps': df_iteration, 'len_mean': df_len_mean, 'algo': df_legend_iteration}
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
    sns.set_palette([p[i] for i in range(len(subfolders))])
    f, axes = plt.subplots(1, 1, figsize=(width, height))
    sns.lineplot(x='timesteps', y='len_mean', hue='algo', ax=axes, data=len_mean_iteration)
    axes.set_xlabel('timesteps')
    axes.set_ylabel('episode length')
    axes.get_legend().remove()
    handles, labels = axes.get_legend_handles_labels()
    f.legend(handles[1:], [alg.upper(), 'SIR', 'SIL'], loc="upper right", ncol=1, title='')
    f.subplots_adjust(top=1. - margin / height, bottom=0.2, wspace=wspace, left=left, right=1. - margin / width)
    plt.savefig(os.path.join(folder_name, '../', os.path.basename(folder_name) + '_successlen.pdf'))
    # plt.show()

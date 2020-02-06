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
    mode = sys.argv[2]
    assert mode in ['success_rate', 'augment', 'eval']
    plt.style.use("ggplot")
    plt.rcParams.update({'font.size': 20, 'legend.fontsize': 20,
                         'axes.formatter.limits': [-5, 3]})
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for subfolder in ['sir_re1', 'sir_re4', 'sir_re8', 'sir_re16']:
        progress_file = os.path.join(folder_name, subfolder, '0', 'progress.csv')
        eval_file = os.path.join(folder_name, subfolder, '0', 'eval.csv')
        success_rate = get_item(progress_file, 'ep_reward_mean')
        total_timesteps = get_item(progress_file, 'total_timesteps')
        original_steps_per_iter = get_item(progress_file, 'original_timesteps')[0]
        augment_steps = get_item(progress_file, 'augment_steps')
        augment_ratio = augment_steps / (augment_steps + original_steps_per_iter)
        eval_reward = get_item(eval_file, 'mean_eval_reward')
        L = np.sum(total_timesteps < 3e7)
        total_timesteps = smooth(total_timesteps[:L], 20)
        success_rate = smooth(success_rate[:L], 20)
        augment_ratio = smooth(augment_ratio[:L], 20)
        augment_number = smooth(augment_steps[:L], 20)
        eval_reward = smooth(eval_reward[:L], 20)
        if mode == 'success_rate':
            ax.plot(total_timesteps, success_rate, label=subfolder.upper())
            ax.set_ylabel('success rate')
        elif mode == 'augment':
            ax.plot(total_timesteps, augment_number, label=subfolder.upper())
            ax.set_ylabel('number of augmented data')
        elif mode == 'eval':
            ax.plot(total_timesteps, eval_reward, label=subfolder.upper())
            ax.set_ylabel('success rate')
        ax.set_xlabel('samples')
    if mode == 'augment':
        plt.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), ncol=1)
    plt.savefig('reuse_' + mode + '.png')



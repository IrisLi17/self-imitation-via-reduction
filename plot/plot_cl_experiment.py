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
    assert env_name in ['maze']
    # assert mode in ['train', 'hard', 'iteration']
    max_timesteps = {'maze': 4e6,
                     }
    max_iterationss = {'maze': 245,}
    df_timesteps, df_sr, df_eval, df_legend, df_iteration, df_success_rate_iteration, df_legend_iteration = [], [], [], [], [], [], []
    # subfolders = ['ppo_sir', 'ppo_cu', 'goal_gan_b1000', ] # 'goal_gan_b10000'
    subfolders = ['goal_gan_b1000', 'goal_gan_b10000']
    last_sr = []
    for subfolder in subfolders:
        if subfolder == "ppo_cu":
            for i in range(3):
                eval_file = os.path.join(folder_name, subfolder, str(i), 'eval.csv')
                raw_success_rate = get_item(eval_file, 'mean_eval_reward')
                raw_total_timesteps = get_item(eval_file, 'n_updates') * 1000
                sr_f = interpolate.interp1d(raw_total_timesteps, raw_success_rate, bounds_error=False)
                timesteps = np.arange(0, max_timesteps[env_name], max_timesteps[env_name] // 245)
                print(timesteps[0], timesteps[-1], raw_total_timesteps[0], raw_total_timesteps[-1])
                success_rate = sr_f(timesteps)
                # eval_reward = eval_f(timesteps)
                timesteps = smooth(timesteps, 10)
                success_rate = smooth(success_rate, 10)
                # eval_reward = smooth(eval_reward, 20)
                df_timesteps.append(timesteps)
                df_sr.append(success_rate)
                last_sr.append(success_rate[-1])
                # df_eval.append(eval_reward)
                df_legend.append(np.array([subfolder.upper()] * len(timesteps)))
        else:
            for i in range(3):
                if not os.path.exists(os.path.join(folder_name, subfolder, str(i), 'progress.csv')):
                    continue
                progress_file = os.path.join(folder_name, subfolder, str(i), 'progress.csv')
                raw_success_rate = get_item(progress_file, 'ep_reward_mean' if subfolder == "ppo_sir" or subfolder == "ppo" else 'Outer_MeanRewards')
                raw_total_timesteps = get_item(progress_file, 'total_timesteps' if subfolder == "ppo_sir" or subfolder == "ppo" else 'Outer_timesteps')
                sr_f = interpolate.interp1d(raw_total_timesteps, raw_success_rate, bounds_error=False)
                timesteps = np.arange(0, max_timesteps[env_name], max_timesteps[env_name] // 245)
                print(timesteps[0], timesteps[-1], raw_total_timesteps[0], raw_total_timesteps[-1])
                success_rate = sr_f(timesteps)
                # eval_reward = eval_f(timesteps)
                timesteps = smooth(timesteps, 5)
                success_rate = smooth(success_rate, 5)
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
    # data = {'samples': df_timesteps, 'eval': df_eval, 'algo': df_legend}
    # eval_timesteps = pandas.DataFrame(data)

    wspace = .3
    bottom = .3
    margin = .1
    left = .15
    width = 1.7 / ((1. - left) / (2 + wspace + margin / 2))
    height = 1.5 / ((1. - bottom) / (1 + margin / 2))

    plt.style.use("ggplot")
    # plt.rcParams.update({'legend.fontsize': 14})
    p = sns.color_palette()
    sns.set_palette([p[0], p[1], p[2], p[3]])
    f, axes = plt.subplots(1, 1, figsize=(width, height))
    sns.lineplot(x='samples', y='success_rate', hue='algo', ax=axes, data=sr_timesteps)
    axes.set_xlabel('samples')
    axes.set_ylabel('success_rate')
    axes.get_legend().remove()
    # sns.lineplot(x='samples', y='eval', hue='algo', ax=axes[1], data=eval_timesteps)
    # axes[1].set_xlabel('samples')
    # axes[1].set_ylabel('')
    # axes[1].get_legend().remove()
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
    # f.legend(handles[:], ['SIR', 'Manual CL', 'GoalGAN', 'GoalGAN_10k'], loc="lower right", ncol=1, bbox_to_anchor=(0.99, 0.18), title='')
    f.legend(handles[:], ['GoalGAN_1k', 'GoalGAN_10k'], loc="lower right", ncol=1,
             bbox_to_anchor=(0.99, 0.18), title='')
    f.subplots_adjust(top=1. - margin / height, bottom=0.21, wspace=wspace, left=left, right=1. - margin / width)
    plt.savefig(os.path.join(folder_name, '../', os.path.basename(folder_name) + '.pdf'))
    # plt.show()

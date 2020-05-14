import sys, os
import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy import interpolate


if __name__ == '__main__':
    option = sys.argv[1]
    log_paths = sys.argv[2:]
    assert option in ['eval']
    window = 20
    def get_item(log_file, label):
        data = pandas.read_csv(log_file, index_col=None, comment='#', error_bad_lines=True)
        return data[label].values
    def smooth(array, window):
        out = np.zeros(array.shape[0] - window)
        for i in range(out.shape[0]):
            out[i] = np.mean(array[i:i + window])
        return out
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for log_path in log_paths:
        progress_file = os.path.join(log_path, 'progress.csv')
        eval_file = os.path.join(log_path, 'eval.csv')
        if 'hiro' in log_path:
            eval_reward = get_item(eval_file, 'Value')
            eval_step = get_item(eval_file, 'Step')
        else:
            eval_reward = get_item(eval_file, 'mean_eval_reward')
            total_timesteps = get_item(progress_file, 'total timesteps')
            original_timesteps = get_item(progress_file, 'original_timesteps')
            step_expand_fn = interpolate.interp1d(original_timesteps, total_timesteps, fill_value="extrapolate")
            eval_step = get_item(eval_file, 'n_updates')
        if option == 'eval':
            ax.plot(smooth(eval_step, window), smooth(eval_reward, window), label=log_path)
    if option == 'eval':
        ax.set_title('success rate')
    ax.grid()
    plt.legend()
    plt.show()
    

import sys, os
import numpy as np
import pandas
import matplotlib.pyplot as plt


if __name__ == '__main__':
    log_path = sys.argv[1]
    window = 10
    def get_item(log_file, label):
        data = pandas.read_csv(log_file, index_col=None, comment='#', error_bad_lines=True)
        return data[label].values
    def smooth(array, window):
        out = np.zeros(array.shape[0] - window)
        for i in range(out.shape[0]):
            out[i] = np.mean(array[i:i + window])
        return out
    # print(get_item(log_path, 'reference_value').shape)
    original_value = get_item(log_path, 'reference_value')[:]
    value1 = get_item(log_path, 'value1')[:]
    value2 = get_item(log_path, 'value2')[:]
    is_success = get_item(log_path, 'is_success')[:]
    success_idx = np.where(is_success > 0.5)[0]
    fail_idx = np.where(is_success < 0.5)[0]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # ax.plot(smooth(original_value, 100), alpha=0.5, label='reference')
    # ax.scatter(fail_idx, value1[fail_idx]-original_value[fail_idx], c='tab:orange', s=0.1, label='fail value1')
    # ax.scatter(fail_idx, value2[fail_idx]-original_value[fail_idx], c='tab:green', s=0.1, label='fail value2')
    # ax.scatter(success_idx, value1[success_idx]-original_value[success_idx], c='tab:red', s=0.1, label='success value1')
    # ax.scatter(success_idx, value2[success_idx]-original_value[success_idx], c='tab:purple', s=0.1, label='success value2')
    ax.scatter(fail_idx, (value1[fail_idx] + value2[fail_idx]) / 2, c='tab:orange', s=0.1, label='fail mean value')
    ax.scatter(success_idx, (value1[success_idx] + value2[success_idx]) / 2, c='tab:green', s=4.0, label='success mean value')
    ax.plot(smooth(np.arange(len(value1)), 500), smooth((value1 + value2) / 2, 500), c='tab:red', label='smoothed mean value')
    # ax.scatter(fail_idx, original_value[fail_idx], c='tab:orange', s=0.1, label='fail original value')
    # ax.scatter(success_idx, original_value[success_idx], c='tab:green', s=4.0, label='success original value')
    plt.legend()
    plt.show()
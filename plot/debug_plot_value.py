import sys, os
import numpy as np
import pandas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == '__main__':
    log_path = sys.argv[1]
    plot_choice = sys.argv[2]
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
    # original_value = get_item(log_path, 'reference_value')[:]
    value1 = get_item(log_path, 'value1')[0:]
    value2 = get_item(log_path, 'value2')[0:]
    normalize_value1 = get_item(log_path,'normalize_value1 ')[0:]
    normalize_value2 = get_item(log_path,'normalize_value2')[0:]
    value_mean = (value1+value2)/2
    value_prod = normalize_value1*normalize_value2
    # value_prod = get_item(log_path,'value_prod')[0:]
    # value_mean = get_item(log_path,'value_mean')[0:]

    min_value = np.min(np.concatenate([np.expand_dims(value1, axis=0), np.expand_dims(value2, axis=0)], axis=0), axis=0)
    is_success = get_item(log_path, 'is_success')[0:]
    num_timesteps = get_item(log_path, 'num_timesteps')[0:]
    filtered_idx = np.where(num_timesteps<3e6)
    value1_select = value1[filtered_idx]
    value2_select = value2[filtered_idx]
    is_success_select = is_success[filtered_idx]

    # print(num_timesteps[20000], num_timesteps[40000], num_timesteps[-1])
    success_idx = np.where(is_success > 0.5)[0]
    fail_idx = np.where(is_success < 0.5)[0]
    success_idx_select = np.where(is_success_select>0.5)[0]
    fail_idx_select = np.where(is_success_select<0.5)[0]
    print(value1_select.shape)
    print(value1.shape)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    plt.rcParams.update({'font.size': 22, 'legend.fontsize': 22, 'xtick.labelsize': 18, 'ytick.labelsize': 18, 'axes.labelsize': 18})
    # ax.plot(smooth(original_value, 100), alpha=0.5, label='reference')
    # ax.scatter(fail_idx, value1[fail_idx]-original_value[fail_idx], c='tab:orange', s=0.1, label='fail value1')
    # ax.scatter(fail_idx, value2[fail_idx]-original_value[fail_idx], c='tab:green', s=0.1, label='fail value2')
    # ax.scatter(success_idx, value1[success_idx]-original_value[success_idx], c='tab:red', s=0.1, label='success value1')
    # ax.scatter(success_idx, value2[success_idx]-original_value[success_idx], c='tab:purple', s=0.1, label='success value2')
    # Mean value
    if plot_choice=='prod':

        ax.scatter(fail_idx,value_prod[fail_idx],c='tab:red',s=1.0,label='fail prod value')
        ax.scatter(success_idx,value_prod[success_idx],c='tab:blue',s=4.0,label='success prod value')
    elif plot_choice == 'norm':
        ax.scatter(fail_idx, normalize_value1[fail_idx], c='tab:orange', s=1.0, label='fail normalize value1')
        ax.scatter(success_idx, normalize_value1[success_idx], c='tab:green', s=4.0, label='success normalize value1')
        ax.scatter(fail_idx, normalize_value2[fail_idx], c='tab:purple', s=1.0, label='fail normalize value2')
        ax.scatter(success_idx, normalize_value2[success_idx], c='tab:pink', s=4.0, label='success normalize value2')
    elif plot_choice=='mean':
        # ax.scatter(fail_idx, (value1[fail_idx] + value2[fail_idx]) / 2, c='tab:orange', s=1.0, label='fail mean value')
        # ax.scatter(success_idx, (value1[success_idx] + value2[success_idx]) / 2, c='tab:green', s=4.0,
        #            label='success mean value')
        ax.scatter(fail_idx_select, (value1_select[fail_idx_select] + value2_select[fail_idx_select]) / 2, c='tab:orange', s=1.0, label='fail mean value')
        ax.scatter(success_idx_select, (value1_select[success_idx_select] + value2_select[success_idx_select]) / 2, c='tab:green', s=4.0,
                   label='success mean value')
        # ax.plot(fail_idx,-1*np.ones(fail_idx.shape),c='tab:red')
        # ax.plot(fail_idx,-0.5*np.ones(fail_idx.shape),c='tab:red')
        # ax.plot(fail_idx,-0.8*np.ones(fail_idx.shape),c='tab:red')
        # ax.plot(fail_idx,-0.9*np.ones(fail_idx.shape),c='tab:red')
        # ax.plot(smooth(success_idx,window),smooth((value1[success_idx]+value2[success_idx])/2,window),c='tab:red')
        # ax.plot(smooth(np.arange(value1.shape[0]),window),smooth((value1+value2)/2,window),c='tab:blue')
        # ax.plot(fail_idx,-1.2*np.ones(fail_idx.shape),c='tab:red')
    elif plot_choice=='value':
        ax.scatter(fail_idx, value1[fail_idx], c='tab:red', s=1.0, label='fail value1')
        ax.scatter(success_idx, value1[success_idx], c='tab:blue', s=4.0, label='success value1')
        ax.scatter(fail_idx, value2[fail_idx], c='tab:purple', s=1.0, label='fail value2')
        ax.scatter(success_idx, value2[success_idx], c='tab:pink', s=4.0, label='success value2')

    # ax.axhline(0.5, linestyle='--', c='tab:blue')
    # ax.axhline(1.0, linestyle='--', c='tab:blue')
    # ax.plot(smooth(np.arange(len(value1)), 500), smooth((value1 + value2) / 2, 500), c='tab:red', label='smoothed mean value')
    # Min value
    # ax.scatter(fail_idx, min_value[fail_idx], c='tab:orange', s=1.0, label='fail min value')
    # ax.scatter(success_idx, min_value[success_idx], c='tab:green', s=4.0, label='succes min value')
    # ax.plot(smooth(np.arange(len(min_value)), 500), smooth(min_value, 500), c='tab:red', label='smoothed min value')
    # ax.scatter(fail_idx, original_value[fail_idx], c='tab:orange', s=0.1, label='fail original value')
    # ax.scatter(success_idx, original_value[success_idx], c='tab:green', s=4.0, label='success original value')
    # ax.set_yscale('log')
    plt.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout(pad=0.05)
    plt.savefig('value_sigma_sac'+str(plot_choice)+'.png')
    # plt.show()

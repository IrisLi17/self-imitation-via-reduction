import sys, os
import numpy as np
import pandas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == '__main__':
    option = sys.argv[1]
    log_paths = sys.argv[2:]
    assert option in ['success_rate', 'eval', 'entropy', 'aug_ratio', 'self_aug_ratio']
    window = 20
    def get_item(log_file, label):
        data = pandas.read_csv(log_file, index_col=None, comment='#', error_bad_lines=True)
        return data[label].values
    def smooth(array, window):
        out = np.zeros(array.shape[0] - window)
        for i in range(out.shape[0]):
            out[i] = np.mean(array[i:i + window])
        return out
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for log_path in log_paths:
        progress_file = os.path.join(log_path, 'progress.csv')
        eval_file = os.path.join(log_path, 'eval.csv')
        # if 'ds' in log_path:
        if True:
            success_rate = get_item(progress_file, 'success rate')
        else:
            success_rate = get_item(progress_file, 'ep_reward_mean')
        ep_reward_mean = get_item(progress_file,'ep_rewmean')
        ep_len_mean = get_item(progress_file,'eplenmean')
            # success_rate = get_item(progress_file, 'ep_reward_mean')
        total_timesteps = get_item(progress_file, 'total timesteps')
        time_total = get_item(progress_file,'time_elapsed')
        idx = np.where(total_timesteps<1e7)
        # total_timesteps=total_timesteps[idx]
        total_timesteps = total_timesteps/1e6
        # success_rate = success_rate[idx]
        # if 'sac_her' in log_path:
        #     print('log_path',log_path)
        #     original_steps = total_timesteps
        #
        # else:
        try:
            train_time = get_item(progress_file,'train_time')
            step_time = get_item(progress_file,'step_time')
            store_time = get_item(progress_file,'store_time')
            original_steps = get_item(progress_file, 'original_timesteps')/1e6
        except:
            original_steps=total_timesteps
            train_time = np.zeros(total_timesteps.shape)
            step_time = np.zeros(total_timesteps.shape)
            store_time = np.zeros(total_timesteps.shape)
        entropy = get_item(progress_file, 'entropy')
        try:
            eval_reward = get_item(eval_file, 'mean_eval_reward')
            n_updates = get_item(eval_file, 'n_updates')
        except:
            pass

        # success_rate = smooth(success_rate, window)
        # total_timesteps = smooth(total_timesteps, window)
        print(log_path)
        label = ""
        if option == 'success_rate':
            if log_path=='../logs/pnr_sac/sac_dense_reward':
                # print(True)
                label = "sir_dense_reward"
            # elif log_path =='../logs/pnr_sac/sac_latent_debug':
            #     label = "sir_sparse_reward"
            elif log_path == '../logs/pnr_sac/sac_her_dense_reward':
                label="sac_her_dense_reward"
            elif log_path == '../logs/pnr_sac/sac_her_sparse_reward':
                label = "sac_her_sparse_reward"
            elif log_path == '../logs/pnr_sac/sac_sir_sparse_reward':
                label = "sir_sparse_reward"
            else:
                label = log_path
            # ax[0].plot(smooth(total_timesteps, window), smooth(success_rate, window), label=label)
            ax[0].plot(smooth(original_steps, window), smooth(success_rate, window), label=label)

            # ax[0].plot(smooth(total_timesteps, window), smooth(ep_reward_mean, window), label=label)

        elif option == 'eval':
            # ax[0].plot(n_updates*65536, eval_reward, label=log_path)
            
            ax[0].plot(smooth(total_timesteps[n_updates-1], window), smooth(eval_reward, window), label=log_path)
        elif option == 'entropy':
            ax[0].plot(smooth(total_timesteps, window), smooth(entropy, window), label=log_path)
        elif option == 'aug_ratio':
            original_success = get_item(progress_file, 'original_success')
            total_success = get_item(progress_file, 'total_success')
            aug_ratio = (total_success - original_success) / (total_success + 1e-8)
            print(total_timesteps.shape, aug_ratio.shape)
            ax[0].plot(smooth(total_timesteps, 2), smooth(aug_ratio, 2), label=log_path)
        elif option == 'self_aug_ratio':
            self_aug_ratio = get_item(progress_file, 'self_aug_ratio')
            ax[0].plot(smooth(total_timesteps, window), smooth(self_aug_ratio, window), label=log_path)
        try:
            # original_steps = get_item(progress_file, 'original_timesteps')[0]
            augment_steps = get_item(progress_file, 'augmented steps') / original_steps
            num_suc_aug_steps = get_item(progress_file,'num_success_aug_steps')
            num_aug_steps = get_item(progress_file,'num_aug_steps')
            # suc_aug_ratio = num_suc_aug_steps/num_aug_steps
            # augment_steps = smooth(augment_steps, window)
        except:
            augment_steps = np.zeros(total_timesteps.shape)
            num_aug_steps = np.zeros(total_timesteps.shape)
            suc_aug_ratio = np.zeros(total_timesteps.shape)
        if log_path == '../logs/pnr_sac/sac_dense_reward':
            # print(True)
            label = 'sir_dense_reward'
        # elif log_path == '../logs/pnr_sac/sac_latent_debug':
        #     label = 'sir_sparse_reward'
        elif log_path == '../logs/pnr_sac/sac_her_dense_reward':
            label = 'sac_her_dense_reward'
        elif log_path == '../logs/pnr_sac/sac_her_sparse_reward':
            label = 'sac_her_sparse_reward'
        elif log_path == '../logs/pnr_sac/sac_sir_sparse_reward':
            label = 'sir_sparse_reward'
        else:
            label = log_path
        # ax[1].plot(smooth(total_timesteps, window), smooth(augment_steps, window), label=label)
        # ax[1].plot(smooth(total_timesteps, window), smooth(ep_len_mean, window), label=label)
        # ax[1].plot(smooth(total_timesteps, window), smooth(suc_aug_ratio, window), label=label)
        # ax[1].plot(total_timesteps,suc_aug_ratio,label=label)
        # ax[1].plot(smooth(total_timesteps, window), smooth(time_total, window), label=label)
        ax[1].plot(smooth(total_timesteps, window), smooth(time_total, window), label=label)

        # ax[1].plot(smooth(total_timesteps,window),smooth(num_suc_aug_steps,window),label=label)
    if option == 'success_rate':
        # ax[0].set_title('ep reward mean')
        ax[0].set_title( 'success rate')
    elif option == 'eval':
        ax[0].set_title('eval success rate')
    elif option == 'entropy':
        ax[0].set_title('entropy')
    elif option == 'aug_ratio':
        ax[0].set_title('aug success episode / total success episode')
    elif option == 'self_aug_ratio':
        ax[0].set_title('self_aug_ratio')
    ax[1].set_title('number of augment steps')
    ax[0].set_xlabel('samples(1e6)')
    ax[0].grid()
    ax[1].grid()
    # plt.legend()
    plt.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout(pad=0.05)

    # plt.show()
    plt.savefig('compare_success_rate'  + '.png')

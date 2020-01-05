import argparse, os
import pandas
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines.results_plotter import plot_results

def arg_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', type=str, default='FetchReach-v1')
    parser.add_argument('--log_path', default=None, type=str)
    parser.add_argument('--xaxis', type=str, default='timesteps')
    parser.add_argument('--yaxis', type=str, default='success_rate')
    args = parser.parse_args()
    return args

def main(args):
    log_dir = args.log_path if args.log_path is not None else os.path.join("./logs", args.env, "her")
    assert args.yaxis in ['success_rate', 'mean_num_augment_ep']
    try:
        plot_results([log_dir], num_timesteps=None, xaxis=args.xaxis, task_name=args.env)
    except:
        progress_file = os.path.join(log_dir, 'progress.csv')

        def get_item(log_file, label):
            data = pandas.read_csv(log_file, index_col=None, comment='#', error_bad_lines=False)
            return data[label].values
        success_rate = get_item(progress_file, 'success rate')
        total_timesteps = get_item(progress_file, 'total timesteps')
        print('final success rate', np.mean(success_rate[-1000:]))
        try:
            mean_num_augment_ep = get_item(progress_file, 'mean_success_augment_ep')
            print('final augment ep', np.mean(mean_num_augment_ep[-1000:]))
        except:
            pass
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.plot(total_timesteps, success_rate)
        ax1.set_ylabel('success rate')
        ax1.set_xlabel('timesteps')
        try:
            ax2.plot(total_timesteps, mean_num_augment_ep)
            ax2.set_ylabel('#augmented successful ep')
            ax2.set_xlabel('timesteps')
        except:
            pass
        # if args.yaxis == 'success_rate':
        #     plt.plot(total_timesteps, success_rate)
        #     plt.ylabel('success rate')
        # elif args.yaxis == 'mean_num_augment_ep':
        #     plt.plot(total_timesteps, mean_num_augment_ep)
        #     plt.ylabel('#augment ep')
        #
        # plt.xlabel('timesteps')
        # plt.title(args.env)
    plt.savefig(os.path.join(log_dir, 'learning_curve.png'))
    plt.show()

if __name__ == '__main__':
    args = arg_parse()
    main(args)

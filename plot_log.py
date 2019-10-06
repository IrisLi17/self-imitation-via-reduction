import argparse, os
import matplotlib.pyplot as plt
from stable_baselines.results_plotter import plot_results

def arg_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', type=str, default='FetchReach-v1')
    parser.add_argument('--xaxis', type=str, default='timesteps')
    args = parser.parse_args()
    return args

def main(args):
    log_dir = os.path.join("./logs", args.env, "her")
    plot_results([log_dir], num_timesteps=None, xaxis=args.xaxis, task_name=args.env)
    plt.show()

if __name__ == '__main__':
    args = arg_parse()
    main(args)

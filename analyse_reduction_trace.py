import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    trace_file = sys.argv[1]
    initial_states = []
    with open(trace_file, 'rb') as f:
        try:
            while True:
                initial_states.append(pickle.load(f))
        except EOFError:
            pass
    print('total number of states', len(initial_states))

    def n_doors_blocked(obs):
        agent_pos = obs[:3]
        box_pos = obs[3: 6]
        goal_pos = obs[-7: -4]
        obstacles_pos = [obs[6 + 3 * i: 9 + 3 * i] for i in range(3)]
        # return sum([abs(pos[1] - 2.5) < 0.1 for pos in obstacles_pos])

        max_x, min_x = max(agent_pos[0], box_pos[0], goal_pos[0]), min(agent_pos[0], box_pos[0], goal_pos[0])
        max_n = int(max_x / 1.7)
        min_n = int(min_x / 1.7)
        count = 0
        for pos_obstacle in obstacles_pos:
            if abs(pos_obstacle[1] - 2.5) < 1e-3 and min_n < round(pos_obstacle[0] / 1.7) < max_n + 1:
                count += 1
        return count

    def smooth(arr, window=100):
        smoothed = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            smoothed[i] = np.mean(arr[max(i - window + 1, 0): i + 1])
        return smoothed

    blocked_doors = list(map(n_doors_blocked, initial_states))
    blocked_doors = np.asarray(blocked_doors)
    print(len(blocked_doors))
    reduction_masks = [(blocked_doors == i).astype(np.float32) for i in range(4)]
    reduction_percents = [smooth(mask, 1000) for mask in reduction_masks]
    for i in range(4):
        plt.plot(reduction_percents[i], label='%d blocked' % i)
    plt.legend()
    plt.show()
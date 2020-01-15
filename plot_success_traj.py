import sys, pandas, os, imageio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_item(log_file, label):
    data = pandas.read_csv(log_file, index_col=None, comment='#')
    return data[label].values

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python plot_success_traj.py [csv_name]')
        exit()
    fname = sys.argv[1]
    dones = get_item(fname, 'done')
    gripper_xs = get_item(fname, 'gripper_x')
    gripper_ys = get_item(fname, 'gripper_y')
    gripper_zs = get_item(fname, 'gripper_z')
    box_xs = get_item(fname, 'box_x')
    box_ys = get_item(fname, 'box_y')
    box_zs = get_item(fname, 'box_z')
    obstacle_xs = get_item(fname, 'obstacle_x')
    obstacle_ys = get_item(fname, 'obstacle_y')
    obstacle_zs = get_item(fname, 'obstacle_z')
    goals = []
    for i in range(5):
        goals.append(get_item(fname, 'goal_' + str(i)))
    goals = np.asarray(goals)
    goals = np.swapaxes(goals, 0, 1)
    end_points = np.where(dones > 0.5)[0]
    print('#episodes', len(end_points))
    for i in end_points:
        assert np.argmax(goals[i][3:]) == 0
        # print(goals[i])
    '''
    _print_end_points = np.random.choice(end_points[:len(end_points) // 100], size=20)
    _print_end_points2 = np.random.choice(end_points[len(end_points) // 100 * 99:], size=20)
    _print_end_points3 = np.random.choice(end_points[len(end_points) // 100 * 50: len(end_points) // 10 * 51], size=20)
    print('first percentile')
    for i in _print_end_points:
        print(i, goals[i])
    print('50 percentile')
    for i in _print_end_points2:
        print(i, goals[i])
    print('last percentile')
    for i in _print_end_points3:
        print(i, goals[i])
    '''
    # print(goals[:, 3:])

    ep_idx = 0
    step = 0
    has_switch = False
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(dones.shape[0]):
        ax.cla()
        ax.set_xlim(1.0, 1.6)
        ax.set_ylim(0.4, 1.1)
        # ax.set_zlim(0, 1.2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.scatter(gripper_xs[i], gripper_ys[i], c='tab:gray')
        ax.scatter(box_xs[i], box_ys[i], c='tab:blue')
        ax.scatter(obstacle_xs[i], obstacle_ys[i], c='tab:brown')
        if not has_switch and np.argmax(goals[i][3:]) == 1:
            print('episode %d switch step %d' % (ep_idx, step))
            print('restart box', box_xs[i], box_ys[i], 'subgoal', goals[i])
            has_switch = True
        if np.argmax(goals[i][3:]) == 0:
            marker = '*'
        else:
            marker = '^'
        ax.scatter(goals[i][0], goals[i][1], c='tab:red', marker=marker)
        ax.set_title('episode %d step %d' % (ep_idx, step))
        step += 1
        if dones[i] > 0.5:
            assert np.argmax(goals[i][3:]) == 0
            print('ultimate goal', goals[i])
            ep_idx += 1
            step = 0
            has_switch = False
        plt.savefig('tempimg' + str(i) + '.png')
        plt.pause(0.1)
    os.system('ffmpeg -r 2 -start_number 0 -i tempimg%d.png -c:v libx264 -pix_fmt yuv420p ' +
              os.path.join(os.path.dirname(fname), 'augment_data.mp4'))
    # images = []
    for i in range(dones.shape[0]):
    #     images.append(plt.imread('tempimg' + str(i) + '.png'))
        os.remove('tempimg' + str(i) + '.png')
    # imageio.mimsave('augment_data.gif', images)

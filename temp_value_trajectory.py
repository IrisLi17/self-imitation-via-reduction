import sys, os
import numpy as np
import matplotlib.pyplot as plt
from run_her import make_env
from baselines import HER_HACK
from gym.wrappers import FlattenDictWrapper

if __name__ == '__main__':
    model_path = sys.argv[1:]
    env = make_env(env_id='FetchPush-v1', seed=0, rank=0)
    env = FlattenDictWrapper(env, ['observation', 'achieved_goal', 'desired_goal'])
    model = HER_HACK.load(model_path[0])
    obs_buf = []
    obs = env.reset()
    done = False
    i = 0
    while not done:
        obs_buf.append(obs)
        img = env.render(mode='rgb_array')
        plt.imsave(os.path.join(os.path.dirname(model_path[0]), 'tempimg%d.png' % i), img)
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        i += 1

    for f in model_path:
        print(f)
        model.model.load_parameters(f)
        values = model.model.sess.run(model.model.step_ops[6],
                                      {model.model.observations_ph: np.asarray(obs_buf)})
        plt.plot(values, label=f)
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(model_path[0]), 'value_compare.png'))
    os.system('ffmpeg -r 5 -start_number 0 -i ' + os.path.dirname(
        model_path[0]) + '/tempimg%d.png -c:v libx264 -pix_fmt yuv420p ' +
              os.path.join(os.path.dirname(model_path[0]), 'FetchPush-v1.mp4'))
    for j in range(i):
        os.remove(os.path.join(os.path.dirname(model_path[0]), 'tempimg%d.png' % j))

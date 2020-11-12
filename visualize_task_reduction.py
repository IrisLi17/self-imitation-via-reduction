import sys, os
import numpy as np
from run_her import make_env, get_env_kwargs
from baselines import HER_HACK
from gym.wrappers import FlattenDictWrapper
from utils.parallel_subproc_vec_env2 import ParallelSubprocVecEnv as ParallelSubprocVecEnv2
from utils.subproc_vec_vae_env2 import ParallelVAESubprocVecEnv as SubprocVaeEnv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from baselines import SAC_augment
from sklearn.neighbors import KNeighborsRegressor
import utils.torch.pytorch_util as ptu
import pickle
from run_her import VAE_LOAD_PATH
import cv2

def decode_goal( vae_model,latents):
    latents = ptu.np_to_var(latents)
    latents = latents.view(-1, vae_model.representation_size)
    decoded = vae_model.decode(latents)
    return ptu.get_numpy(decoded)
def save_video(video_frames, filename, fps=1):
    assert fps == int(fps), fps
    import skvideo.io
    skvideo.io.vwrite(filename, video_frames, inputdict={'-r': str(int(fps))})

def plt_to_numpy(fig):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data
if __name__ == '__main__':
    env_id = sys.argv[1]
    reward_type = sys.argv[2]
    model_path = sys.argv[3]
    reward_threshold = float(sys.argv[4])
    env_kwargs = get_env_kwargs(env_id, random_ratio=0.0,reward_type=reward_type,reward_object=1,reward_threshold=reward_threshold)
    log_dir = 'logs/tmp'
    train_latent = np.load('sawyer_dataset_latents.npy')
    train_state = np.load('sawyer_dataset_states.npy')
    regressor = KNeighborsRegressor()
    regressor.fit(train_latent, train_state)

    vae_file = open(VAE_LOAD_PATH[env_id], 'rb')
    vae_model = pickle.load(vae_file)
    vae_model.cuda()
    import utils.torch.pytorch_util as ptu

    ptu.set_device(0)
    ptu.set_gpu_mode(True)
    print('regressor fit finished!')

    def make_thunk(rank):
        return lambda: make_env(env_id=env_id, seed=0, rank=rank, kwargs=env_kwargs)

    # env = ParallelSubprocVecEnv2([make_thunk(i) for i in range(1)])
    env = SubprocVaeEnv2([make_thunk(i) for i in range(1)],env_id=env_id,log_dir=log_dir,regressor=regressor)

    aug_env_id = env_id.split('-')[0] + 'Unlimit-' + env_id.split('-')[1]
    aug_env_kwargs = env_kwargs.copy()
    aug_env_kwargs['max_episode_steps'] = 100

    # def make_thunk_aug(rank):
    #     return lambda: FlattenDictWrapper(make_env(env_id=aug_env_id, seed=0, rank=rank, kwargs=aug_env_kwargs),
    #                                       ['observation', 'achieved_goal', 'desired_goal'])
    #
    # aug_env = ParallelSubprocVecEnv([make_thunk_aug(i) for i in range(1)])
    aug_env = make_env(aug_env_id, seed=0, rank=0, kwargs=aug_env_kwargs)
    # aug_env = FlattenDictWrapper(aug_env, ['observation', 'achieved_goal', 'desired_goal'])
    # goal_dim = aug_env.get_attr('goal')[0].shape[0]
    # obs_dim = aug_env.observation_space.shape[0] - 2 * goal_dim
    # noise_mag = aug_env.get_attr('size_obstacle')[0][1]
    # n_object = aug_env.get_attr('n_object')[0]

    goal_dim = aug_env.get_goal()['desired_goal'].shape[0]
    obs_dim = goal_dim
    # noise_mag = aug_env.size_obstacle[1]
    # n_object = aug_env.n_object
    model = HER_HACK.load(model_path, env=env,env_id=env_id)
    model.model.reward_type='dense'
    model.model.env_id = env_id
    model.model.goal_dim = goal_dim
    model.model.obs_dim = obs_dim
    model.model.aug_env = env
    model.model.debug_value1 = []
    model.model.debug_value2 = []
    model.model.normalize_value1=[]
    model.model.normalize_value2=[]
    model.model.value_prod=[]
    model.model.value_mean=[]
    # model.model.noise_mag = noise_mag
    # model.model.n_object = n_object
    success_flag=False
    episode = 0
    fig_buf = []
    fig_viewer_buf = []
    while episode < 5:
        success_flag=False
        episode +=1
        obs_buf, state_buf = [], []
        transition_buf = []
        img_buf = []
        img_viewer_buf = []
        # obs = env.env_method('reset')[0]
        obs = env.reset()
        initial_obs = (obs['image_observation'][0].reshape(3,vae_model.imsize,vae_model.imsize).transpose(1,2,0)*255).astype('uint8')
        initial_goal = (obs['image_desired_goal'][0].reshape(3,vae_model.imsize,vae_model.imsize).transpose(1,2,0)*255).astype('uint8')
        initial_goal_record = obs['image_desired_goal'][0]
        cv2.imwrite(os.path.join(os.path.dirname(model_path), 'initial_obs%d.png'%episode), initial_obs)

        cv2.imwrite(os.path.join(os.path.dirname(model_path), 'initial_goal%d.png'%episode), initial_goal)
        done = False
        fig1, ax1 = plt.subplots(1, 1, figsize=(16, 16))
        step = 0
        fig_buf = []
        fig_viewer_buf = []
        while not done:
            # tower_height = env.tower_height
            state = env.env_method('get_state')[0]
            # obs_buf.append(np.concatenate([obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']], axis=-1)[0])
            obs_img_obs = obs['image_observation'][0]
            obs_img_achieved_goal = obs['image_achieved_goal'][0]
            obs_img_desired_goal = obs['image_desired_goal'][0]
            obs_img = np.stack([obs_img_obs, obs_img_achieved_goal, obs_img_desired_goal])
            obs_latent = ptu.get_numpy(vae_model.encode(ptu.np_to_var(obs_img))[0])
            obs_latent = obs_latent.reshape(-1, vae_model.representation_size * 3)
            obs_state_pred = regressor.predict(obs_latent.reshape(-1, vae_model.representation_size))

            predict_state = obs_state_pred[0]
            predict_desired_goal = obs_state_pred[2]
            hand_pred_dist = round(np.linalg.norm(predict_state[:2] - predict_desired_goal[:2]), 3)
            puck_pred_dist = round(np.linalg.norm(predict_state[-2:] - predict_desired_goal[-2:]), 3)
            dist = np.linalg.norm(predict_state - predict_desired_goal)
            if step == 0:
                if reward_type =='dense':

                    reward = -dist
                else:
                    reward = -1.0 *(dist >= reward_threshold)
            obs_buf.append(obs_latent[0])
            state_buf.append(state)
            # tower_height_buf.append(tower_height)
            # img = env.env_method('render', ['rgb_array'])[0]
            img =  (obs['image_observation'][0].reshape(3,vae_model.imsize,vae_model.imsize).transpose(1,2,0)*255).astype('uint8')
            img_buf.append(img)

            img_viewer = env.env_method('get_image_plt')[0]
            img_viewer_buf.append(img_viewer_buf)
            obs_state = obs['state_observation'][0]
            state_goal = obs['state_desired_goal'][0]
            print('obs_state',obs_state)
            print('state_goal',state_goal)
            hand_dist = round(np.linalg.norm(obs_state[:2] - state_goal[:2]), 3)
            puck_dist = round(np.linalg.norm(obs_state[-2:] - state_goal[-2:]), 3)
            print('hand_dist',hand_dist)
            print('puck_dist',puck_dist)
            ax1.cla()
            ax1.imshow(img)
            ax1.set_title('frame %d' % step + 'reward' + str(reward) +
                          'puck_dist' + str(puck_dist) + 'hand_dist' + str(hand_dist) +
                          'puck_pred_dist' + str(puck_pred_dist) + 'hand_pred_dist' + str(hand_pred_dist))
            fig_buf.append(plt_to_numpy(fig1))
            # print('fig1',fig1)
            # plt.savefig(os.path.join(os.path.dirname(model_path), 'successimgviewer%d.png' % step))

            # plt.pause(0.2)
            ax1.cla()
            ax1.imshow(img_viewer)
            ax1.set_title('frame %d' % step + 'reward' + str(reward) +
                          'puck_dist' + str(puck_dist) + 'hand_dist' + str(hand_dist) +
                          'puck_pred_dist' + str(puck_pred_dist) + 'hand_pred_dist' + str(hand_pred_dist))
            fig_viewer_buf.append(plt_to_numpy(fig1))
            # plt.savefig(os.path.join(os.path.dirname(model_path), 'successimg%d.png' % step))

            # plt.pause(0.2)
            action, _ = model.predict(obs_latent)
            actions = np.repeat(action, 1, axis=0)
            obs, reward, done, info = env.step(actions)
            print('reward',reward)
            step +=1
            transition_buf.append((obs_buf[-1], action[0]))  # dummy items after obs

        if info[0]['is_success']:
            print('No need to do reduction')
            success_flag=True
            save_video(np.stack(fig_buf),os.path.join(os.path.dirname(model_path), 'test_time_viewersuccessr%d.mp4' % episode))
            save_video(np.stack(fig_viewer_buf),os.path.join(os.path.dirname(model_path), 'test_timesuccess%d.mp4' % episode))
            # os.system('ffmpeg -r 5 -start_number 0 -i ' + os.path.dirname(
            #     model_path) + '/successimgviewer%d.png -c:v libx264 -pix_fmt yuv420p ' +
            #           os.path.join(os.path.dirname(model_path), 'test_time_viewersuccessr%d.mp4' % episode))
            # os.system('ffmpeg -r 5 -start_number 0 -i ' + os.path.dirname(
            #     model_path) + '/successimg%d.png -c:v libx264 -pix_fmt yuv420p ' +
            #           os.path.join(os.path.dirname(model_path), 'test_timesuccess%d.mp4' % episode))

            # for i in range(step):
            #     os.remove(os.path.join(os.path.dirname(model_path), 'successimgviewer%d.png' % i))
            #     os.remove(os.path.join(os.path.dirname(model_path), 'successimg%d.png' % i))
            # exit()
        else:
            save_video(np.stack(fig_buf),
                       os.path.join(os.path.dirname(model_path), 'test_time_viewerfail%d.mp4' % episode))
            save_video(np.stack(fig_viewer_buf),
                       os.path.join(os.path.dirname(model_path), 'test_timefail%d.mp4' % episode))
            # for i in range(step):
            #     os.remove(os.path.join(os.path.dirname(model_path), 'successimgviewer%d.png' % i))
            #     os.remove(os.path.join(os.path.dirname(model_path), 'successimg%d.png' % i))
            SAC_augment.reward_type='dense'
            SAC_augment.aug_env = env
            # print(SAC_augment.aug_env)
            # print(SAC_augment.aug_env.vae_model.representation_size)

            fig_buf = []
            fig_viewer_buf = []
            restart_step, subgoal = SAC_augment.select_subgoal_cem(model.model, transition_buf, 1)

            # restart_step, subgoal = model.model.select_subgoal(transition_buf, 1, tower_height_buf if 'FetchStack' in env_id else None)
            print(restart_step, subgoal)
            img_buf = img_buf[:restart_step[0]]
            aug_env.reset()
            aug_env.set_state(state_buf[restart_step[0]])
            # aug_env.set_task_mode(0)
            ## set_goal
            img_goal = decode_goal(vae_model,subgoal[0])
            state_goal = regressor.predict(subgoal[0].reshape(-1,env.vae_model.representation_size))
            ## logging the subgoal image for debugging
            vae_decoded_img_subgoal = (img_goal[0].reshape(3,vae_model.imsize,vae_model.imsize).transpose(1,2,0)*255).astype('uint8')
            cv2.imwrite(os.path.join(os.path.dirname(model_path), 'vae_decoded_subgoal%d.png'%episode),vae_decoded_img_subgoal)

            goal_dict = dict(image_desired_goal= img_goal[0],state_desired_goal = state_goal[0])
            aug_env.set_goal(goal_dict)
            # aug_env.set_goal(subgoal[0])
            step_so_far = restart_step[0]
            done = False
            obs = aug_env.get_obs()
            obs_img_obs = obs['image_observation']
            obs_img_achieved_goal = obs['image_achieved_goal']
            obs_img_desired_goal = obs['image_desired_goal']

            ## logging the subgoal image for debugging
            # obs_img_desired_subgoal_transform = (obs_img_desired_goal.reshape(3,vae_model.imsize,vae_model.imsize).transpose(1,2,0)*255).astype('uint8')
            # cv2.imwrite(os.path.join(os.path.dirname(model_path), 'obs_desired_subgoal%d.png'%episode), obs_img_desired_subgoal_transform)

            obs_img = np.stack([obs_img_obs, obs_img_achieved_goal, obs_img_desired_goal])
            obs_latent = ptu.get_numpy(vae_model.encode(ptu.np_to_var(obs_img))[0])
            obs_latent = obs_latent.reshape(-1, vae_model.representation_size * 3)
            # obs_latent = np.concatenate([obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']])
            fig, ax = plt.subplots(1, 1,figsize=(16,16))
            while not done:
                if step_so_far > env_kwargs['max_episode_steps']:
                    break
                action, _ = model.predict(obs_latent)
                obs, reward, done, info = aug_env.step(action[0])
                obs_state = obs['state_observation']
                state_goal = obs['state_desired_goal']
                hand_dist = round(np.linalg.norm(obs_state[:2] - state_goal[:2]), 3)
                puck_dist = round(np.linalg.norm(obs_state[-2:] - state_goal[-2:]), 3)
                obs_img_obs = obs['image_observation']
                obs_img_achieved_goal = obs['image_achieved_goal']
                obs_img_desired_goal = obs['image_desired_goal']
                obs_img = np.stack([obs_img_obs, obs_img_achieved_goal, obs_img_desired_goal])
                obs_latent = ptu.get_numpy(vae_model.encode(ptu.np_to_var(obs_img))[0])
                obs_latent = obs_latent.reshape(-1, vae_model.representation_size * 3)
                obs_state = regressor.predict(obs_latent.reshape(-1,vae_model.representation_size))
                predict_state = obs_state[0]
                predict_desired_goal = obs_state[2]
                hand_pred_dist = round(np.linalg.norm(predict_state[:2] - state_goal[:2]), 3)
                puck_pred_dist = round(np.linalg.norm(predict_state[-2:] - state_goal[-2:]), 3)
                dist = np.linalg.norm(predict_state-predict_desired_goal)
                reward = -dist
                done = done or (dist < reward_threshold)
                print('reward',reward)
                # img = aug_env.render(mode='rgb_array')
                img =  (obs['image_observation'].reshape(3,vae_model.imsize,vae_model.imsize).transpose(1,2,0)*255).astype('uint8')
                img_viewer = aug_env.get_image_plt()
                ax.cla()
                ax.imshow(img)
                ax.set_title('frame %d' % step_so_far + 'sub_goal ' + 'reward' + str(reward) +
                             'puck_dist' + str(puck_dist) + 'hand_dist' + str(hand_dist) +
                             'puck_pred_dist' + str(puck_pred_dist) + 'hand_pred_dist' + str(hand_pred_dist))
                # plt.savefig(os.path.join(os.path.dirname(model_path), 'tempimgviewer%d.png' % step_so_far))
                fig_buf.append(plt_to_numpy(fig))
                # plt.pause(0.2)
                ax.cla()
                ax.imshow(img_viewer)
                ax.set_title('frame %d' % step_so_far + 'sub_goal ' + 'reward' + str(reward) +
                             'puck_dist' + str(puck_dist) + 'hand_dist' + str(hand_dist) +
                             'puck_pred_dist' + str(puck_pred_dist) + 'hand_pred_dist' + str(hand_pred_dist))
                # plt.savefig(os.path.join(os.path.dirname(model_path), 'tempimg%d.png' % step_so_far))
                fig_viewer_buf.append(plt_to_numpy(fig))
                # plt.pause(0.2)
                img_buf.append(img)
                img_viewer_buf.append(img)
                step_so_far += 1
            # aug_env.set_task_mode(1)
            # final_goal = obs_buf[0][-goal_dim:]
            # img_goal = decode_goal(vae_model, final_goal)
            img_goal = initial_goal_record
            ## logging the subgoal image for debugging
            # vae_decoded_img_finalgoal= (
            #             img_goal[0].reshape(3, vae_model.imsize, vae_model.imsize).
            #             transpose(1, 2, 0) * 255).astype( 'uint8')
            # cv2.imwrite(os.path.join(os.path.dirname(model_path), 'vae_decoded_img_finalgoal.png'),
            #             vae_decoded_img_finalgoal)

            # final_goal = final_goal.reshape(-1,env.vae_model.representation_size)
            final_goal = ptu.get_numpy(vae_model.encode(ptu.np_to_var(img_goal))[0])
            state_goal = regressor.predict(final_goal)
            goal_dict = dict(image_desired_goal=img_goal, state_desired_goal=state_goal[0])
            aug_env.set_goal(goal_dict)
            # aug_env.set_goal(obs_buf[0][-goal_dim:])
            print('Switch to ultimate goal', obs_buf[0][-goal_dim:])
            done = False
            while not done:
                if step_so_far > env_kwargs['max_episode_steps']:
                    break
                action, _ = model.predict(obs_latent)
                obs, reward, done, info = aug_env.step(action[0])
                obs_state = obs['state_observation']
                state_goal = obs['state_desired_goal']
                hand_dist = round( np.linalg.norm(obs_state[:2]-state_goal[:2]),3)
                puck_dist = round(np.linalg.norm(obs_state[-2:]-state_goal[-2:]),3)
                obs_img_obs = obs['image_observation']
                obs_img_achieved_goal = obs['image_achieved_goal']
                obs_img_desired_goal = obs['image_desired_goal']
                ## logging the subgoal image for debugging
                # obs_img_desired_finalgoal_transform = (
                #             obs_img_desired_goal.reshape(3, vae_model.imsize, vae_model.imsize).transpose(1, 2,
                #                                                                                           0) * 255).astype(
                #     'uint8')
                # cv2.imwrite(os.path.join(os.path.dirname(model_path), 'obs_desired_finalgoal%d.png'%episode),
                #             obs_img_desired_finalgoal_transform)
                obs_img = np.stack([obs_img_obs, obs_img_achieved_goal, obs_img_desired_goal])
                obs_latent = ptu.get_numpy(vae_model.encode(ptu.np_to_var(obs_img))[0])
                obs_latent = obs_latent.reshape(-1, vae_model.representation_size * 3)
                obs_state = regressor.predict(obs_latent.reshape(-1, vae_model.representation_size))
                predict_state = obs_state[0]
                predict_desired_goal = obs_state[2]
                hand_pred_dist = round(np.linalg.norm(predict_state[:2] - state_goal[:2]), 3)
                puck_pred_dist = round(np.linalg.norm(predict_state[-2:] - state_goal[-2:]), 3)
                dist = np.linalg.norm(predict_state - predict_desired_goal)
                reward = -dist
                done = done or (dist < reward_threshold)
                print('reward', reward)
                img = (obs['image_observation'].reshape(3, vae_model.imsize, vae_model.imsize).
                       transpose(1, 2, 0) * 255).astype('uint8')
                img_viewer = aug_env.get_image_plt()
                print('show_image')
                ax.cla()
                ax.imshow(img)
                ax.set_title('frame %d' % step_so_far+'ultimate_goal '+'reward'+str(reward)+
                             'puck_dist'+str(puck_dist)+'hand_dist'+str(hand_dist)+
                             'puck_pred_dist'+str(puck_pred_dist)+'hand_pred_dist'+str(hand_pred_dist))

                # plt.savefig(os.path.join(os.path.dirname(model_path), 'tempimgviewer%d.png' % step_so_far))
                fig_buf.append(plt_to_numpy(fig))
                # plt.pause(0.2)
                ax.cla()
                ax.imshow(img_viewer)
                ax.set_title('frame %d' % step_so_far + 'ultimate_goal ' + 'reward' + str(reward) +
                             'puck_dist' + str(puck_dist) + 'hand_dist' + str(hand_dist) +
                             'puck_pred_dist' + str(puck_pred_dist) + 'hand_pred_dist' + str(hand_pred_dist))

                # plt.savefig(os.path.join(os.path.dirname(model_path), 'tempimg%d.png' % step_so_far))
                fig_viewer_buf.append(plt_to_numpy(fig))
                # plt.pause(0.2)
                img_buf.append(img)
                img_viewer_buf.append(img_viewer)
                step_so_far += 1
        # for i in range(len(img_buf)):
        #     plt.imsave(os.path.join(os.path.dirname(model_path), 'tempimgviewer%d.png' % i), img_buf[i])
        #     plt.imsave(os.path.join(os.path.dirname(model_path),'tempimg%d.png'%i),img_viewer_buf[i])
        if not success_flag:
            save_video(np.stack(fig_buf),
                       os.path.join(os.path.dirname(model_path), 'test_time_viewer%d.mp4' % episode))
            save_video(np.stack(fig_viewer_buf),
                       os.path.join(os.path.dirname(model_path), 'test_time%d.mp4' % episode))
    print('saving finished')
            # os.system('ffmpeg -r 5 -start_number 0 -i ' + os.path.dirname(
            #     model_path) + '/tempimgviewer%d.png -c:v libx264 -pix_fmt yuv420p ' +
            #           os.path.join(os.path.dirname(model_path), 'test_time_viewer%d.mp4'%episode))
            # os.system('ffmpeg -r 5 -start_number 0 -i ' + os.path.dirname(
            #     model_path) + '/tempimg%d.png -c:v libx264 -pix_fmt yuv420p ' +
            #           os.path.join(os.path.dirname(model_path), 'test_time%d.mp4'%episode))
            #
            # for i in range(len(img_buf)):
            #     os.remove(os.path.join(os.path.dirname(model_path), 'tempimgviewer%d.png' % i))
            #     os.remove(os.path.join(os.path.dirname(model_path), 'tempimg%d.png' % i))


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
from mpl_toolkits.mplot3d import Axes3D
import csv
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

# def get_value_and_pos(model,regressor,obs):
#     value = model.value(obs)
#     obs_reshape = obs.reshape(-1,16)
#     obs_states = regressor.predict(obs_reshape)
#     state = obs_states[0]
#     goal = obs_states[2]
#     hand_dist =np.linalg.norm( state[:2]-goal[:2])
#     puck_dist = np.linalg.norm(state[-2:]-goal[-2:])
#     return value,hand_dist,puck_dist

if __name__ == '__main__':
    env_id = sys.argv[1]
    reward_type = sys.argv[2]
    model_path = sys.argv[3]
    reward_threshold = float(sys.argv[4])
    n_trials = int(sys.argv[5])
    n_interval = int(sys.argv[6])
    env_kwargs = get_env_kwargs(env_id, random_ratio=0.0,reward_type=reward_type,reward_object=1,reward_threshold=reward_threshold)
    log_dir = 'logs/tmp'
    file_handler = open(os.path.join(os.path.dirname(model_path),'plot.csv'), "wt")
    # file_handler.write('#%s\n' % json.dumps({"t_start": self.t_start}))
    logger = csv.DictWriter(file_handler,
                                         fieldnames=('episode', 'restart_step', 'final_step') + ('subgoal_1','subgoal_2'))
    logger.writeheader()
    file_handler.flush()

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
    while episode < n_trials:
        success_flag=False
        episode +=1
        obs_buf, state_buf = [], []
        # transition_buf = []
        img_buf = []
        img_viewer_buf = []
        # obs = env.env_method('reset')[0]
        obs = env.reset()
        initial_obs_origin = obs
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
        # while not done:
        #     # tower_height = env.tower_height
        #     state = env.env_method('get_state')[0]
        #     # obs_buf.append(np.concatenate([obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']], axis=-1)[0])
        #     obs_img_obs = obs['image_observation'][0]
        #     obs_img_achieved_goal = obs['image_achieved_goal'][0]
        #     obs_img_desired_goal = obs['image_desired_goal'][0]
        #     obs_img = np.stack([obs_img_obs, obs_img_achieved_goal, obs_img_desired_goal])
        #     obs_latent = ptu.get_numpy(vae_model.encode(ptu.np_to_var(obs_img))[0])
        #     obs_latent = obs_latent.reshape(-1, vae_model.representation_size * 3)
        #     obs_state_pred = regressor.predict(obs_latent.reshape(-1, vae_model.representation_size))
        #
        #     predict_state = obs_state_pred[0]
        #     predict_desired_goal = obs_state_pred[2]
        #     hand_pred_dist = round(np.linalg.norm(predict_state[:2] - predict_desired_goal[:2]), 3)
        #     puck_pred_dist = round(np.linalg.norm(predict_state[-2:] - predict_desired_goal[-2:]), 3)
        #     dist = np.linalg.norm(predict_state - predict_desired_goal)
        #     if step == 0:
        #         if reward_type =='dense':
        #
        #             reward = -dist
        #         else:
        #             reward = -1.0 *(dist >= reward_threshold)
        #     obs_buf.append(obs_latent[0])
        #     state_buf.append(state)
        #     # tower_height_buf.append(tower_height)
        #     # img = env.env_method('render', ['rgb_array'])[0]
        #     img =  (obs['image_observation'][0].reshape(3,vae_model.imsize,vae_model.imsize).transpose(1,2,0)*255).astype('uint8')
        #     img_buf.append(img)
        #
        #     img_viewer = env.env_method('get_image_plt')[0]
        #     img_viewer_buf.append(img_viewer_buf)
        #     obs_state = obs['state_observation'][0]
        #     state_goal = obs['state_desired_goal'][0]
        #     print('obs_state',obs_state)
        #     print('state_goal',state_goal)
        #     hand_dist = round(np.linalg.norm(obs_state[:2] - state_goal[:2]), 3)
        #     puck_dist = round(np.linalg.norm(obs_state[-2:] - state_goal[-2:]), 3)
        #     print('hand_dist',hand_dist)
        #     print('puck_dist',puck_dist)
        #     ax1.cla()
        #     ax1.imshow(img)
        #     ax1.set_title('frame %d' % step + 'reward' + str(reward) +
        #                   'puck_dist' + str(puck_dist) + 'hand_dist' + str(hand_dist) +
        #                   'puck_pred_dist' + str(puck_pred_dist) + 'hand_pred_dist' + str(hand_pred_dist))
        #     fig_buf.append(plt_to_numpy(fig1))
        #     # print('fig1',fig1)
        #     # plt.savefig(os.path.join(os.path.dirname(model_path), 'successimgviewer%d.png' % step))
        #
        #     # plt.pause(0.2)
        #     ax1.cla()
        #     ax1.imshow(img_viewer)
        #     ax1.set_title('frame %d' % step + 'reward' + str(reward) +
        #                   'puck_dist' + str(puck_dist) + 'hand_dist' + str(hand_dist) +
        #                   'puck_pred_dist' + str(puck_pred_dist) + 'hand_pred_dist' + str(hand_pred_dist))
        #     fig_viewer_buf.append(plt_to_numpy(fig1))
        #     # plt.savefig(os.path.join(os.path.dirname(model_path), 'successimg%d.png' % step))
        #
        #     # plt.pause(0.2)
        #     action, _ = model.predict(obs_latent)
        #     actions = np.repeat(action, 1, axis=0)
        #     obs, reward, done, info = env.step(actions)
        #     print('reward',reward)
        #     step +=1
        #     transition_buf.append((obs_buf[-1], action[0]))  # dummy items after obs
        #
        # if info[0]['is_success']:
        #     print('No need to do reduction')
        #     success_flag=True
        #     save_video(np.stack(fig_buf),os.path.join(os.path.dirname(model_path), 'test_time_viewersuccessr%d.mp4' % episode))
        #     save_video(np.stack(fig_viewer_buf),os.path.join(os.path.dirname(model_path), 'test_timesuccess%d.mp4' % episode))


        ## get the init step and goal from first observation from reset
        image_keys = ['image_observation','image_achieved_goal','image_desired_goal']
        final_goal = initial_obs_origin['image_desired_goal'][0]
        init_step = initial_obs_origin['image_observation'][0]
        init_state = initial_obs_origin['state_observation'][0]
        init_state_goal = initial_obs_origin['state_desired_goal'][0]
        init_hand_dist = np.linalg.norm(init_state[:2]-init_state_goal[:2])
        init_puck_dist = np.linalg.norm(init_state[-2:]-init_state_goal[-2:])
        transition_buf = []
        obs_img = np.concatenate([initial_obs_origin[key] for key in image_keys])
        print('obs_img',obs_img.shape)
        obs_latent = ptu.get_numpy(vae_model.encode(ptu.np_to_var(obs_img))[0])
        obs_latent = obs_latent.reshape(-1,vae_model.representation_size)
        print('obs_latent_shape',obs_latent.shape)
        transition_buf.append((obs_latent.reshape(vae_model.representation_size*3,),0))

        # select subgoal with cem
        SAC_augment.reward_type = 'dense'
        SAC_augment.aug_env = env
        restart_step, subgoal_min,debug_dict_min = SAC_augment.select_subgoal_cem(model.model, transition_buf, 2,'min')
        _, subgoal_mean,debug_dict_mean = SAC_augment.select_subgoal_cem(model.model, transition_buf, 2,'mean')

        img_goal_min = decode_goal(vae_model, subgoal_min)
        img_goal_mean = decode_goal(vae_model,subgoal_mean)
        for i in range(2):
            min_img_subgoal = (
                        img_goal_min[i].reshape(3, vae_model.imsize, vae_model.imsize).transpose(1, 2, 0) * 255).astype(
                'uint8')
            cv2.imwrite(os.path.join(os.path.dirname(model_path), 'min_subgoal_'+str(i)+'episode_'+str(episode)+'.png' ),
                        min_img_subgoal)
            mean_img_subgoal = (
                        img_goal_mean[i].reshape(3, vae_model.imsize, vae_model.imsize).transpose(1, 2, 0) * 255).astype(
                'uint8')
            cv2.imwrite(os.path.join(os.path.dirname(model_path), 'mean_subgoal_'+str(i)+'episode_'+str(episode)+'.png' ),
                        mean_img_subgoal)

        # selected_subgoal_hand_dist_to_init,selected_subgoal_hand_dist_to_goal,\
        # selected_subgoal_puck_dist_to_init,selected_subgoal_puck_dist_to_goal,\
        #     value1,value2,value_mean= [np.zeros((2))]*8
        ## TODO DEBUG purpose
        debug_mu_min = debug_dict_min['debug_mu']
        debug_chosen_value_min = debug_dict_min['debug_chosen_value']
        print('mu_shape',np.stack(debug_mu_min).shape)
        img_mu_min_decode_goal = decode_goal(vae_model,np.stack(debug_mu_min))
        debug_mu_mean = debug_dict_mean['debug_mu']
        debug_chosen_value_mean = debug_dict_mean['debug_chosen_value']
        print('mu_shape', np.stack(debug_mu_mean).shape)
        img_mu_mean_decode_goal = decode_goal(vae_model, np.stack(debug_mu_mean))
        for i in range(img_mu_min_decode_goal.shape[0]):
            vae_mu_min_decoded_img_subgoal = (
                    img_mu_min_decode_goal[i].reshape(3, vae_model.imsize, vae_model.imsize).transpose(1, 2, 0) * 255).astype(
                'uint8')
            cv2.imwrite(os.path.join(os.path.dirname(model_path),
                                     'vae_mu_min_subgoal_' + str(i) + 'value2_'+ str(int(round(np.mean(debug_chosen_value_min[i]),3)*1000)) +'episode_' + str(episode) + '.png'),
                        vae_mu_min_decoded_img_subgoal)
            vae_mu_mean_decoded_img_subgoal = (
                    img_mu_mean_decode_goal[i].reshape(3, vae_model.imsize, vae_model.imsize).transpose(1, 2, 0) * 255).astype(
                'uint8')
            cv2.imwrite(os.path.join(os.path.dirname(model_path),
                                     'vae_mu_mean_subgoal_' + str(i) + 'value2_'+ str(int(round(np.mean(debug_chosen_value_mean[i]),3)*1000)) +'episode_' + str(episode) + '.png'),
                        vae_mu_mean_decoded_img_subgoal)
       #  init_step_latent = obs_latent[0]
       #  final_goal_latent = obs_latent[2]
       #  num_subgoal = subgoal.shape[0]
       #  print('num_subgoal',num_subgoal)
       #  state = regressor.predict(subgoal)
       #  selected_subgoal_hand_dist_to_init = np.linalg.norm(init_state[:2] - state[:,:2],axis=1)
       #  print('selected_subgoal_state',state[:,:2])
       #  # exit()
       #  selected_subgoal_hand_dist_to_goal = np.linalg.norm(init_state_goal[:2] - state[:,:2],axis=1)
       #  selected_subgoal_puck_dist_to_init = np.linalg.norm(init_state[-2:] - state[:,-2:],axis=1)
       #  selected_subgoal_puck_dist_to_goal = np.linalg.norm(init_state_goal[-2:] - state[:,-2:],axis=1)
       #  selected_subgoal_dist_to_goal = np.linalg.norm(init_state_goal-state,axis=1)
       #  selected_subgoal_dist_to_init = np.linalg.norm(init_state-state,axis=1)
       #  obs_1 = np.repeat(obs_latent.reshape(-1,vae_model.representation_size*3), num_subgoal, axis=0)
       #  obs_1[:,-vae_model.representation_size:] = subgoal
       #  obs_2 = np.repeat(obs_latent.reshape(-1,vae_model.representation_size*3), num_subgoal, axis=0)
       #  obs_2[:,:vae_model.representation_size] = subgoal
       #  obs_2[:,vae_model.representation_size:vae_model.representation_size * 2] = subgoal
       #  value1 = model.model.sess.run(model.model.step_ops[6],
       #                                     {model.model.observations_ph: obs_1})
       #  # value1_list = model.value(obs_1)
       #  # value2_list = model.value(obs_2)
       #  value2 = model.model.sess.run(model.model.step_ops[6],
       #                                     {model.model.observations_ph: obs_2})
       #  value_mean = (value1+value2)/2
       #
       #  # select 21*21 point in the state space as the subgoal
       #  hand_space = aug_env.wrapped_env.wrapped_env.hand_space
       #
       #  pos_x,pos_y = np.meshgrid(np.linspace(hand_space.low[0],hand_space.high[0],n_interval),
       #                            np.linspace(hand_space.low[1],hand_space.high[1],n_interval))
       #  # set the hand_pos to be over the whole space and the puck dist should be out of 0.055
       #  # hand_pos_xy = np.concatenate([pos_x.reshape(-1,1),pos_y.reshape(-1,1)],axis=1)
       #  # puck_pos_xy = np.repeat(init_state_goal[-2:].reshape(1,2),n_interval**2,axis=0)
       #  flag=True
       #  random_pos=0
       #  while flag:
       #      random_pos = np.random.uniform(low=[-0.2,0.5],high=[0.2,0.7])
       #      flag = np.linalg.norm(random_pos-init_state_goal[:2]) <= 0.055
       #      print('random_pos',random_pos)
       #  print('hand_random_pos_to_goal',np.linalg.norm(random_pos-init_state_goal[:2]))
       #  obs = aug_env.set_state_xypos(np.concatenate([random_pos,init_state_goal[-2:]]))
       #  obs_random=(obs['image_observation'].reshape(3, vae_model.imsize, vae_model.imsize).transpose(1, 2, 0) * 255).astype('uint8')
       #  cv2.imwrite( os.path.join(os.path.dirname(model_path), 'obs_random%d.png'%episode), obs_random)
       #
       #  hand_pos_xy = np.repeat(random_pos.reshape(1,2),n_interval**2,axis=0)
       #  puck_pos_xy = np.concatenate([pos_x.reshape(-1,1),pos_y.reshape(-1,1)],axis=1)
       #  index = np.where(np.linalg.norm(hand_pos_xy-puck_pos_xy,axis=1)>0.055)
       #  pos_xy = np.concatenate([hand_pos_xy,puck_pos_xy],axis=1)[index]
       #  # pos_xy = np.conatenate([])
       #  # hand_pos_repeat_1 = np.repeat(hand_pos_xy,n_interval**2,axis=0)
       #  # hand_pos_repeat_2 = np.tile(hand_pos_xy.reshape(2*(n_interval**2)),n_interval**2).reshape(n_interval**4,2)
       #  # index = np.where(np.linalg.norm(hand_pos_repeat_1-hand_pos_repeat_2,axis=1)>0.055)
       #  # hand_puck_pos_list = np.concatenate([hand_pos_repeat_1,hand_pos_repeat_2],axis=1)
       #  # hand_puck_pos_filter = hand_puck_pos_list[index]
       #  # pos_xy = hand_puck_pos_filter
       #  print('pos_xy',pos_xy)
       #  # print('pos_list',hand_puck_pos_list)
       #  # hand_pos_bool = (hand_pos_xy[:,0] + 0.056) <= hand_space.high[0]
       #  # puck_pos_xy = hand_pos_xy.copy()
       #  # puck_pos_xy[:,0] += (hand_pos_bool * 0.056 - np.invert(hand_pos_bool)*0.056)
       #  # pos_xy = np.concatenate([hand_pos_xy,puck_pos_xy],axis=1)
       #  img_subgoal_buf = []
       #  for i in range(pos_xy.shape[0]):
       #      obs = aug_env.set_state_xypos(pos_xy[i])
       #      if not isinstance(obs,dict):
       #          print('wrong pos setting',pos_xy[i])
       #          exit()
       #
       #      img_subgoal_buf.append(obs['image_observation'])
       #  img_subgoal_np = np.stack(img_subgoal_buf)
       #
       #  subgoal_latents = ptu.get_numpy(vae_model.encode(ptu.np_to_var(img_subgoal_np))[0])
       #
       #  subgoal_states = regressor.predict(subgoal_latents)
       #  subgoal_states = np.clip(subgoal_states,[hand_space.low[0],hand_space.low[1],hand_space.low[0],hand_space.low[1]],
       #                           [hand_space.high[0],hand_space.high[1],hand_space.high[0],hand_space.high[1]])
       #  print('subgoal_states',subgoal_states)
       #  diff = np.linalg.norm(subgoal_states-pos_xy,axis=1)
       #  print('max_diff',np.max(diff),'min_diff',np.min(diff),'mean_diff',np.mean(diff))
       #  # diff_mean = np.mean(diff)
       #  # index_1 = np.where(diff > diff_mean)
       #  # index_above = np.where(diff > 0.10)
       #  # print('num above mean',len(index_1[0]),'num_above_0.10',len(index_above[0]))
       #  # print('index_above mean',index_1,'index_above_0.10',index_above)
       #  # subgoal_states_set = subgoal_states[index_above]
       #  # index_truth= np.where(np.linalg.norm(subgoal_states_set[:,:2]-subgoal_states_set[:,-2:],axis=1)>0.055)
       #  # subgoal_states_set = subgoal_states_set[index_truth]
       #  # index_truth_filter = index_above[0][index_truth[0]]
       #  # print('index_truth_filter',index_truth_filter)
       #  # subgoal_states_set = subgoal_states[index_truth_filter]
       #  # img_subgoal_truth_buf = []
       #  # for i in range(subgoal_states_set.shape[0]):
       #  #     obs = aug_env.set_state_xypos(subgoal_states_set[i])
       #  #     if not isinstance(obs, dict):
       #  #         print('wrong pos setting', subgoal_states_set[i])
       #  #         exit()
       #  #
       #  #     img_subgoal_truth_buf.append(obs['image_observation'])
       #  # for i in range(len(index_above[0])):
       #  #     # print(index_1)
       #  #     image_transform = (
       #  #             img_subgoal_buf[index_above[0][i]].reshape(3, vae_model.imsize, vae_model.imsize).transpose(1, 2,
       #  #                                                                                               0) * 255).astype(
       #  #         'uint8')
       #  #     cv2.imwrite(os.path.join(os.path.dirname(model_path),
       #  #                              'linear_groundtruth_subgoal_' +str(index_above[0][i]) + 'episode_' + str(episode) + '.png'),
       #  #                 image_transform)
       #  # for i in range(len(img_subgoal_truth_buf)):
       #  # #     if index_1[0][i] in index_truth_filter[0]:
       #  #         image_transform_truth = (
       #  #                 img_subgoal_truth_buf[i].reshape(3, vae_model.imsize, vae_model.imsize).transpose(1, 2,
       #  #                                                                                                         0) * 255).astype(
       #  #             'uint8')
       #  #         cv2.imwrite(os.path.join(os.path.dirname(model_path),
       #  #                                  'linear_abnormal_subgoal_' + str(index_above[0][index_truth[0][i]]) + 'episode_' + str(episode) + '.png'),
       #  #                     image_transform_truth)
       #  # index_abnormal=np.where(abs(subgoal_states-pos_xy)>0.06)
       #  # print('index_abnormal',index_abnormal[0].shape,index_abnormal)
       #  # hand_states = subgoal_states[:,:2]
       #  subgoal_hand_dist_to_init= np.linalg.norm(init_state[:2]-subgoal_states[:,:2],axis=1)
       #  subgoal_hand_dist_to_init_true = np.linalg.norm(init_state[:2]-pos_xy[:,:2],axis=1)
       #  subgoal_hand_dist_to_goal= np.linalg.norm(init_state_goal[:2]-subgoal_states[:,:2],axis=1)
       #  subgoal_hand_dist_to_goal_true = np.linalg.norm(init_state_goal[:2]-pos_xy[:,:2],axis=1)
       #  subgoal_dist_to_goal = np.linalg.norm(init_state_goal-subgoal_states,axis=1)
       #  subgoal_dist_to_goal_true = np.linalg.norm(init_state_goal-pos_xy,axis=1)
       #
       #  subgoal_puck_dist_to_init = np.linalg.norm(init_state[-2:] - subgoal_states[:,-2:],axis=1)
       #  subgoal_puck_dist_to_init_true = np.linalg.norm(init_state[-2:] - pos_xy[:,-2:],axis=1)
       #  subgoal_puck_dist_to_goal= np.linalg.norm(init_state_goal[-2:] - subgoal_states[:,-2:],axis=1)
       #  subgoal_puck_dist_to_goal_true = np.linalg.norm(init_state_goal[-2:] - pos_xy[:,-2:],axis=1)
       #  subgoal_dist_to_init = np.linalg.norm(init_state-subgoal_states,axis=1)
       #  subgoal_dist_to_init_true = np.linalg.norm(init_state-pos_xy,axis=1)
       #
       #  obs_1 = np.repeat(obs_latent.reshape(-1,vae_model.representation_size*3),img_subgoal_np.shape[0],axis=0)
       #  obs_1[:,-vae_model.representation_size:]= subgoal_latents
       #  obs_2 =np.repeat(obs_latent.reshape(-1,vae_model.representation_size*3),img_subgoal_np.shape[0],axis=0)
       #
       #  obs_2[:,:vae_model.representation_size] = subgoal_latents
       #  obs_2[:,vae_model.representation_size:vae_model.representation_size*2]=subgoal_latents
       #  value1_list = model.model.sess.run(model.model.step_ops[6],
       #                                     {model.model.observations_ph: obs_1})
       #  # value1_list = model.value(obs_1)
       #  # value2_list = model.value(obs_2)
       #  value2_list = model.model.sess.run(model.model.step_ops[6],
       #                                     {model.model.observations_ph: obs_2})
       #
       #  value_mean_list= (value1_list+ value2_list)/2
       #  print('value_mean',value_mean_list.shape)
       #  # obs_latent = obs_latent.reshape(-1, vae_model.representation_size * 3)
       #  print('subgoal_state',pos_xy[:,-2:])
       #  print('subgoal_state_diff',init_state_goal[-2:]-pos_xy[:,-2:])
       #  print('ground_truth_shape',subgoal_hand_dist_to_goal_true.shape)
       #  print('subgoal_puck_dist_to_goal_true',subgoal_puck_dist_to_goal_true)
       #  print('init_state_goal',init_state_goal,init_state_goal[-2:])
       #  # obs_state_pred = regressor.predict(obs_latent.reshape(-1, vae_model.representation_size))
       #  # log the episode info
       #  ep_info = {'episode':episode,'restart_step':init_state,'final_step':init_state_goal,'subgoal_1':subgoal_states[0],'subgoal_2':subgoal_states[1]}
       #  logger.writerow(ep_info)
       #  file_handler.flush()
       #  ## draw the plot
       #  # fig, ax = plt.subplots(1, 3, figsize=(10, 5),projection='3d')
       #  matplotlib.rcParams['legend.fontsize'] = 10
       #  ax = []
       #  fig = []
       #  for i in range(12):
       #      fig.append(plt.figure())
       #      ax.append(Axes3D(fig[i]))
       #  # ax = []
       #  # ax.append( fig.add_subplot(131))
       #  # ax.append( fig.add_subplot(132))
       #  # ax.append( fig.add_subplot(133))
       #  ## draw the value with dist_to_goal
       #  # ax.cla()
       #  print('subgoal_hand_dist_to_goal',subgoal_hand_dist_to_goal.shape)
       #  # figure = plt.figure(0)
       #  # fig.append(figure)
       #  # ax.append(Axes3D(figure))
       #  ax[0].scatter(subgoal_puck_dist_to_goal,subgoal_hand_dist_to_goal,value_mean_list,c='tab:orange',s=4.0,marker='o',label='value_mean')
       #  ax[0].scatter(subgoal_puck_dist_to_goal_true,subgoal_hand_dist_to_goal_true,value_mean_list,s=4.0,c='tab:red',marker='*',label='value_mean_truth')
       #
       #  ax[0].scatter(selected_subgoal_puck_dist_to_goal,selected_subgoal_hand_dist_to_goal,value_mean,s=20.0,c='tab:green',marker='^',label='value_mean_subgoal')
       #
       #  ax[0].set_title('3D mean value plot for subggoal_dist_to_goal')
       #  ax[0].set_xlabel('puck_to_goal_dist')
       #  ax[0].set_ylabel('hand_to_goal_dist')
       #  ax[0].set_zlabel('value')
       #  fig[0].legend(loc="lower right")
       #  fig[0].savefig(os.path.join(os.path.dirname(model_path), 'value_mean_dist_to_goal_%d.png' % episode))
       #  plt.close(fig[0])
       #
       #
       #  ax[1].scatter(subgoal_puck_dist_to_goal,subgoal_hand_dist_to_goal,value1_list,c='tab:orange',s=4.0,marker='o',label='value1')
       #  ax[1].scatter(subgoal_puck_dist_to_goal_true,subgoal_hand_dist_to_goal_true,value1_list,c='tab:red',s=4.0,marker='*',label='value1_truth')
       #  ax[1].scatter(selected_subgoal_puck_dist_to_goal,selected_subgoal_hand_dist_to_goal,value1,s=20.0,c='tab:green',marker='^',label='value1_subgoal')
       #  ax[1].set_title('3D mean value plot for subggoal_dist_to_goal')
       #  ax[1].set_xlabel('puck_to_goal_dist')
       #  ax[1].set_ylabel('hand_to_goal_dist')
       #  ax[1].set_zlabel('value')
       #  fig[1].legend(loc="lower right")
       #  fig[1].savefig(os.path.join(os.path.dirname(model_path), 'value_1_dist_to_goal_%d.png' % episode))
       #  plt.close(fig[1])
       #
       #
       #  ax[2].scatter(subgoal_puck_dist_to_goal,subgoal_hand_dist_to_goal,value2_list,c='tab:orange',s=4.0,marker='o',label='value2')
       #  ax[2].scatter(subgoal_puck_dist_to_goal_true,subgoal_hand_dist_to_goal_true,value2_list,c='tab:red',s=4.0,marker='*',label='value2_truth')
       #  ax[2].scatter(selected_subgoal_puck_dist_to_goal,selected_subgoal_hand_dist_to_goal,value2,s=20.0,c='tab:green',marker='^',label='value2_subgoal')
       #  ax[2].set_title('3D mean value plot for subggoal_dist_to_goal')
       #  ax[2].set_xlabel('puck_to_goal_dist')
       #  ax[2].set_ylabel('hand_to_goal_dist')
       #  ax[2].set_zlabel('value')
       #  fig[2].legend(loc="lower right")
       #  fig[2].savefig(os.path.join(os.path.dirname(model_path), 'value_2_dist_to_goal_%d.png' % episode))
       #  plt.close(fig[2])
       #
       #
       #
       #  # ax.scatter(selected_subgoal_hand_dist_to_goal,selected_subgoal_puck_dist_to_goal,value_mean,marker='^')
       #  #
       #  # ax.set_title('3D mean value plot for subggoal_dist_to_goal')
       #  # ax.set_xlabel('hand_to_goal_dist')
       #  # ax.set_ylabel('puck_to_goal_dist')
       #  # ax.set_zlabel('value')
       #  # fig.legend(loc="lower right")
       #  # # plt.tight_layout(pad=0.05)
       #  # # plt.show()
       #  # fig.savefig(os.path.join(os.path.dirname(model_path), 'value_dist_to_goal_%d.png' % episode))
       #  # plt.pause(0.2)
       #
       #  ## draw the value1
       #  # fig1 = plt.figure()
       #  # ax1= Axes3D(fig1)
       #  #mean value
       #  fig_2d, ax_2d = plt.subplots(1, 1, figsize=(8, 5))
       #  ax_2d.scatter(subgoal_dist_to_init,value_mean_list,c='tab:orange',marker='o',s=4.0,label='value_mean')
       #  ax_2d.scatter(subgoal_dist_to_init_true,value_mean_list,c='tab:red',marker='*',s=4.0,label='value_mean_truth')
       #  ax_2d.scatter(selected_subgoal_dist_to_init,value_mean,c='tab:green',marker='^',s=8.0,label='value_mean_selected')
       #  ax_2d.set_title('2d mean value plot for subgoal_dist_to_init')
       #  ax_2d.set_xlabel('subgoal_mean_value_to_init_dist')
       #  ax_2d.set_ylabel('value')
       #  fig_2d.legend(loc='lower right')
       #  fig_2d.savefig(os.path.join(os.path.dirname(model_path), '2d_value_mean_dist_to_init_%d.png' % episode))
       #  plt.close(fig_2d)
       #
       #  fig_2d, ax_2d = plt.subplots(1, 1, figsize=(8, 5))
       #  ax_2d.scatter(subgoal_dist_to_init,value1_list,c='tab:orange',marker='o',s=4.0,label='value_1')
       #  ax_2d.scatter(subgoal_dist_to_init_true,value1_list,c='tab:red',marker='*',s=4.0,label='value_1_truth')
       #  ax_2d.scatter(selected_subgoal_dist_to_init,value1,c='tab:green',marker='^',s=8.0,label='value_1_selected')
       #  ax_2d.set_title('2d value1 plot for subgoal_dist_to_init')
       #  ax_2d.set_xlabel('subgoal_value1_to_init_dist')
       #  ax_2d.set_ylabel('value')
       #  fig_2d.legend(loc='lower right')
       #  fig_2d.savefig(os.path.join(os.path.dirname(model_path), '2d_value1_dist_to_init_%d.png' % episode))
       #  plt.close(fig_2d)
       #
       #  fig_2d, ax_2d = plt.subplots(1, 1, figsize=(8, 5))
       #  ax_2d.scatter(subgoal_dist_to_init,value2_list,c='tab:orange',marker='o',s=4.0,label='value_2')
       #  ax_2d.scatter(subgoal_dist_to_init_true,value2_list,c='tab:red',marker='*',s=4.0,label='value_2_truth')
       #  ax_2d.scatter(selected_subgoal_dist_to_init,value2,c='tab:green',marker='^',s=8.0,label='value_2_selected')
       #  ax_2d.set_title('2d value2 plot for subgoal_dist_to_init')
       #  ax_2d.set_xlabel('subgoal_value1_to_init_dist')
       #  ax_2d.set_ylabel('value')
       #  fig_2d.legend(loc='lower right')
       #  fig_2d.savefig(os.path.join(os.path.dirname(model_path), '2d_value2_dist_to_init_%d.png' % episode))
       #  plt.close(fig_2d)
       #
       #  fig_2d, ax_2d = plt.subplots(1, 1, figsize=(8, 5))
       #  ax_2d.scatter(subgoal_dist_to_goal, value_mean_list, c='tab:orange', marker='o', s=4.0, label='value_mean')
       #  ax_2d.scatter(subgoal_dist_to_goal_true, value_mean_list, c='tab:red', marker='*', s=4.0, label='value_mean_truth')
       #  ax_2d.scatter(selected_subgoal_dist_to_goal, value_mean, c='tab:green', marker='^', s=8.0,
       #                label='value_mean_selected')
       #  ax_2d.set_title('2d mean value plot for subgoal_dist_to_goal')
       #  ax_2d.set_xlabel('subgoal_mean_value_to_goal_dist')
       #  ax_2d.set_ylabel('value')
       #  fig_2d.legend(loc='lower right')
       #  fig_2d.savefig(os.path.join(os.path.dirname(model_path), '2d_value_mean_dist_to_goal_%d.png' % episode))
       #  plt.close(fig_2d)
       #
       #  fig_2d, ax_2d = plt.subplots(1, 1, figsize=(8, 5))
       #  ax_2d.scatter(subgoal_dist_to_goal, value1_list, c='tab:orange', marker='o', s=4.0, label='value_1')
       #  ax_2d.scatter(subgoal_dist_to_goal_true, value1_list, c='tab:red', marker='*', s=4.0, label='value_1_truth')
       #  ax_2d.scatter(selected_subgoal_dist_to_goal, value1, c='tab:green', marker='^', s=8.0, label='value_1_selected')
       #  ax_2d.set_title('2d value1 plot for subgoal_dist_to_goal')
       #  ax_2d.set_xlabel('subgoal_value1_to_goal_dist')
       #  ax_2d.set_ylabel('value')
       #  fig_2d.legend(loc='lower right')
       #  fig_2d.savefig(os.path.join(os.path.dirname(model_path), '2d_value1_dist_to_goal_%d.png' % episode))
       #  plt.close(fig_2d)
       #
       #  fig_2d, ax_2d = plt.subplots(1, 1, figsize=(8, 5))
       #  ax_2d.scatter(subgoal_dist_to_goal, value2_list, c='tab:orange', marker='o', s=4.0, label='value_2')
       #  ax_2d.scatter(subgoal_dist_to_goal_true, value2_list, c='tab:red', marker='*', s=4.0, label='value_2_truth')
       #  ax_2d.scatter(selected_subgoal_dist_to_goal, value2, c='tab:green', marker='^', s=8.0, label='value_2_selected')
       #  ax_2d.set_title('2d value2 plot for subgoal_dist_to_goal')
       #  ax_2d.set_xlabel('subgoal_value2_to_goal_dist')
       #  ax_2d.set_ylabel('value')
       #  fig_2d.legend(loc='lower right')
       #  fig_2d.savefig(os.path.join(os.path.dirname(model_path), '2d_value2_dist_to_goal_%d.png' % episode))
       #  plt.close(fig_2d)
       #
       #  ax[3].scatter(subgoal_puck_dist_to_init, subgoal_hand_dist_to_init, value_mean_list, c='tab:orange',marker='o',s=4.0,label='value_mean')
       #  ax[3].scatter(subgoal_puck_dist_to_init_true, subgoal_hand_dist_to_init_true, value_mean_list, c='tab:red',marker='*',s=4.0,label='value_mean_truth')
       #  ax[3].scatter(selected_subgoal_puck_dist_to_init, selected_subgoal_hand_dist_to_init, value_mean, c='tab:green',s=20.0,
       #              marker='^', label='value_mean_subgoal')
       #  ax[3].set_title('3D mean value plot for subgoal_dist_to_init')
       #  ax[3].set_xlabel('puck_to_init_dist')
       #  ax[3].set_ylabel('hand_to_init_dist')
       #  ax[3].set_zlabel('value')
       #  fig[3].legend(loc="lower right", )
       #  fig[3].savefig(os.path.join(os.path.dirname(model_path), 'value_mean_dist_to_init_%d.png' % episode))
       #  plt.close(fig[3])
       #
       #  # value 1
       #  ax[4].scatter(subgoal_puck_dist_to_init, subgoal_hand_dist_to_init, value1_list, c='tab:orange',marker='o',s=4.0,label='value1')
       #  ax[4].scatter(subgoal_puck_dist_to_init_true, subgoal_hand_dist_to_init_true, value1_list, c='tab:red',marker='*',s=4.0,label='value1_truth')
       #
       #  ax[4].scatter(selected_subgoal_puck_dist_to_init, selected_subgoal_hand_dist_to_init, value1,c='tab:green',s=20.0, marker='^',label='value1_subogal')
       #
       #  ax[4].set_title('3D mean value plot for subgoal_dist_to_init')
       #  ax[4].set_xlabel('puck_to_init_dist')
       #  ax[4].set_ylabel('hand_to_init_dist')
       #  ax[4].set_zlabel('value')
       #  fig[4].legend(loc="lower right", )
       #  fig[4].savefig(os.path.join(os.path.dirname(model_path), 'value_1_dist_to_init_%d.png' % episode))
       #  plt.close(fig[4])
       #
       # ## value2
       #  ax[5].scatter(subgoal_puck_dist_to_init, subgoal_hand_dist_to_init, value2_list, c='tab:orange',s=4.0,marker='o',label='value2')
       #  ax[5].scatter(subgoal_puck_dist_to_init_true, subgoal_hand_dist_to_init_true, value2_list, c='tab:red',s=4.0,marker='*',label='value2_truth')
       #
       #  ax[5].scatter(selected_subgoal_puck_dist_to_init, selected_subgoal_hand_dist_to_init, value2,s=20.0,c='tab:green', marker='^',label='value2_subgoal')
       #
       #  ax[5].set_title('3D mean value plot for subgoal_dist_to_init')
       #  ax[5].set_xlabel('puck_to_init_dist')
       #  ax[5].set_ylabel('hand_to_init_dist')
       #  ax[5].set_zlabel('value')
       #  fig[5].legend(loc="lower right", )
       #  fig[5].savefig(os.path.join(os.path.dirname(model_path), 'value_2_dist_to_init_%d.png' % episode))
       #  plt.close(fig[5])
       #
       #
       #  # ax1.set_title('3D mean value plot for subgoal_dist_to_init')
       #  # ax1.set_xlabel('hand_to_init_dist')
       #  # ax1.set_ylabel('puck_to_init_dist')
       #  # ax1.set_zlabel('value')
       #  # fig1.legend(loc="lower right",)
       #  # # fig1.tight_layout(pad=0.05)
       #  # # plt.show()
       #  # fig1.savefig(os.path.join(os.path.dirname(model_path), 'value_dist_to_init_%d.png' % episode))
       #  # plt.pause(0.2)
       #  ## draw the value with dist_to_init
       #  # ax.scatter(subgoal_hand_dist_to_goal, subgoal_puck_dist_to_goal, value2_list, marker='o')
       #  # ax.scatter(selected_subgoal_hand_dist_to_goal, selected_subgoal_puck_dist_to_goal, value2, marker='^')
       #  # ax.set_title('3D mean value plot for subggoal_dist_to_goal')
       #  # ax.set_xlabel('hand_to_goal_dist')
       #  # ax.set_ylabel('puck_to_goal_dist')
       #  # ax.set_zlabel('value')
       #  # plt.show()
       #  # plt.savefig(os.path.join(os.path.dirname(model_path), 'value2_%d.png' % episode))
       #  # plt.pause(0.2)
        print('plot finished!')

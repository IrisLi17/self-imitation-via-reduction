## SAC for visual gripper
CUDA_VISIBLE_DEVICES=2 python run_her.py --env Image84SawyerPushAndReachArenaTrainEnvBig-v0  --num_timesteps 3.5e6  --num_workers 32  --priority --log_path logs/pnr_sac/sac_her_sparse_reward  --reward_type state_sparse --reward_object 1 --epsilon 0.08

## SIR for visual gripper
CUDA_VISIBLE_DEVICES=6 python run_her_augment.py --env Image84SawyerPushAndReachArenaTrainEnvBig-v0 --num_timesteps 3.5e6  --num_workers 32 --imitation_coef 0.1 --priority --log_path logs/pnr_sac/sac_sir_sparse_reward --reward_type state_sparse --reward_object 1 --epsilon 0.08

## SIL for visual gripper
CUDA_VISIBLE_DEVICES=1 python run_her.py --env Image84SawyerPushAndReachArenaTrainEnvBig-v0  --num_timesteps 3.5e6  --num_workers 16 --sil_coef 0.1 --priority --log_path logs/pnr_sac/sac_sil_sparse_0.08 --sil --reward_type state_sparse --reward_object 1 --epsilon 0.08


## DS for visual gripper
CUDA_VISIBLE_DEVICES=1 python run_her.py --env Image84SawyerPushAndReachArenaTrainEnvBig-v0  --num_timesteps 3.5e6  --num_workers 16  --priority --log_path logs/pnr_sac/sac_her_dense_0.08  --reward_type dense --reward_object 1 --epsilon 0.08


### comment
##separate threshold: reward_object 2, specifying the threshold in the wrapper.py  compute_state_reward_and_success function
##one threshold for l2 distance: reward_object 1, epsilon is  the success threshold

## reward_type: dense or state_sparse
## SAC for visual gripper
CUDA_VISIBLE_DEVICES=2 python run_her.py --env Image84SawyerPushAndReachArenaTrainEnvBig-v0  --num_timesteps 3.5e6  --num_workers 32  --priority --log_path logs/pnr_sac/sac_her_sparse_reward  --reward_type state_sparse --reward_object 1 --epsilon 0.08

## SIR for visual gripper
CUDA_VISIBLE_DEVICES=6 python run_her_augment.py --env Image84SawyerPushAndReachArenaTrainEnvBig-v0 --num_timesteps 3.5e6  --num_workers 32 --imitation_coef 0.1 --priority --log_path logs/pnr_sac/sac_sir_sparse_reward --reward_type sparse --reward_object 1 --epsilon 0.08
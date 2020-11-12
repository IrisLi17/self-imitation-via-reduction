# SAC-based
# Hard case 30%.
# SAC
CUDA_VISIBLE_DEVICES=0 python run_her.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e6 --num_workers 32 --priority --log_path logs/FetchPushWallObstacle-v4_random0.7/her_sac_32workers/0
# SIR
CUDA_VISIBLE_DEVICES=0 python run_her_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e6 --num_workers 32 --start_augment 0 --imitation_coef 0.1 --priority --log_path logs/FetchPushWallObstacle-v4_random0.7/her_sac_sir_32workers/0
# SIL
CUDA_VISIBLE_DEVICES=0 python run_her.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --sil --num_timesteps 1e6 --num_workers 32 --sil_coef 0.1 --priority --log_path logs/FetchPushWallObstacle-v4_random0.7/her_sac_sil_32workers/0
# DS
CUDA_VISIBLE_DEVICES=0 python run_her.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --reward_type dense --num_timesteps 1e6 --num_workers 32 --priority --log_path logs/FetchPushWallObstacle-v4_random0.7/her_sac_ds_32workers/0

# Uniform.
CUDA_VISIBLE_DEVICES=0 python run_her.py --env FetchPushWallObstacle-v4 --random_ratio 1.0 --num_timesteps 1e6 --num_workers 32 --priority --log_path logs/FetchPushWallObstacle-v4_random1.0/her_sac_32workers/0
CUDA_VISIBLE_DEVICES=0 python run_her_augment.py --env FetchPushWallObstacle-v4 --random_ratio 1.0 --num_timesteps 1e6 --num_workers 32 --start_augment 0 --imitation_coef 0.1 --priority --log_path logs/FetchPushWallObstacle-v4_random1.0/her_sac_sir_32workers/0
CUDA_VISIBLE_DEVICES=0 python run_her.py --env FetchPushWallObstacle-v4 --random_ratio 1.0 --sil --num_timesteps 1e6 --num_workers 32 --sil_coef 0.1 --priority --log_path logs/FetchPushWallObstacle-v4_random1.0/her_sac_sil_32workers/0
CUDA_VISIBLE_DEVICES=0 python run_her.py --env FetchPushWallObstacle-v4 --random_ratio 1.0 --reward_type dense --num_timesteps 1e6 --num_workers 32 --priority --log_path logs/FetchPushWallObstacle-v4_random1.0/her_sac_ds_32workers/0

# PPO-based
# Hard 30%.
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 5e7 --log_path logs/FetchPushWallObstacle-v4_random0.7/ppo/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 5e7 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 8 --log_path logs/FetchPushWallObstacle-v4_random0.7/ppo_sir/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 5e7 --n_subgoal 2 --parallel --self_imitate --aug_clip 0.0 --reuse_times 1 --log_path logs/FetchPushWallObstacle-v4_random0.7/ppo_sil/0
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 5e7 --reward_type dense --log_path logs/FetchPushWallObstacle-v4_random0.7/ppo_ds/0

# Uniform.
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchPushWallObstacle-v4 --random_ratio 1.0 --num_timesteps 5e7 --log_path logs/FetchPushWallObstacle-v4_random1.0/ppo/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 1.0 --num_timesteps 5e7 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 8 --log_path logs/FetchPushWallObstacle-v4_random1.0/ppo_sir/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 1.0 --num_timesteps 5e7 --n_subgoal 2 --parallel --self_imitate --aug_clip 0.0 --reuse_times 1 --log_path logs/FetchPushWallObstacle-v4_random1.0/ppo_sil/0
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchPushWallObstacle-v4 --random_ratio 1.0 --num_timesteps 5e7 --reward_type dense --log_path logs/FetchPushWallObstacle-v4_random1.0/ppo_ds/0

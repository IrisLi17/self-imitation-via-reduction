CUDA_VISIBLE_DEVICES=0 python run_her.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e7 --num_workers 32 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/her_sac_32workers/0
CUDA_VISIBLE_DEVICES=0 python run_her_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e7 --num_workers 32 --start_augment 0 --imitation_coef 1 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/her_sac_aug_32workers/start0
CUDA_VISIBLE_DEVICES=0 python run_her.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --sil --num_timesteps 1e7 --num_workers 32 --sil_coef 0.1 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/her_sac_sil_32workers/coef0.1
# Priority
CUDA_VISIBLE_DEVICES=0 python run_her.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e7 --num_workers 32 --priority --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/her_sac_32workers/0_priority
CUDA_VISIBLE_DEVICES=0 python run_her_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e7 --num_workers 32 --start_augment 0 --imitation_coef 0.1 --priority --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/her_sac_aug_32workers/start0_silloss0.1uni_threshold_priority
CUDA_VISIBLE_DEVICES=0 python run_her.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --sil --num_timesteps 1e7 --num_workers 32 --sil_coef 0.1 --priority --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/her_sac_sil_32workers/coef0.1_priority

# New env
CUDA_VISIBLE_DEVICES=0 python run_her.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e6 --num_workers 32 --priority --log_path logs/FetchPushWallObstacle-v4new_random0.7/her_sac_32workers/0_priority
CUDA_VISIBLE_DEVICES=0 python run_her_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e6 --num_workers 32 --start_augment 0 --imitation_coef 0.1 --priority --log_path logs/FetchPushWallObstacle-v4new_random0.7/her_sac_aug_32workers/start0_silloss0.1uni_threshold_priority
CUDA_VISIBLE_DEVICES=0 python run_her.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --sil --num_timesteps 1e6 --num_workers 32 --sil_coef 0.1 --priority --log_path logs/FetchPushWallObstacle-v4new_random0.7/her_sac_sil_32workers/coef0.1_priority
CUDA_VISIBLE_DEVICES=0 python run_her.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --reward_type dense --num_timesteps 1e6 --num_workers 32 --priority --log_path logs/FetchPushWallObstacle-v4new_random0.7/her_sac_32workers/ds_priority
CUDA_VISIBLE_DEVICES=0 python run_her_augment.py --env FetchPushWallObstacle-v4 --random_ratio 1.0 --num_timesteps 1e6 --num_workers 32 --start_augment 0 --imitation_coef 0.1 --priority --log_path logs/FetchPushWallObstacle-v4new_random1.0/her_sac_aug_32workers/0_filter0.71.0_priority
CUDA_VISIBLE_DEVICES=0 python run_her.py --env FetchPushWallObstacle-v4 --random_ratio 1.0 --sil --num_timesteps 1e6 --num_workers 32 --sil_coef 0.1 --priority --log_path logs/FetchPushWallObstacle-v4new_random1.0/her_sac_sil_32workers/coef0.1_priority


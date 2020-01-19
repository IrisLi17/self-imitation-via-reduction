CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e8 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo_value/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e8 --n_subgoal 2 --parallel --aug_clip 0.1 --reuse_times 3 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo_augment_value/0_reuse3
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchPushWallObstacle-v4 --random_ratio 1.0 --num_timesteps 1e8 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/ppo_value/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 1.0 --num_timesteps 1e8 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 8 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/ppo_augment_value/0_reuse8
# Reward covers all cases
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchPushWallObstacle-v4 --random_ratio 1.0 --num_timesteps 1e8 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/ppo_value/1
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 1.0 --num_timesteps 1e8 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 8 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/ppo_augment_value/1_reuse8


python run_ppo.py --env FetchPushWallObstacle-v4 --num_timesteps 1e8 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/ppo/0
CUDA_VISIBLE_DEVICES=6 python run_ppo.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e8 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo/0
CUDA_VISIBLE_DEVICES=1 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --num_timesteps 1e8 --n_subgoal 4 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/ppo_augment/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e8 --n_subgoal 1 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo_augment/0
CUDA_VISIBLE_DEVICES=2 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e8 --n_subgoal 4 --parallel --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo_augment/1
# clip adv from demo to non-negative
CUDA_VISIBLE_DEVICES=1 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e8 --n_subgoal 4 --parallel --aug_clip 0.0 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo_augment/2
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e8 --n_subgoal 4 --parallel --aug_clip 0.0 --start_augment 5e6  --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo_augment/6
CUDA_VISIBLE_DEVICES=1 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e8 --n_subgoal 2 --parallel --aug_clip 0.0 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo_augment/5
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e8 --n_subgoal 2 --parallel --aug_clip 0.0 --start_augment 5e6 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo_augment/7
# clip adv from demo to 0.1 std
CUDA_VISIBLE_DEVICES=1 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e8 --n_subgoal 4 --parallel --aug_clip 0.1 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo_augment/3
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e8 --n_subgoal 2 --parallel --aug_clip 0.1 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo_augment/4
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e8 --n_subgoal 2 --parallel --aug_clip 0.1 --reuse_times 3 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo_augment/4_reuse3
# Update online part
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e8 --n_subgoal 2 --parallel --aug_clip 0.1 --reuse_times 3 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo_augment/4_reuse3_new
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e8 --n_subgoal 2 --parallel --aug_clip 0.1 --reuse_times 5 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo_augment/4_reuse5
# Update online part
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e8 --n_subgoal 2 --parallel --aug_clip 0.1 --reuse_times 5 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo_augment/4_reuse5_new
# Schedule
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e8 --n_subgoal 2 --parallel --aug_clip 0.1 --reuse_times 5 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo_augment/4_reuse1-5
CUDA_VISIBLE_DEVICES=2 python run_ppo.py --env FetchPushWallDoubleObstacle-v1 --num_timesteps 1e8 --log_path logs/FetchPushWallDoubleObstacle-v1/ppo/0

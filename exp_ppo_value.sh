CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e8 --n_subgoal 2 --parallel --aug_clip 0.1 --reuse_times 5 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo_augment_value/0_reuse1-5


python run_ppo.py --env FetchPushWallObstacle-v4 --num_timesteps 1e8 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/ppo/0
CUDA_VISIBLE_DEVICES=6 python run_ppo.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e8 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo/0

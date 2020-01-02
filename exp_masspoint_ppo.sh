CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env MasspointPushSingleObstacle-v1 --num_timesteps 1e8 --log_path logs/MasspointPushSingleObstacle-v1/ppo/0
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 1e8 --log_path logs/MasspointPushDoubleObstacle-v1/ppo/0

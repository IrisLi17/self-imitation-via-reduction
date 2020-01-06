CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env MasspointPushSingleObstacle-v2 --num_timesteps 1e8 --log_path logs/MasspointPushSingleObstacle-v2/ppo/0
CUDA_VISIBLE_DEVICES=1 python run_ppo_augment.py --env MasspointPushSingleObstacle-v2 --num_timesteps 1e8 --n_subgoal 2 --parallel --aug_clip 0.0 --log_path logs/MasspointPushSingleObstacle-v2/ppo_augment/0
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 1e8 --log_path logs/MasspointPushDoubleObstacle-v1/ppo/0

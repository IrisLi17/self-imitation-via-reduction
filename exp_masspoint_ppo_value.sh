CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 4e8 --random_ratio 0.7 --log_path logs/MasspointPushDoubleObstacle-v1/ppo_value/0
# evaluate 20 times
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 4e8 --random_ratio 0.7 --log_path logs/MasspointPushDoubleObstacle-v1/ppo_value/1

CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 4e8 --random_ratio 0.7 --n_subgoal 2 --parallel --aug_clip 0.1 --reuse_times 3 --log_path logs/MasspointPushDoubleObstacle-v1/ppo_augment_value/0_reuse3
# evaluate 20 times
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 4e8 --random_ratio 0.7 --n_subgoal 2 --parallel --aug_clip 0.1 --reuse_times 4 --log_path logs/MasspointPushDoubleObstacle-v1/ppo_augment_value/1_reuse4


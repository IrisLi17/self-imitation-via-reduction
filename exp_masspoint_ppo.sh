# CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env MasspointPushSingleObstacle-v2 --num_timesteps 1e8 --log_path logs/MasspointPushSingleObstacle-v2/ppo/0
# CUDA_VISIBLE_DEVICES=1 python run_ppo_augment.py --env MasspointPushSingleObstacle-v2 --num_timesteps 1e8 --n_subgoal 2 --parallel --aug_clip 0.0 --log_path logs/MasspointPushSingleObstacle-v2/ppo_augment/0
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 4e8 --log_path logs/MasspointPushDoubleObstacle-v1/ppo/0
# range 2.0
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 4e8 --random_ratio 0.7 --log_path logs/MasspointPushDoubleObstacle-v1/ppo/1
# range 1.5
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 4e8 --random_ratio 0.7 --log_path logs/MasspointPushDoubleObstacle-v1/ppo/2
# Force 50% harder case in hard sample
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 4e8 --random_ratio 0.7 --log_path logs/MasspointPushDoubleObstacle-v1/ppo/3
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 4e8 --n_subgoal 4 --parallel --aug_clip 0.0 --log_path logs/MasspointPushDoubleObstacle-v1/ppo_augment/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 4e8 --n_subgoal 2 --parallel --aug_clip 0.1 --log_path logs/MasspointPushDoubleObstacle-v1/ppo_augment/1
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 4e8 --random_ratio 0.7 --n_subgoal 2 --parallel --aug_clip 0.1 --reuse_times 3 --log_path logs/MasspointPushDoubleObstacle-v1/ppo_augment/2_reuse3
# Force 50% harder case in hard sample
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 4e8 --random_ratio 0.7 --n_subgoal 2 --parallel --aug_clip 0.1 --reuse_times 3 --log_path logs/MasspointPushDoubleObstacle-v1/ppo_augment/3_reuse3

CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 4e8 --random_ratio 0.7 --log_path logs/MasspointPushDoubleObstacle-v1/ppo_value/0
# evaluate 20 times
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 4e8 --random_ratio 0.7 --log_path logs/MasspointPushDoubleObstacle-v1/ppo_value/1
# Uniform, reward covers all cases
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 6e8 --random_ratio 1.0 --log_path logs/MasspointPushDoubleObstacle-v1_uniform/ppo_value/0
# Hard case 30%, reward covers all cases
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 4e8 --random_ratio 0.7 --log_path logs/MasspointPushDoubleObstacle-v1/ppo_value/2

CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 4e8 --random_ratio 0.7 --n_subgoal 2 --parallel --aug_clip 0.1 --reuse_times 3 --log_path logs/MasspointPushDoubleObstacle-v1/ppo_augment_value/0_reuse3
# evaluate 20 times
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 4e8 --random_ratio 0.7 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 4 --log_path logs/MasspointPushDoubleObstacle-v1/ppo_augment_value/1_reuse4
# Uniform, reward covers all cases
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 6e8 --random_ratio 1.0 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 8 --log_path logs/MasspointPushDoubleObstacle-v1_uniform/ppo_augment_value/0_reuse8
# Uniform, reward covers all cases
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 6e8 --random_ratio 1.0 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 8 --log_path logs/MasspointPushDoubleObstacle-v1_uniform/ppo_augment_value/0_reuse8_adapt
# Uniform, weight advantage
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 6e8 --random_ratio 1.0 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 8 --log_path logs/MasspointPushDoubleObstacle-v1_uniform/ppo_augment_value/0_reuse8_weight0.2
# Hard case 30%, reward covers all cases
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 4e8 --random_ratio 0.7 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 8 --log_path logs/MasspointPushDoubleObstacle-v1/ppo_augment_value/2_reuse8
# Hard case 30%, reward covers all cases
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 4e8 --random_ratio 0.7 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 8 --log_path logs/MasspointPushDoubleObstacle-v1/ppo_augment_value/2_reuse8_adapt


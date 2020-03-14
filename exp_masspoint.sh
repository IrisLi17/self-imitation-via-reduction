# Uniform.
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 3e8 --random_ratio 1.0 --log_path logs/MasspointPushDoubleObstacle-v1_uniform/ppo/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 3e8 --random_ratio 1.0 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 8 --log_path logs/MasspointPushDoubleObstacle-v1_uniform/sir_re1-8/0

# Hard case 30%.
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 3e8 --random_ratio 0.7 --log_path logs/MasspointPushDoubleObstacle-v1_random0.7/ppo/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env MasspointPushDoubleObstacle-v1 --num_timesteps 3e8 --random_ratio 0.7 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 8 --log_path logs/MasspointPushDoubleObstacle-v1_random0.7/sir_re1-8/0

# AttentionPolicy
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env MasspointPushDoubleObstacle-v1 --policy AttentionPolicy --num_timesteps 3e8 --random_ratio 1.0 --log_path logs/MasspointPushDoubleObstacle-v1_uniform/ppo_attention/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env MasspointPushDoubleObstacle-v1 --policy AttentionPolicy --num_timesteps 3e8 --random_ratio 1.0 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 8 --log_path logs/MasspointPushDoubleObstacle-v1_uniform/ppo_attention_aug/0

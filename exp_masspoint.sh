# Hard case 30%.
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env MasspointPushDoubleObstacle-v1 --policy AttentionPolicy --num_timesteps 3e8 --random_ratio 0.7 --log_path logs/MasspointPushDoubleObstacle-v1_random0.7/ppo_attention/0
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env MasspointPushDoubleObstacle-v1 --policy AttentionPolicy --num_timesteps 3e8 --random_ratio 0.7 --reward_type dense --log_path logs/MasspointPushDoubleObstacle-v1_random0.7/ppo_attention_ds/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env MasspointPushDoubleObstacle-v1 --policy AttentionPolicy --num_timesteps 3e8 --random_ratio 0.7 --parallel --aug_clip 0.0 --reuse_times 1 --self_imitate --sil_clip 0.15 --log_path logs/MasspointPushDoubleObstacle-v1_random0.7/ppo_attention_sil/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env MasspointPushDoubleObstacle-v1 --policy AttentionPolicy --num_timesteps 3e8 --random_ratio 0.7 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 4 --start_augment 3e7 --log_path logs/MasspointPushDoubleObstacle-v1_random0.7/ppo_attention_sir/0

# Uniform.
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env MasspointPushDoubleObstacle-v1 --policy AttentionPolicy --num_timesteps 3e8 --random_ratio 1.0 --log_path logs/MasspointPushDoubleObstacle-v1_random1.0/ppo_attention/0
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env MasspointPushDoubleObstacle-v1 --policy AttentionPolicy --num_timesteps 3e8 --random_ratio 1.0 --reward_type dense --log_path logs/MasspointPushDoubleObstacle-v1_random1.0/ppo_attention_ds/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env MasspointPushDoubleObstacle-v1 --policy AttentionPolicy --num_timesteps 3e8 --random_ratio 1.0 --parallel --aug_clip 0.0 --reuse_times 1 --self_imitate --sil_clip 0.15 --log_path logs/MasspointPushDoubleObstacle-v1_random1.0/ppo_attention_sil/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env MasspointPushDoubleObstacle-v1 --policy AttentionPolicy --num_timesteps 3e8 --random_ratio 1.0 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 4 --start_augment 3e7 --log_path logs/MasspointPushDoubleObstacle-v1_random1.0/ppo_attention_sir/0

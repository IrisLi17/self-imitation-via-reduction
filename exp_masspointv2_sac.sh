CUDA_VISIBLE_DEVICES=0 python run_her.py --env MasspointPushDoubleObstacle-v2 --random_ratio 0.7 --num_timesteps 1e7 --num_workers 32 --policy AttentionPolicy --priority --log_path logs/MasspointPushDoubleObstacle-v2_random0.7/her_sac_32workers/attention_priority_logstd0.0
CUDA_VISIBLE_DEVICES=0 python run_her_augment.py --env MasspointPushDoubleObstacle-v2 --random_ratio 0.7 --num_timesteps 1e7 --num_workers 32 --policy AttentionPolicy --priority --start_augment 0 --imitation_coef 0.1 --log_path logs/MasspointPushDoubleObstacle-v2_random0.7/her_sac_aug_32workers/attention_start0_priority_logstd0.0


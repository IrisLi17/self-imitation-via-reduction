CUDA_VISIBLE_DEVICES=0 python run_her.py --env MasspointPushDoubleObstacle-v2 --random_ratio 0.7 --num_timesteps 1e7 --num_workers 32 --policy CustomSACPolicy --priority --log_path logs/MasspointPushDoubleObstacle-v2_random0.7/her_sac_32workers/mlp_priority_egoeval
CUDA_VISIBLE_DEVICES=0 python run_her_augment.py --env MasspointPushDoubleObstacle-v2 --random_ratio 0.7 --num_timesteps 1e7 --num_workers 32 --policy CustomSACPolicy --priority --start_augment 0 --imitation_coef 0.1 --log_path logs/MasspointPushDoubleObstacle-v2_random0.7/her_sac_aug_32workers/mlp_start0_priority_egoeval


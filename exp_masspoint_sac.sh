CUDA_VISIBLE_DEVICES=0 python run_her.py --env MasspointPushDoubleObstacle-v1 --random_ratio 0.7 --num_timesteps 1e7 --num_workers 32 --log_path logs/MasspointPushDoubleObstacle-v1_random0.7/her_sac_32workers/0
CUDA_VISIBLE_DEVICES=0 python run_her.py --env MasspointPushDoubleObstacle-v1 --random_ratio 0.7 --policy AttentionPolicy --num_timesteps 1e7 --num_workers 32 --log_path logs/MasspointPushDoubleObstacle-v1_random0.7/her_sac_32workers/attention_0
CUDA_VISIBLE_DEVICES=0 python run_her_augment.py --env MasspointPushDoubleObstacle-v1 --random_ratio 0.7 --num_timesteps 1e7 --num_workers 32 --start_augment 0 --imitation_coef 1 --log_path logs/MasspointPushDoubleObstacle-v1_random0.7/her_sac_aug_32workers/start0
CUDA_VISIBLE_DEVICES=0 python run_her_augment.py --env MasspointPushDoubleObstacle-v1 --random_ratio 0.7 --policy AttentionPolicy --num_timesteps 1e7 --num_workers 32 --start_augment 0 --imitation_coef 0.1 --log_path logs/MasspointPushDoubleObstacle-v1_random0.7/her_sac_aug_32workers/attention_start0_silloss0.1
# Priority
CUDA_VISIBLE_DEVICES=0 python run_her.py --env MasspointPushDoubleObstacle-v1 --random_ratio 0.7 --policy AttentionPolicy --num_timesteps 1e7 --num_workers 32 --priority --log_path logs/MasspointPushDoubleObstacle-v1_random0.7/her_sac_32workers/attention_priority
CUDA_VISIBLE_DEVICES=0 python run_her.py --env MasspointPushDoubleObstacle-v1 --random_ratio 0.7 --policy AttentionPolicy --num_timesteps 1e7 --num_workers 32 --sil --sil_coef 0.1 --priority --log_path logs/MasspointPushDoubleObstacle-v1_random0.7/her_sac_sil_32workers/attention_0.1_priority
CUDA_VISIBLE_DEVICES=0 python run_her_augment.py --env MasspointPushDoubleObstacle-v1 --random_ratio 0.7 --policy AttentionPolicy --num_timesteps 1e7 --num_workers 32 --start_augment 1e7 --imitation_coef 0.1 --priority --log_path logs/MasspointPushDoubleObstacle-v1_random0.7/her_sac_aug_32workers/attention_logstd0.0_start1e7_silloss0.1_priority

# S-Maze
CUDA_VISIBLE_DEVICES=0 python run_her.py --env MasspointMaze-v2 --num_timesteps 1e5 --num_workers 4 --log_path logs/MasspointMaze-v2/her_sac_4workers/0


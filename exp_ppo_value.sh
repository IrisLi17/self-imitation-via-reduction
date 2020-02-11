# CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e8 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo_value/0
# CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e8 --n_subgoal 2 --parallel --aug_clip 0.1 --reuse_times 3 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo_augment_value/0_reuse3
# CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchPushWallObstacle-v4 --random_ratio 1.0 --num_timesteps 1e8 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/ppo_value/0
# CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 1.0 --num_timesteps 1e8 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 8 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/ppo_augment_value/0_reuse8
# Reward covers all cases
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchPushWallObstacle-v4 --random_ratio 1.0 --num_timesteps 1e8 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/ppo_value/1
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 1.0 --num_timesteps 1e8 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 8 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/ppo_augment_value/1_reuse8
# Weight adv, also self augment
# CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 1.0 --num_timesteps 1e8 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 4 --aug_adv_weight 1.0 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/ppo_augment_value/1_reuse4_selfaug
# CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 1.0 --num_timesteps 1e8 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 8 --aug_adv_weight 0.2 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/ppo_augment_value/1_reuse8_weight0.2_selfaug

CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchPushWallObstacle-v4 --random_ratio 1.0 --num_timesteps 1e8 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/ppo_value/1_run1
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 1.0 --num_timesteps 1e8 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 8 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/ppo_augment_value/1_reuse8_run1
# Hard 70%
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchPushWallObstacle-v4 --random_ratio 0.3 --num_timesteps 1e8 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/ppo_value/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.3 --num_timesteps 5e7 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 8 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/ppo_augment_value/0_reuse8
# Curriculum
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchPushWallObstacle-v4 --num_timesteps 5e7 --curriculum --log_path logs/FetchPushWallObstacle-v4_heavy_cu_fixz/ppo_value/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --num_timesteps 5e7 --curriculum --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 8 --log_path logs/FetchPushWallObstacle-v4_heavy_cu_fixz/ppo_augment_value/0_reuse8
# Reuse times
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e8 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 1 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo_augment_value/0_reuse1
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e8 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 2 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo_augment_value/0_reuse2
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e8 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 4 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo_augment_value/0_reuse4
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e8 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 8 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo_augment_value/0_reuse8
# Self imitate
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 1.0 --num_timesteps 1e8 --n_subgoal 2 --parallel --self_imitate --aug_clip 0.0 --reuse_times 8 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/ppo_sil/0_reuse8
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 1e8 --n_subgoal 2 --parallel --self_imitate --aug_clip 0.0 --reuse_times 8 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo_sil/0_reuse8
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.3 --num_timesteps 1e8 --n_subgoal 2 --parallel --self_imitate --aug_clip 0.0 --reuse_times 1 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/ppo_sil/0_reuse1

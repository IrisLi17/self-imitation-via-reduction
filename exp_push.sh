# Uniform.
# PPO
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchPushWallObstacle-v4 --random_ratio 1.0 --num_timesteps 5e7 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/ppo/0
# SIR
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 1.0 --num_timesteps 5e7 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 8 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/sir_re8/0

# Hard 30%
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 5e7 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 5e7 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 8 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/sir_re8/0

# Hard 70%
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchPushWallObstacle-v4 --random_ratio 0.3 --num_timesteps 5e7 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/ppo/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.3 --num_timesteps 5e7 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 8 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/sir_re8/0

# Adaptive curriculum
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchPushWallObstacle-v4 --num_timesteps 5e7 --curriculum --log_path logs/FetchPushWallObstacle-v4_heavy_cu_fixz/ppo/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --num_timesteps 5e7 --curriculum --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 8 --log_path logs/FetchPushWallObstacle-v4_heavy_cu_fixz/sir_re8/0

# Data staleness
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 5e7 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 1 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/sir_re1/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 5e7 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 2 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/sir_re2/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 5e7 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 4 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/sir_re4/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 5e7 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 16 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/sir_re16/0

# Self imitate
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 5e7 --n_subgoal 2 --parallel --self_imitate --aug_clip 0.0 --reuse_times 1 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo_sil_clip0.2/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchPushWallObstacle-v4 --random_ratio 0.7 --num_timesteps 5e7 --parallel --self_imitate --aug_adv_weight 1.0 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/ppo_sil/0

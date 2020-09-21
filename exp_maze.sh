CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env MasspointMaze-v3 --random_ratio 1.0 --num_timesteps 1e5 --log_path logs/MasspointMaze-v3/ppo/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env MasspointMaze-v3 --random_ratio 1.0 --num_timesteps 1e5 --n_subgoal 1 --parallel --aug_clip 0.0 --reuse_times 2 --log_path logs/MasspointMaze-v3/ppo_sir/0
CUDA_VISIBLE_DEVICES=0 python run_her.py --env MasspointMaze-v3 --random_ratio 1.0 --num_timesteps 1e5 --num_workers 1 --priority --log_path logs/MasspointMaze-v3/her_sac_1workers/0
CUDA_VISIBLE_DEVICES=0 python run_her_augment.py --env MasspointMaze-v3 --random_ratio 1.0 --num_timesteps 1e5 --num_workers 1 --start_augment 0 --imitation_coef 0.1 --priority --log_path logs/MasspointMaze-v3/her_sac_sir_1workers/0


CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env MasspointMaze-v1 --num_timesteps 5e5 --log_path logs/MasspointMaze-v1/ppo_value/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env MasspointMaze-v1 --num_timesteps 5e5 --n_subgoal 3 --parallel --aug_clip 0.0 --reuse_times 3 --log_path logs/MasspointMaze-v1/ppo_augment_value/0_reuse3


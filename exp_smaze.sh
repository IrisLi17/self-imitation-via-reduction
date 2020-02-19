# Uniform.
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env MasspointMaze-v2 --num_timesteps 1.5e6 --log_path logs/MasspointMaze-v2/ppo/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env MasspointMaze-v2 --num_timesteps 1.5e6 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 2 --log_path logs/MasspointMaze-v2/sir_re2/0


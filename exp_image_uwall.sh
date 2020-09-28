CUDA_VISIBLE_DEVICES=7 python run_ppo_augment.py --env Image48PointmassUWallTrainEnvBig-v0  --num_timesteps 3e8 --random_ratio 1.0 --log_path logs/MasspointPushDoubleObstacle-v1_random1.0/uwall_state_dist_eps_1.0_cpu_1 --parallel --start_augment 0 --aug_clip 0.0 --reuse_times 4 --n_subgoal 2 --epsilon 1.0 


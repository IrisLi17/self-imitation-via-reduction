trap '' 15
CUDA_VISIBLE_DEVICES=2 python test_universal.py --env FetchPushWallObstacle-v4 --heavy_obstacle --random_gripper --num_timesteps 1e7 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom/her_sac

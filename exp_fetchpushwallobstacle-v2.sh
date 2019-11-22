trap '' 15
CUDA_VISIBLE_DEVICES=1 python test_multiplegoals.py --env FetchPushWallObstacle-v2 --policy CustomSACPolicy --heavy_obstacle --random_gripper --num_timesteps 1e7 --log_path logs/FetchPushWallObstacle-v2_heavy_purerandom/her_sac
CUDA_VISIBLE_DEVICES=1 python test_multiplegoals.py --env FetchPushWallObstacle-v2 --policy CustomSACPolicy --heavy_obstacle --random_gripper --num_timesteps 1e7 --buffer_size 2e6 --batch_size 128 --log_path logs/FetchPushWallObstacle-v2_heavy_purerandom/her_sac1


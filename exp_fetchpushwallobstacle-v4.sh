trap '' 15
# CUDA_VISIBLE_DEVICES=2 python test_universal.py --env FetchPushWallObstacle-v4 --heavy_obstacle --random_gripper --num_timesteps 1e7 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom/her_sac
CUDA_VISIBLE_DEVICES=2 python test_universal.py --env FetchPushWallObstacle-v4 --policy LnMlpPolicy --heavy_obstacle --random_gripper --num_timesteps 1e7 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom/her_sac/ln
CUDA_VISIBLE_DEVICES=2 python test_universal.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --heavy_obstacle --random_gripper --num_timesteps 1e7 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom/her_sac/custom

# Visualization.
# CUDA_VISIBLE_DEVICES=2 python test_universal.py --env FetchPushWallObstacle-v4 --heavy_obstacle --random_gripper --load_path logs/FetchPushWallObstacle-v4_heavy_purerandom/her_sac/custom/final.zip --play

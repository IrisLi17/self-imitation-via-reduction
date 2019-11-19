trap '' 15
# CUDA_VISIBLE_DEVICES=2 python test_universal.py --env FetchPushWallObstacle-v4 --heavy_obstacle --random_gripper --num_timesteps 1e7 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom/her_sac
CUDA_VISIBLE_DEVICES=2 python test_universal.py --env FetchPushWallObstacle-v4 --policy LnMlpPolicy --heavy_obstacle --random_gripper --num_timesteps 1e7 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom/her_sac/ln
CUDA_VISIBLE_DEVICES=2 python test_universal.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --heavy_obstacle --random_gripper --num_timesteps 1e7 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom/her_sac/custom
# ensemble
CUDA_VISIBLE_DEVICES=2 python test_ensemble.py --env FetchPushWallObstacle-v4 --policy EnsembleCustomSACPolicy --batch_size 128 --buffer_size 2e6 --heavy_obstacle --random_gripper --num_timesteps 1e7 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom/her_sac_ensemble

# ensemble
CUDA_VISIBLE_DEVICES=1 python test_ensemble.py --env FetchPushWallObstacle-v4 --load_path logs/FetchPushWallObstacle-v4_heavy_purerandom/her_sac_ensemble/model_0.zip --play --heavy_obstacle --random_gripper
# Visualization.
# CUDA_VISIBLE_DEVICES=2 python test_universal.py --env FetchPushWallObstacle-v4 --heavy_obstacle --random_gripper --load_path logs/FetchPushWallObstacle-v4_heavy_purerandom/her_sac/custom/final.zip --play

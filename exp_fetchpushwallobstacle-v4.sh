trap '' 15
# CUDA_VISIBLE_DEVICES=2 python test_universal.py --env FetchPushWallObstacle-v4 --heavy_obstacle --random_gripper --num_timesteps 1e7 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom/her_sac
CUDA_VISIBLE_DEVICES=2 python test_universal.py --env FetchPushWallObstacle-v4 --policy LnMlpPolicy --heavy_obstacle --random_gripper --num_timesteps 1e7 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom/her_sac/ln
CUDA_VISIBLE_DEVICES=2 python test_universal.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --heavy_obstacle --random_gripper --num_timesteps 1e7 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom/her_sac/custom
# fix z
CUDA_VISIBLE_DEVICES=2 python test_universal.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --heavy_obstacle --random_gripper --num_timesteps 1e7 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/her_sac
# fix z, augment
CUDA_VISIBLE_DEVICES=0 python test_universal_augment.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/final.zip --heavy_obstacle --random_gripper --num_timesteps 1e7 --augment_when_success --n_subgoal 10 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/her_sac_augment
CUDA_VISIBLE_DEVICES=0 python test_universal_augment.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/final.zip --heavy_obstacle --random_gripper --num_timesteps 1e7 --augment_when_success --n_subgoal 10 --random_ratio 0.7 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.7_fixz/her_sac_augment
# fix z, augment no filter
CUDA_VISIBLE_DEVICES=0 python test_universal_augment.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/final.zip --heavy_obstacle --random_gripper --num_timesteps 1e7 --buffer_size 5e6 --n_subgoal 4 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/her_sac_augment_nofilter
CUDA_VISIBLE_DEVICES=0 python test_universal_augment.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/final.zip --heavy_obstacle --random_gripper --num_timesteps 1e7 --buffer_size 2e6 --n_subgoal 1 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/her_sac_augment_nofilter_1
# ensemble
CUDA_VISIBLE_DEVICES=2 python test_ensemble.py --env FetchPushWallObstacle-v4 --policy EnsembleCustomSACPolicy --batch_size 128 --buffer_size 2e6 --heavy_obstacle --random_gripper --num_timesteps 1e7 --log_path logs/FetchPushWallObstacle-v4_heavy_purerandom/her_sac_ensemble

# ensemble
CUDA_VISIBLE_DEVICES=1 python test_ensemble.py --env FetchPushWallObstacle-v4 --load_path logs/FetchPushWallObstacle-v4_heavy_purerandom/her_sac_ensemble/model_0.zip --play --heavy_obstacle --random_gripper
# Visualization.
# CUDA_VISIBLE_DEVICES=2 python test_universal.py --env FetchPushWallObstacle-v4 --heavy_obstacle --random_gripper --load_path logs/FetchPushWallObstacle-v4_heavy_purerandom/her_sac/custom/final.zip --play

# fix z
CUDA_VISIBLE_DEVICES=2 python test_universal.py --env FetchPushWallObstacle-v5 --policy CustomSACPolicy --gamma 0.99 --random_gripper --num_timesteps 1e7 --log_path logs/FetchPushWallObstacle-v5/her_sac
CUDA_VISIBLE_DEVICES=0 python test_universal.py --env FetchPushWallObstacle-v5 --policy CustomSACPolicy --gamma 0.95 --random_gripper --num_timesteps 1e7 --log_path logs/FetchPushWallObstacle-v5/her_sac_1
# random 0.7, augment hack
CUDA_VISIBLE_DEVICES=0 python test_universal_augment.py --env FetchPushWallObstacle-v5 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v5/her_sac_1/final.zip --random_gripper --num_timesteps 1e7 --augment_when_success --hack_augment_time --n_subgoal 4 --random_ratio 0.7 --log_path logs/FetchPushWallObstacle-v5_random0.7/her_sac_augment_hack
CUDA_VISIBLE_DEVICES=1 python test_universal_augment.py --env FetchPushWallObstacle-v5 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v5/her_sac_1/final.zip --load_path logs/FetchPushWallObstacle-v5/her_sac_1/final.zip --random_gripper --num_timesteps 1e6 --augment_when_success --hack_augment_policy --n_subgoal 1 --random_ratio 0.0 --log_path logs/FetchPushWallObstacle-v5_random0.0/her_sac_augment_hack/0

# fix z, augment no filter
CUDA_VISIBLE_DEVICES=0 python test_universal_augment.py --env FetchPushWallObstacle-v5 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v5/her_sac/final.zip --random_gripper --num_timesteps 1e7 --buffer_size 5e6 --n_subgoal 4 --log_path logs/FetchPushWallObstacle-v5/her_sac_augment_nofilter_4
CUDA_VISIBLE_DEVICES=0 python test_universal_augment.py --env FetchPushWallObstacle-v5 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v5/her_sac/final.zip --random_gripper --num_timesteps 1e7 --buffer_size 2e6 --n_subgoal 1 --log_path logs/FetchPushWallObstacle-v5/her_sac_augment_nofilter_1


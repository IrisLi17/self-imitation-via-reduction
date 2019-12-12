CUDA_VISIBLE_DEVICES=0 python test_universal_augment.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/final.zip --heavy_obstacle --random_gripper --num_timesteps 1e7 --augment_when_success --n_subgoal 2 --random_ratio 0.3 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/her_sac_augment/0
CUDA_VISIBLE_DEVICES=0 python test_universal_augment.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/final.zip --heavy_obstacle --random_gripper --num_timesteps 1e7 --augment_when_success --n_subgoal 4 --random_ratio 0.3 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/her_sac_augment/1
CUDA_VISIBLE_DEVICES=0 python test_universal_augment.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/final.zip --heavy_obstacle --random_gripper --num_timesteps 1e7 --augment_when_success --n_subgoal 8 --random_ratio 0.3 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/her_sac_augment/2
CUDA_VISIBLE_DEVICES=0 python test_universal_augment.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/final.zip --heavy_obstacle --random_gripper --num_timesteps 2e7 --augment_when_success --n_subgoal 2 --random_ratio 0.3 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/her_sac_augment/3
CUDA_VISIBLE_DEVICES=0 python test_universal_augment.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/final.zip --heavy_obstacle --random_gripper --num_timesteps 2e7 --augment_when_success --n_subgoal 4 --random_ratio 0.3 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/her_sac_augment/4
CUDA_VISIBLE_DEVICES=0 python test_universal_augment.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/final.zip --heavy_obstacle --random_gripper --num_timesteps 2e7 --goal_selection_strategy future_and_final --augment_when_success --n_subgoal 2 --random_ratio 0.3 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/her_sac_augment/5
CUDA_VISIBLE_DEVICES=0 python test_universal_augment.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/final.zip --heavy_obstacle --random_gripper --num_timesteps 2e7 --augment_when_success --n_subgoal 2 --random_ratio 0.3 --buffer_size 3e6 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/her_sac_augment/6
CUDA_VISIBLE_DEVICES=0 python test_universal_augment.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/final.zip --heavy_obstacle --random_gripper --num_timesteps 2e7 --augment_when_success --n_subgoal 4 --random_ratio 0.3 --buffer_size 5e6 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/her_sac_augment/7

CUDA_VISIBLE_DEVICES=0 python test_universal_augment.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/final.zip --heavy_obstacle --random_gripper --num_timesteps 1e7 --n_subgoal 1 --random_ratio 0.3 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/her_sac_augment_nofilter/0
CUDA_VISIBLE_DEVICES=0 python test_universal_augment.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/final.zip --heavy_obstacle --random_gripper --num_timesteps 1e7 --n_subgoal 2 --random_ratio 0.3 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/her_sac_augment_nofilter/1
CUDA_VISIBLE_DEVICES=0 python test_universal_augment.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/final.zip --heavy_obstacle --random_gripper --num_timesteps 1e7 --n_subgoal 4 --random_ratio 0.3 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/her_sac_augment_nofilter/2
CUDA_VISIBLE_DEVICES=0 python test_universal_augment.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/final.zip --heavy_obstacle --random_gripper --num_timesteps 2e7 --n_subgoal 1 --random_ratio 0.3 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/her_sac_augment_nofilter/3
CUDA_VISIBLE_DEVICES=1 python test_universal_augment.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/final.zip --heavy_obstacle --random_gripper --num_timesteps 2e7 --goal_selection_strategy future_and_final --n_subgoal 2 --random_ratio 0.3 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/her_sac_augment_nofilter/4
CUDA_VISIBLE_DEVICES=1 python test_universal_augment.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/final.zip --heavy_obstacle --random_gripper --num_timesteps 2e7 --goal_selection_strategy future_and_final --n_subgoal 1 --random_ratio 0.3 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/her_sac_augment_nofilter/9
CUDA_VISIBLE_DEVICES=0 python test_universal_augment.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/final.zip --heavy_obstacle --random_gripper --num_timesteps 2e7 --n_subgoal 2 --random_ratio 0.3 --buffer_size 3e6 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/her_sac_augment_nofilter/5
CUDA_VISIBLE_DEVICES=0 python test_universal_augment.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/final.zip --heavy_obstacle --random_gripper --num_timesteps 2e7 --n_subgoal 1 --random_ratio 0.3 --buffer_size 2e6 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/her_sac_augment_nofilter/6
CUDA_VISIBLE_DEVICES=0 python test_universal_augment.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/final.zip --heavy_obstacle --random_gripper --num_timesteps 2e7 --goal_selection_strategy future_and_final --n_subgoal 2 --random_ratio 0.3 --buffer_size 3e6 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/her_sac_augment_nofilter/7
CUDA_VISIBLE_DEVICES=0 python test_universal_augment.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/final.zip --heavy_obstacle --random_gripper --num_timesteps 2e7 --goal_selection_strategy future_and_final --n_subgoal 1 --random_ratio 0.3 --buffer_size 2e6 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/her_sac_augment_nofilter/8

# Double buffer
CUDA_VISIBLE_DEVICES=2 python test_universal_augment.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/final.zip --heavy_obstacle --random_gripper --num_timesteps 1e7 --augment_when_success --double_buffer --n_subgoal 2 --random_ratio 0.3 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/her_sac_augment_double/0
CUDA_VISIBLE_DEVICES=2 python test_universal_augment.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/final.zip --heavy_obstacle --random_gripper --num_timesteps 1e7 --augment_when_success --double_buffer --n_subgoal 4 --random_ratio 0.3 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/her_sac_augment_double/1
CUDA_VISIBLE_DEVICES=3 python test_universal_augment.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/final.zip --heavy_obstacle --random_gripper --num_timesteps 1e7 --augment_when_success --double_buffer --n_subgoal 8 --random_ratio 0.3 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/her_sac_augment_double/2
CUDA_VISIBLE_DEVICES=3 python test_universal_augment.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/final.zip --heavy_obstacle --random_gripper --num_timesteps 1e7 --double_buffer --n_subgoal 2 --random_ratio 0.3 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/her_sac_augment_nofilter_double/1
CUDA_VISIBLE_DEVICES=4 python test_universal_augment.py --env FetchPushWallObstacle-v4 --policy CustomSACPolicy --trained_model logs/FetchPushWallObstacle-v4_heavy_purerandom_fixz/final.zip --heavy_obstacle --random_gripper --num_timesteps 1e7 --double_buffer --n_subgoal 4 --random_ratio 0.3 --log_path logs/FetchPushWallObstacle-v4_heavy_random0.3_fixz/her_sac_augment_nofilter_double/2


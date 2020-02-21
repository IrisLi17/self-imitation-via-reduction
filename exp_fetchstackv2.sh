CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchStack-v2 --random_ratio 0.5 --reward_type sparse --n_object 2 --num_timesteps 4e8 --log_path logs/FetchStack-v2_0.5fix/ppo_attention/sp0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchStack-v2 --random_ratio 0.5 --reward_type sparse --n_object 2 --num_timesteps 4e8 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 8 --log_path logs/FetchStack-v2_0.5fix/ppo_augment/sp0_reuse8

# Adaptive schedule
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchStack-v2 --curriculum --reward_type sparse --n_object 2 --num_timesteps 4e8 --log_path logs/FetchStack-v2_adapt/ppo_attention/sp0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchStack-v2 --curriculum --reward_type sparse --n_object 2 --num_timesteps 4e8 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 4 --log_path logs/FetchStack-v2_adapt/ppo_augment/pretrain_reuse4


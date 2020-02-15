# CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchStack-v1 --num_timesteps 2e8 --log_path logs/FetchStack-v1/ppo/0
# CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchStack-v1 --num_timesteps 2e8 --log_path logs/FetchStack-v1/ppo_attention/0
# CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchStack-v1 --num_timesteps 2e8 --log_path logs/FetchStack-v1/ppo_attention/obj3_sparse
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchStack-v1 --random_ratio 0.5 --reward_type sparse --n_object 2 --num_timesteps 4e8 --log_path logs/FetchStack-v1_0.5fix/ppo_attention/stack2_new/sp0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchStack-v1 --random_ratio 0.5 --reward_type sparse --n_object 2 --num_timesteps 4e8 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 8 --log_path logs/FetchStack-v1_0.5fix/ppo_augment/stack2_new/sp0_reuse8
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchStack-v1 --curriculum --reward_type sparse --n_object 2 --num_timesteps 4e8 --log_path logs/FetchStack-v1_0.7decay/ppo_attention/stack2_new/sp0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchStack-v1 --curriculum --reward_type sparse --n_object 2 --num_timesteps 4e8 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 4 --log_path logs/FetchStack-v1_0.7decay/ppo_augment/stack2_new/sp0_reuse4
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchStack-v1 --curriculum --reward_type sparse --n_object 2 --num_timesteps 4e8 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 8 --log_path logs/FetchStack-v1_0.7decay/ppo_augment/stack2_new/sp0_reuse8

# Adaptive schedule
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchStack-v1 --curriculum --reward_type sparse --n_object 2 --num_timesteps 4e8 --log_path logs/FetchStack-v1_adapt/ppo_attention/stack2_v2/sp0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchStack-v1 --curriculum --reward_type sparse --n_object 2 --num_timesteps 4e8 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 4 --log_path logs/FetchStack-v1_adapt/ppo_augment/stack2_v2/pretrain_reuse4
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchStack-v1 --random_ratio 0.3 --reward_type sparse --n_object 2 --num_timesteps 4e8 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 4 --log_path logs/FetchStack-v1_0.3fix/ppo_augment/stack2_v2/pretrain_reuse4
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchStack-v1 --random_ratio 0.3 --reward_type sparse --n_object 2 --num_timesteps 4e8 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 8 --log_path logs/FetchStack-v1_0.3fix/ppo_augment/stack2_v2/pretrain_reuse8
# CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchStack-v1 --random_ratio 0.4 --reward_type sparse --n_object 3 --num_timesteps 2e8 --log_path logs/FetchStack-v1/ppo_attention/stack3_new/sp0_pp0.4
# CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchStack-v1 --curriculum --reward_type sparse --n_object 3 --num_timesteps 2e8 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 4 --log_path logs/FetchStack-v1/ppo_augment/stack3_new/sp0_reuse4_pp0.7decay
# CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchStack-v1 --random_ratio 0.4 --reward_type sparse --n_object 3 --num_timesteps 2e8 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 4 --log_path logs/FetchStack-v1/ppo_augment/stack3_new/sp0_reuse4_pp0.4
# CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchStack-v1 --random_ratio 0.7 --reward_type dense --n_object 3 --num_timesteps 2e8 --log_path logs/FetchStack-v1/ppo_attention/stack3/ds0
# CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchStack-v1 --random_ratio 0.7 --reward_type dense --n_object 3 --num_timesteps 2e8 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 4 --log_path logs/FetchStack-v1/ppo_augment/stack3/ds0_reuse4
# CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchStack-v1 --random_ratio 0.3 --num_timesteps 2e8 --n_subgoal 4 --parallel --aug_clip 0.0 --reuse_times 4 --log_path logs/FetchStack-v1/ppo_augment/stack2/0_reuse4_subgoal4


CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchStack-v1 --num_timesteps 2e8 --log_path logs/FetchStack-v1/ppo/0
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchStack-v1 --num_timesteps 2e8 --log_path logs/FetchStack-v1/ppo_attention/0
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchStack-v1 --num_timesteps 2e8 --log_path logs/FetchStack-v1/ppo_attention/obj3_sparse
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchStack-v1 --random_ratio 0.7 --reward_type sparse --n_object 2 --num_timesteps 2e8 --log_path logs/FetchStack-v1/ppo_attention/stack2/sp0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchStack-v1 --curriculum --reward_type sparse --n_object 2 --num_timesteps 2e8 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 4 --log_path logs/FetchStack-v1/ppo_augment/stack2_new/sp0_reuse4
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchStack-v1 --random_ratio 0.7 --reward_type dense --n_object 3 --num_timesteps 2e8 --log_path logs/FetchStack-v1/ppo_attention/stack3/ds0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchStack-v1 --random_ratio 0.7 --reward_type dense --n_object 3 --num_timesteps 2e8 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 4 --log_path logs/FetchStack-v1/ppo_augment/stack3/ds0_reuse4
# CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchStack-v1 --random_ratio 0.3 --num_timesteps 2e8 --n_subgoal 4 --parallel --aug_clip 0.0 --reuse_times 4 --log_path logs/FetchStack-v1/ppo_augment/stack2/0_reuse4_subgoal4


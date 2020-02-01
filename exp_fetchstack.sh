CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchStack-v1 --num_timesteps 2e8 --log_path logs/FetchStack-v1/ppo/0
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchStack-v1 --num_timesteps 2e8 --log_path logs/FetchStack-v1/ppo_attention/0
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchStack-v1 --num_timesteps 2e8 --log_path logs/FetchStack-v1/ppo_attention/obj3_sparse
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchStack-v1 --random_ratio 0.3 --num_timesteps 2e8 --log_path logs/FetchStack-v1/ppo_attention/stack2/0
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchStack-v1 --random_ratio 0.3 --num_timesteps 2e8 --n_subgoal 2 --parallel --aug_clip 0.0 --reuse_times 4 --log_path logs/FetchStack-v1/ppo_augment/stack2/0_reuse4
CUDA_VISIBLE_DEVICES=0 python run_ppo_augment.py --env FetchStack-v1 --random_ratio 0.3 --num_timesteps 2e8 --n_subgoal 4 --parallel --aug_clip 0.0 --reuse_times 4 --log_path logs/FetchStack-v1/ppo_augment/stack2/0_reuse4_subgoal4


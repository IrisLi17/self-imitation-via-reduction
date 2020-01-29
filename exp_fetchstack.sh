CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchStack-v1 --num_timesteps 2e8 --log_path logs/FetchStack-v1/ppo/0
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchStack-v1 --num_timesteps 2e8 --log_path logs/FetchStack-v1/ppo_attention/0
CUDA_VISIBLE_DEVICES=0 python run_ppo.py --env FetchStack-v1 --num_timesteps 2e8 --log_path logs/FetchStack-v1/ppo_attention/obj3_sparse

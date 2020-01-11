CUDA_VISIBLE_DEVICES=1 python run_ppo.py --env FetchPickAndPlace-v1 --num_timesteps 1e8 --log_path logs/FetchPickAndPlace-v1/ppo/0
CUDA_VISIBLE_DEVICES=2 python run_ppo.py --env FetchOpenCloseBox-v1 --num_timesteps 1e8 --log_path logs/FetchOpenCloseBox-v1/ppo/0


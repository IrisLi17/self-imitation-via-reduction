# Training.
# CUDA_VISIBLE_DEVICES=0 mpirun -n 8 python test_her.py --env FetchPushObstacle-v1 --num_timesteps 5e6 --log_path ./logs/FetchPushObstacle-v1/her
CUDA_VISIBLE_DEVICES=0 python test_her.py --env FetchPushWallObstacle-v1 --log_path logs/FetchPushWallObstacle-v1/her_sac --num_timesteps 6e6

# Visualize trained agent.
# CUDA_VISIBLE_DEVICES=0 python test_her.py --env FetchPushObstacle-v1 --play --load_path ./logs/FetchPushObstacle-v1/her/final

# Plot learning curve.
# python plot_log.py --env FetchPushObstacle-v1 --log_path ./logs/FetchPushObstacle-v1/her --xaxis timesteps

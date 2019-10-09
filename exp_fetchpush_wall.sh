# Training.
CUDA_VISIBLE_DEVICES=0 mpirun -n 8 python test_her.py \
--env FetchPushWall-v1 \
--log_path ./logs/FetchPushWall-v1/her

# Visualize trained agent.
# CUDA_VISIBLE_DEVICES=0 python test_her.py \
# --env FetchPushWall-v1 --play \
# --load_path ./logs/FetchPushWall-v1/her/final

# Plot learning curve.
# python plot_log.py --env FetchPushWall-v1 --log_path ./logs/FetchPushWall-v1/her --xaxis timesteps

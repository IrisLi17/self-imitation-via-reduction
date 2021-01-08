# 2 boxes
CUDA_VISIBLE_DEVICES=1 python run_her.py --env FetchStack-v1 --num_timesteps 1e6 --curriculum --num_workers 32 --policy AttentionPolicy --reward_type sparse --n_object 2 --priority --log_path logs/FetchStack-v1_adapt/her_sac_32workers/2obj/0
CUDA_VISIBLE_DEVICES=0 python run_her_augment.py --env FetchStack-v1 --num_timesteps 1e6 --curriculum --num_workers 32 --policy AttentionPolicy --reward_type sparse --n_object 2 --imitation_coef 0.1 --priority --log_path logs/FetchStack-v1_adapt/her_sac_sir_32workers/2obj/0
CUDA_VISIBLE_DEVICES=1 python run_her.py --env FetchStack-v1 --num_timesteps 1e6 --curriculum --num_workers 32 --policy AttentionPolicy --reward_type sparse --n_object 2 --sil --sil_coef 0.1 --priority --log_path logs/FetchStack-v1_adapt/her_sac_sil_32workers/2obj/0
CUDA_VISIBLE_DEVICES=1 python run_her.py --env FetchStack-v1 --num_timesteps 1e6 --curriculum --num_workers 32 --policy AttentionPolicy --reward_type dense --n_object 2 --priority --log_path logs/FetchStack-v1_adapt/her_sac_ds_32workers/2obj/0
# 3 boxes
CUDA_VISIBLE_DEVICES=1 python run_her.py --env FetchStack-v1 --num_timesteps 3.5e6 --curriculum --num_workers 32 --policy AttentionPolicy --reward_type sparse --n_object 3 --priority --log_path logs/FetchStack-v1_adapt/her_sac_32workers/3obj/0
CUDA_VISIBLE_DEVICES=0 python run_her_augment.py --env FetchStack-v1 --num_timesteps 3.5e6 --curriculum --num_workers 32 --policy AttentionPolicy --reward_type sparse --n_object 3 --imitation_coef 0.1 --priority --log_path logs/FetchStack-v1_adapt/her_sac_sir_32workers/3obj/0
CUDA_VISIBLE_DEVICES=1 python run_her.py --env FetchStack-v1 --num_timesteps 3.5e6 --curriculum --num_workers 32 --policy AttentionPolicy --reward_type sparse --n_object 3 --sil --sil_coef 0.1 --priority --log_path logs/FetchStack-v1_adapt/her_sac_sil_32workers/3obj/0
CUDA_VISIBLE_DEVICES=1 python run_her.py --env FetchStack-v1 --num_timesteps 3.5e6 --curriculum --num_workers 32 --policy AttentionPolicy --reward_type dense --n_object 3 --priority --log_path logs/FetchStack-v1_adapt/her_sac_ds_32workers/3obj/0


# experimental
CUDA_VISIBLE_DEVICES=0 python run_her.py --env FetchStack-v1 --num_timesteps 3.5e6 --curriculum --num_workers 32 --policy RelationalPolicy --reward_type sparse --n_object 3 --priority --log_path logs/FetchStack-v1_adapt/her_sac_32workers/relational/3obj/0
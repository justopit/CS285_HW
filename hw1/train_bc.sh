# 基线
# for env in HalfCheetah-v4 
# do
#     len=${#env}
#     pre=${env:0:$(($len-3))}
#     python cs285/scripts/run_hw1.py \
#     --expert_policy_file cs285/policies/experts/${pre}.pkl \
#     --env_name ${env} --exp_name bc_${env} --n_iter 1 \
#     --expert_data cs285/expert_data/expert_data_${env}.pkl \
#     --video_log_freq -1 --eval_batch_size 5000 
# done


# # Behaviour Clone
# # 增大BatchSize
for env in HalfCheetah-v4 Hopper-v4 Walker2d-v4 Ant-v4
do
    len=${#env}
    pre=${env:0:$(($len-3))}
    python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/${pre}.pkl \
    --env_name ${env} --exp_name bc_${env} --n_iter 1 \
    --expert_data cs285/expert_data/expert_data_${env}.pkl \
    --video_log_freq 10 --eval_batch_size 5000 \
    --num_agent_train_steps_per_iter 50000
done
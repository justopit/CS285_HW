for lambda in 0 0.95 0.98 0.99 1
do
python cs285/scripts/run_hw2.py \
--env_name LunarLander-v2 --ep_len 1000 \
--discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 \
--use_reward_to_go --use_baseline --gae_lambda ${lambda} \
--exp_name lunar_lander_lambda${lambda}
done
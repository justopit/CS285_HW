for lambda in 0.95 0.98 0.99
do
    for seed in $(seq 1 1); do
        python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 -n 100 \
        --exp_name discount_gae_${lambda}_pendulum_default_s$seed \
        -rtg --use_baseline -na \
        --batch_size 5000 \
        --gae_lambda ${lambda} \
        --discount 0.98 \
        --seed $seed
    done
done

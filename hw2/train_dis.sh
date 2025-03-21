for discount in 0.98
do
    for seed in $(seq 1 2); do
        python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 -n 100 \
        --exp_name discount_${discount}_pendulum_default_s$seed \
        -rtg --use_baseline -na \
        --batch_size 5000 \
        --discount $discount \
        --seed $seed
    done
done
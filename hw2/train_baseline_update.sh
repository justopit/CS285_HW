for bgs in 15 30
do
    for seed in $(seq 1 1); do
        python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 -n 100 \
        --exp_name discount_bgs_${bgs}_pendulum_default_s$seed \
        -rtg --use_baseline -na \
        -bgs $bgs \
        --batch_size 5000 \
        --discount 0.98 \
        --seed $seed
    done
done

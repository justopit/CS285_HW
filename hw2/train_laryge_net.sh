for layer in 3 4
do
    for seed in $(seq 1 1); do
        python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 -n 100 \
        --exp_name udl_discount_layer_${layer}_pendulum_default_s$seed \
        -rtg --use_baseline -na \
        --batch_size 5000 \
        --discount 0.98 \
        -l $layer \
        -lr 7e-3 -blr 7e-3 \
        -udl \
        --seed $seed
    done
done

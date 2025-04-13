for layer in 2 3 4
do
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/cartpole-layer${layer}.yaml --seed 1
done

# Experiemnt 1(CartPole)
+ 小BatchSize下(average return vs. number of environment steps)的对比
![alt text](image.png)
+ 大BatchSize下(average return vs. number of environment steps)的对比

问题回答
+ 在没有 advantage normalization的时候，the trajectorycentric one, or the one using reward-to-go哪一个好？
reward-to-go 最后收敛到200
+ advantage normalization help?
在trajectory centric one的时候，normalization在收敛速度和最终reward上都有帮助，无论batchsize小还是大
在reward-to-go的时候，normalization可以帮助更快收敛。然而其在小batchsize的时候，后期会出现震荡。大batchsize的时候，则不会有这种现象。
+ batch size 对效果有影响吗？
有影响。特别是在收敛速度和reward的稳定。其都有积极效应。

# Experiment 2( Using a Neural Network Baseline)
代码如下，baseline + 100 epoch达不到300 reward的效果。
因此直接开大到200 epoch，轻松到达500的reward。
```bash
# No baseline
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
-n 200 -b 5000 -rtg --discount 0.95 -lr 0.01 \
--exp_name cheetah
# Baseline
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
-n 200 -b 5000 -rtg --discount 0.95 -lr 0.01 \
--use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline
```
+ 训练Loss曲线图
![alt text](image-1.png)
最后在15左右的loss震荡

+ Eval Return曲线图
![alt text](image-2.png)

+ 调整baseline学习
降低baseline gradient steps让baseline网络学习的更慢了.

+ 增加norm
增加norm之后，reward到了800左右，比之前的500多很多。
![alt text](image-3.png)
更值得说的是,baseline loss更大了。很奇怪哈。我的理解是polciy更新更加频繁，其policy的reward上涨的速度快，导致其value network的loss不停的涨。因为value network的预估出来的reward分布其实是过去的policy下的，而非现在的。
![alt text](image-4.png)

## Expriment 3 (LunarLander-v2)
使用默认配置的话，只变化$\lambda$的话，$\lambda = 0.99$的效果最好。最后差不多在160左右大幅度波动
![alt text](image-5.png)

+ 当$\lambda = 0$的时候， 就是纯粹的TD，其估计出来的reward bias很大，虽然low variance，因此可以看到它学习的很慢。当$\lambda = 1$的时候，就是蒙塔卡洛-基线，虽然无bias，但是高方差。因此其学习的效果不是最好。$\lambda$为0.99的效果才是最好的。
# CS 285 Hw 1
## Analysis Soltuion
### 1 求解真实状态和专家状态分布的L1距离的bound
Google一下相关的答案，看上去都是错的。在此，给出我认为是正确解决的方法。
首先采取课件的类似思路，对$p_{\pi_{\theta}}(s_t)$进行贝叶斯拆解，假设到$s_t$之前，真实动作偏离专家轨迹，也就是真实动作偏离专家动作的概率为$p_e(t)$, 其与$t$有关，但是与$s_t$无关。那么
$p_{\pi_{\theta}}(s_t) = p_e(t) \times p_{mistake}(s_t) + (1-p_e(t)) \times p_{\pi^{*}}(s_t) $。因此对不等式进行求解
$$
\sum_{s_t}| p_{\pi_{\theta}}(s_t) - p_{\pi^{*}}(s_t)| \leq \sum_{s_t} p_e(t) |p_{mistake}(s_t) - p_{\pi^{*}}(s_t)| \\
\leq p_e(t) (\sum_{s_t}p_{mistake}(s_t) + p_{\pi^{*}}(s_t))
= 2 p_e(t)
$$
接下来求解$p_e(t)$。$p_e(t)$可以认为是第$i$时刻才发生错误动作的概率的并事件。因此
$$
p_e(t) = P[\cup_i \text{wrong action until $i$ time}]  \\
\leq \sum_{i} P(\text{wrong action until $i$ time}) \\
\leq \sum_ i \mathbb E_{p_{\pi^*(s_i)}}\pi_\theta(a_i \neq \pi^*(s_i) | s_i)
\leq T \epsilon \\
$$
因此最后得到
$$
\sum_{s_t}| p_{\pi_{\theta}}(s_t) - p_{\pi^{*}}(s_t)| \leq 2 T \epsilon
$$
### 2 求解Reward的bound
很简单，直接展开期望为sum，然后根据上面的分布bound对其求和即可。证明如下
$$
|J(\pi^*) - J(\pi_\theta)| = \sum_t \mathbb E_{p_{\pi^*}(s_t)}r(s_t) - \sum_t \mathbb E_{p_{\pi_\theta}(s_t)}r(s_t) \\ \leq \sum_t  \sum_{s_t}| p_{\pi_{\theta}}(s_t) - p_{\pi^{*}}(s_t)|R_{max}
$$
然后a)，b)易证

## Edit Code
### 1.行为克隆
使用高斯分布下的损失函数。在Ent环境中测试了一下，对比平方的MSE损失函数，平均reward从4600到4700。
```python
    def forward(self, observation: torch.FloatTensor) -> Any:
        """
        Defines the forward pass of the network

        :param observation: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
        # TODO: implement the forward pass of the network.
        # You can return anything you want, but you should be able to differentiate
        # through it. For example, you can return a torch.FloatTensor. You can also
        # return more flexible objects, such as a
        # `torch.distributions.Distribution` object. It's up to you!
        return [self.mean_net(observation), self.logstd]

    def update(self, observations, actions):
        """
        Updates/trains the policy

        :param observations: observation(s) to query the policy
        :param actions: actions we want the policy to imitate
        :return:
            dict: 'Training Loss': supervised learning loss
        """
        # TODO: update the policy and return the loss
        self.optimizer.zero_grad()

        pred = self.forward(observations)
        pred_ac = pred[0]
        stds = torch.exp(self.logstd)
        loss = ((pred_ac - actions) ** 2 / stds[None,:]).sum(axis=-1).mean() + self.logstd.sum()
        loss.backward()
        self.optimizer.step()

        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': loss,
        }
```

steps为50000。训练的reward基本和专家reward基本差不多了。
| Metric                                 | HalfCheetah-v4 | Hopper-v4 | Walker2d-v4 | Ant-v4  |
|----------------------------------------|-----------------|-----------|--------------|---------|
| Eval Average Return                    | 4016.85         | 3382.31   | 5314.56      | 4721.99 |
| Eval Std Return                        | 70.96           | 747.13    | 32.70        | 67.85   |
| Eval Max Return                        | 4137.79         | 3725.99   | 5353.25      | 4841.49 |
| Eval Min Return                        | 3945.63         | 1711.75   | 5275.39      | 4646.11 |
| Eval Average Ep Len                    | 1000.00         | 913.50    | 1000.00      | 1000.00 |
| Train Average Return                   | 4034.80         | 3717.51   | 5383.31      | 4681.89 |
| Train Std Return                       | 32.87           | 0.35      | 54.15        | 30.71   |
| Train Max Return                       | 4067.67         | 3717.87   | 5437.46      | 4712.60 |
| Train Min Return                       | 4001.93         | 3717.16   | 5329.16      | 4651.18 |
| Train Average Ep Len                   | 1000.00         | 1000.00   | 1000.00      | 1000.00 |
| Training Loss                          | -32.34          | -20.30    | -31.51       | -64.10  |
| Train Envsteps So Far                  | 0               | 0         | 0            | 0       |
| Time Since Start                       | 13.74           | 14.32     | 14.26        | 14.97   |
| Initial Data Collection Average Return | 4034.80         | 3717.51   | 5383.31      | 4681.89 |

### 2.Dagger
Dagger的运行结果如下，只在Hopper-v4上有观测到模型优势，对比行为克隆。
| Metric                                 | HalfCheetah-v4 | Hopper-v4 | Walker2d-v4 | Ant-v4  |
|----------------------------------------|-----------------|-----------|--------------|---------|
| Eval Average Return                    | 4062.33         | 3718.85   | 5255.06      | 4699.59 |
| Eval Std Return                        | 78.59           | 1.40      | 166.90       | 54.49   |
| Eval Max Return                        | 4175.53         | 3721.48   | 5416.17      | 4777.15 |
| Eval Min Return                        | 3954.29         | 3717.64   | 4956.33      | 4621.74 |
| Eval Average Ep Len                    | 1000.00         | 1000.00   | 1000.00      | 1000.00 |
| Train Average Return                   | 4058.48         | 3717.79   | 5428.42      | 4618.28 |
| Train Std Return                       | 62.69           | 2.35      | 21.13        | 120.61  |
| Train Max Return                       | 4175.65         | 3720.70   | 5463.73      | 4800.54 |
| Train Min Return                       | 4006.09         | 3714.60   | 5397.71      | 4463.63 |
| Train Average Ep Len                   | 1000.00         | 1000.00   | 1000.00      | 1000.00 |
| Training Loss                          | -31.24          | -14.79    | -22.10       | -56.63  |
| Train Envsteps So Far                  | 45000           | 46245     | 46549        | 45000   |
| Time Since Start                       | 10.28           | 14.05     | 14.67        | 21.53   |


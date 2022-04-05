# MOEA/D 全文阅读

> Highlights

1. 本文提出了一种新的多目标进化算法（MOEA/D）

这种算法基于分解的策略。这种算法中将多目标游湖啊问题分解成一系列的标量优化子问题（scalar optimization subproblems）。每一个子问题（subproblem）仅仅使用它的临近子问题（neighboring subproblems）进行优化，和MOGLS和NSGA-II对比，这样的方式具有**更小的时间复杂度**

2. 实验结果也验证了MOEA/D出色的性能

使用了简单的分解策略的MOEA/D算法相比于MOGLS和NSGA-II算法在多目标0-1背包问题和连续多目标优化问题上**性能表现接近或者更优**

3. MOEA/D使用的场景

MOEA/D using objective normalization can deal with disparately-scaled objectives, and MOEA/D with an advanced decomposition method can generate a set of very evenly distributed solutions for 3-objective test instances. 

4. 本文还针对MOEA/D 进行了那些探索？

本文还探索了具有小种群的MOEA/D的性能，MOEA/D的泛用性（scalebility）和MOEA/D的敏感性

5. MOEA/D的特性（features）
    - MOEA/D提供了一种简单但是高效的分解策略
    - MOEA/D同时完成N个标量优化，而不是直接处理MOP问题
    - MOEA/D相较于NSGA-II和MOGLS的时间复杂度更低
## Definitions

> Definition of Problem

本文中的问题定义为：
$$
maximize \, F(x) = \left( f_1(x), \dots, f_m(x)\right)^T \\
subject \, to \, x \in \Omega
$$
$F(x)$表示一个映射$F:\Omega \rightarrow R^m$，所有的目标都是连续的，本文的MOP问题是一个**连续**的MOP问题。

> Decomposition of Multiobjective Optimization

MOEA/D的分解是指将当前的多目标问题分解成一系列的标量优化问题（scalar optimization，这里应该是指单目标优化问题）。有三种主要的分解方式：权重和，切比雪夫，PBI。使用的比较多的是切比雪夫和PBI。

![切比雪夫](https://s1.ax1x.com/2022/04/05/qO1paq.jpg)

![PBI](https://s1.ax1x.com/2022/04/05/qOlqG8.jpg)

## The Framework of Multiobjective Evolutionary Algorithm Based on Decomposition (MOEA/D)

> Weight Vectors

MOEA/D算法首先需要$N$条均匀分布的权重向量， 记作$\lambda_1, \dots,\lambda_N$，$z^*$是参考点，参考点的选取（取最大？还是取最小），取决于当前的问题。
如果是一个最小化问题：
$z_* = min \left\{f_i(x) | x \in \Omega \right\}$

> A Neighborhood of Weight Vector

权重向量$\lambda_i$的邻居（neighbor）是其最近的权重向量的集合。【最近】的衡量在原文中使用的是欧氏距离

![qO8R2T.png](https://s1.ax1x.com/2022/04/05/qO8R2T.png)

**Discussions and Problems**：
1. 为什么每一个点都需要对临近的权重向量进行计算？

首先注意一个细节，生成的自带的个体是$N$，同时也生成了$N$个权重向量，而且种群是由每一个子问题的最好的解所构成的。在选择生成子代个体$y$的时候是通过父代个体$x^k, x^l$生成，$x^k, x^l$被视作是子问题$i$的相邻的子问题，因此生成的$y$也可以看做子问题$i$的解。

2. 权重向量是如何进行均匀分布的呢？

这个问题在本文中没有详细说明

3. MOEA/D如何维持种群的多样性

从文中讨论部分的说明来看，MOEA/D的多样性和其分解（划分权重向量）的方式有着联系。因为MOEA/D在每轮迭代中会优化其$N$个子问题，如果划分权重向量均匀，解也会分布均匀。



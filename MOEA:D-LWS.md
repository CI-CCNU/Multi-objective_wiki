# Localized weighted sum method for many-objective Optimization （Rui Wang, Zhongbao Zhou, Hisao Ishibuchi， Year 2016, From IEEE TEVC）

> Hightlights

1. 本文提出了一种新型的分解策略（MOEA/D-LWS）。在这种算法中在局部搜索阶段使用了Weighted sum方法。
2. 本文中的邻居（neighbor）的定义基于超圆锥（hypercone）。

# Definition and Keywords
本文中使用的一些关键性的定义以及涉及到的一些重要的观点如下

**观点**
本文通过引用一些研究表权重和分解方法的选择对于MOEA/D的性能有着非常重要的影响。（本文中分解方法被称作scalarizing methods，两者意思接近）
1. 对于权重的指定主要影响了逼近的PF（approximated）的分布。对于具有非线性形状的（convex或者non-convex）的MOP，均匀分布的权重不一定可以得到均匀分布的解。
2. 对于分解方法的指定主要影响了搜索效率
- 基于权重的切比雪夫方法可以对于凸和非凸PF都能找到解，但是权重和方法不行
- 权重和方法可以获得更好的收敛效果，但是基于权重的切比雪夫方法不行。

**Ideal Point**
目标向量$z^* = (z_1^*,\dots,z_m^*)$，其中$z_i^*$表示$f_i(x)$的下界表示的值。

**Utopian Point**
$z^u$表示一个不可行解，其中$z^u_i = z_1^* - \varepsilon_i$,$\varepsilon_i > 0$表示一个很小，但是在计算上仍然显著的标量。

**Nadir Point**
目标向量$z^{nad} = (z^{nad}_1,\dots,z^{nad}_m)$，$z^{nad}_i$表示$f(x)$的上确界，$x\in PS$ （注意Ideal point没有这个要求）。

当$z_i^*$和$z_i^{nad}$不知道时，我们可以使用当前（best so far）所有的非支配解中**最小的**和**最大的**$f_i$作为$z_i^*$和$z_i^{nad}$的接近值（approximation）。


**关键词**

- The neighborhood size of a LWS

这部分的大小使用了超圆锥定义，超圆锥的顶角等于他最近的$m$个权重的角度的平均值。

- Decomposition approaches: $L_p$ scalarizing method
本文中提到的分解方法：
$$
g^{wd}(x|w, p) = \left(\mathop{\sum}\limits^m_{i=1}\lambda_i(f_i(x)-z_i^u)^p\right)^{\frac{1}{p}} , p \ge 1\\
\lambda_i = \left(\frac{1}{w_i}\right)
$$

常用的weighted sum方法和切比雪夫方法可以看做是这个方法设置$p=1$和设置$p=\infty$的特殊情况:
$g^{ws}(x|w) = \mathop{\sum}\limits^m_{i=1}(\lambda_i(f_i(x)-z^u_i))$

$g^{ch}(x|w) = \mathop{max}\limits^m_{i=1}(\lambda_i(f_i(x)-z^u_i))$

本文中对于权重和方法做了一个总结：
1. 权重和方法对于非凸问题（non-convex）问题不能找到解
2. 权重和方法对于切比雪夫方法有着更高的搜索效率

## Views
下面对于本文中提出的一些重要的观点进行记录，这些观点有些作为MOEA/-LWS算法重要的支撑
1. 基于Pareto-dominance的的适应度衡量方法对于很多many-objective问题不能有很好的解[4][72]
2. 对于一些many-objective的问题，比如[45][73]中，使用weighted-sum获取的结果比使用切比雪夫方法获取的结果要更好

## MOEA/D-LWS

**对于权重和方法和切比雪夫方法的效率分析**

如图可以看出，图中使用箭头指出了等势线的分布，等势线都是垂直于权重的直线，使用权重和方法（权重向量为(0.5,0.5)）**差不多**将搜索空间分成了一半。但是在图二中，当增加$p$的时候，分割的部分越来越小。
![LkWoLQ.png](https://s1.ax1x.com/2022/04/10/LkWoLQ.png)
从以上的图中可以看出来以下结论：

使用权重和方法找出一个更好的解的概率是1/2，使用其他的$L_p$分解方法的概率小于1/2，对于切比雪夫方法是$\frac{1}{2^m}$其中的$m$是问题的数量，当问题数量也增加时，对于$p$很大的分解方法而言，找到更好的解的概率变的非常小。

因此，对于many-objective问题而言，使用切比雪夫方法比较困难，效率不高。

![Lk4taj.png](https://s1.ax1x.com/2022/04/10/Lk4taj.png) 本文在ZDT1测试机上进行了实验，发现使用权重和方法的PF相对比使用切比雪芙的PF更加好

**Methodology: the localized weighted sum method**

Q1: 为什么要使用hypercone而不是hypercylinder
A1: 使用超圆锥有两方面的好处，如图所示
1. 超圆锥相对于超圆柱，有更大的搜索空间可以在进化的过程中提供了一个更大的搜索空间，可以在**搜索早期**找到更多的潜在解（关于这一点我是这样理解的，同样半径的圆柱的体积是圆柱的三倍，但是这里不能假设是相同半径）
2. 在搜索的后续阶段，应该逐渐减小搜索区域，使用超圆锥也可以满足这个需求

![LAdLsx.png](https://s1.ax1x.com/2022/04/10/LAdLsx.png)
另外，使用了这种方法的权重和算法值计算了在使用权重向量$w_i$标记的超圆锥区域内的潜在解，之外的解的值被设计为$\infty$，超圆锥的顶角通过如下的公式计算：
$\Theta_i = \frac{\sum^{j=m}_{j=1}\theta^{ww}_{ij}}{m}$

在上图中，按照传统的权重和方法得到的$x_3$应该是最好的，但是在超圆锥中$x_2$应该看做和$x_3$一样好

$\theta^{ww}_{ij}$表示相邻最近的权重向量$j$和$i$之间的角度。因为权重向量基本上是均匀分布的，所以这些角度应该是相同的。

## Incorporation of the LWS method into MOEA/D

[![LADmFO.png](https://s1.ax1x.com/2022/04/10/LADmFO.png)](https://imgtu.com/i/LADmFO)
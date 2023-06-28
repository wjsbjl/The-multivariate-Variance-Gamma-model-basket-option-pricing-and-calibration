# The multivariate Variance Gamma model basket option pricing and calibration 论文复刻
## 文档目标
这里想找一篇期权相关论文进行复刻，以完成对于期权定价、数值积分、数值求解等的主题学习。

## 文章简介
本文档复刻内容是违约期权领域的文献。文章通过设置上限和下限，实现对一篮子期权的近似估计。

## 代码笔记
230628 蒙特卡洛部分通过np.outer进行修正，使得原本for-loop循环变成直接向量化运算。这种指令并行的方法让原有的5小时模拟时间缩减为6秒。
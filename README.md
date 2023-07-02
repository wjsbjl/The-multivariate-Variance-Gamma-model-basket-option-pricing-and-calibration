# The multivariate Variance Gamma model basket option pricing and calibration 论文复刻
## 文档目标
这里想找一篇期权相关论文进行复刻，以完成对于资产定价、期权定价、数值积分、数值求解、蒙特卡洛、指令并行等的主题学习。

## 文章简介
本文档复刻内容是违约期权领域的文献。文章通过设置上限和下限，实现对一篮子期权的近似估计。

## 公式整理
Parameter  
$r = 3\%$  
$\rho_{i,j}=0, ~for~i \ne j ~and~ i, j = 1, 2, 3$  
$N = 10^6$

| 1| 1 | 2 | 3|
|- |- |- |- |
|$\mu$|-0.1368 | -0.056 | -0.1984 |
|$\sigma$| 0.1099 | 0.1677 | 0.0365 |
| $X(0)$ | 100 | 200 | 300 |
| $\omega$ | 1/3 | 1/6 | 1/9 |

$$\varepsilon[K]=\frac{|\overline{C}[K] - C^{sim}[K]|}{C^{sim}[K]}$$

$$\lambda_i = w_j X_j(0)\exp\{(r-q_j+\omega_j)T+\mu_j y+\frac{\sigma_j^2 y}{2}\}$$  
$$\sigma_{\Lambda_y}^2=\sum_{i=1}^n\lambda_i^2\sigma_i^2+\sum_{i=1, i\ne j}^n \lambda_i \lambda_j \sigma_i \sigma_j \rho_{i,j}$$  
$$r_i=\frac{\sum_{i=1}^n\lambda_j\sigma_j\rho_{i,j}}{\sigma_{\Lambda_y}}$$

var upper  

$$
\begin{align*}
& \mathrm{Var}\left[S_y^c\right]=\sum_{i=1}^n \sum_{j=1}^n w_i w_j X_i(0) X_j(0) \\
& \quad e^{2 r T+\left(\omega_i-q_i+\omega_j-q_j\right) T+\left(\mu_i+\mu_j\right) y+\frac{\sigma_i^2+\sigma_j^2}{2} y}\left(e^{\sigma_i \sigma_j y}-1\right) .
\end{align*}
$$

var lower
$$
\begin{align*}
\mathrm{Var}  {\left[S_y^l\right]}&={\sum_{i=1}^n \sum_{j=1}^n w_i w_j X_i(0) X_j(0) } \\
&\quad \times \mathrm{e}^{2 r T+\left(\omega_i-q_i+\omega_j-q_j\right) T+\left(\mu_i+\mu_j+\frac{1}{2}\left(\sigma_i^2\left(1-r_i^2\right)+\sigma_j^2\left(1-r_j^2\right)\right)\right) y} \\
&\quad \times e^{\frac{1}{2}(r^2_i\sigma^2_i+r^2_j\sigma^2_j)y}(e^{r_ir_j\sigma_i\sigma_jy}-1)\\
& {=\sum_{i=1}^n \sum_{j=1}^n w_i w_j X_i(0) X_j(0) } \\
&\quad \times \mathrm{e}^{2 r T+\left(\omega_i-q_i+\omega_j-q_j\right) T+\left(\mu_i+\mu_j+\frac{1}{2}\left(\sigma_i^2+\sigma_j^2\right)\right) y} \\
& \quad\times (e^{r_ir_j\sigma_i\sigma_jy}-1)
\end{align*}
$$

权重是$w$  
$$\lambda_i = w_j X_j(0)\exp\{(r-q_j+\omega_j)T+\mu_j y+\frac{\sigma_j^2 y}{2}\}$$  
$$\sigma_{\Lambda_y}^2=\sum_{i=1}^n\lambda_i^2\sigma_i^2+\sum_{i=1, i\ne j}^n \lambda_i \lambda_j \sigma_i \sigma_j \rho_{i,j}$$  
$$r_i=\frac{\sum_{i=1}^n\lambda_j\sigma_j\rho_{i,j}}{\sigma_{\Lambda_y}}$$

var Sy
$$
\begin{align*}
	\mathrm{Var}[{_y}S]=\sum_{i=1}^{n} \sum_{j=1}^{n} w_{i} w_{j} X_{i}(0) X_{j}(0)  
	\mathrm{e}^{2 r T+\left(\omega_{i}-q_{i}+\omega_{j}-q_{j}\right) T+\left(\mu_{i}+\mu_{j}\right) y} 
	e^{\frac{\sigma_i^2+\sigma_j^2+\rho_{i,j}\sigma_i\sigma_j
	}{4-\rho^2_{i,j}}2(1-\rho^2_{i,j})y}
\end{align*}
$$

z_y
$$z_y = \frac{Var[S_y^c]-Var[S_y]}{Var[S_y^c]-Var[S_y^l]}$$  

概率密度函数
$$f_{\bar{S}}(K)=\mathrm{e}^{r T} \frac{\partial \bar{C}^2[K]}{\partial K^2}$$

## Timeline
230702 修正upper bound计算问题（检查发现是需要先加权后算积分），但部分结果仍然与原论文存在很大差距。  
230629 目前感觉蒙特卡洛结果没什么问题，理论价格计算存在很大偏误，可能是upper bound计算存在问题。  
230628 蒙特卡洛部分通过np.outer进行修正，使得原本for-loop循环变成直接向量化运算。这种指令并行的方法让原有的5小时模拟时间缩减为6秒。
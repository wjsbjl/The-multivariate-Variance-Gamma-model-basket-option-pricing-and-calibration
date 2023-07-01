# 环境配置
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from tqdm import trange
from scipy.stats import norm
import scipy.integrate as sci
import math
from scipy.stats import norm
from scipy.optimize import fsolve # 解决非线性方程组的数值求解问题
from scipy.optimize import bisect # 解决非线性方程组的数值求解问题
from scipy.stats import gamma # gamma分布的概率密度
from tqdm import tqdm

## table2
### 定义函数
def C_approximate_calculate_func(T, v, K, sigma, mu, r, q):
    def f(x):
        omega = 1/v * np.log(1 - 1/2 * (sigma ** 2) * v - mu * v)
        K_i = K / weight.copy() / len(omega)
    
        d_1_upper = (np.log(X_0/K_i) + (r - q + omega) * T + mu * x + sigma ** 2 * x)/(sigma * np.sqrt(x))
        d_2_upper = d_1_upper - sigma * np.sqrt(x)
        exp_upper = np.exp((r - q + omega ) * T + (mu + sigma ** 2 / 2) * x)
        X_price_upper = ((X_0 * exp_upper * norm.cdf(d_1_upper) - K_i * norm.cdf(d_2_upper)))
        S_upper = np.dot(X_price_upper, weight)
        
        lambda_i = weight * X_0 * np.exp((r - q + omega) * T + (mu + sigma ** 2 / 2) * x)
        sigma2Lambda = sum(sum(np.outer(lambda_i * sigma, lambda_i * sigma) * rho))
        if sigma2Lambda == 0:
            r_i = np.sign(np.dot((lambda_i * sigma),rho))
        else:
            r_i = np.dot((lambda_i * sigma),rho)/np.sqrt(sigma2Lambda)
        r_i = r_i.clip(-1,1)
        zero_indices = np.where(r_i == 0)[0]
        r_i[zero_indices] = 0.1
        d_1_lower = (np.log(X_0/K_i) + (r - q + omega) * T + mu * x + sigma ** 2 * x / 2 * (1 + r_i ** 2))/(sigma * np.sqrt(x) * r_i)
        d_2_lower = d_1_lower - sigma * np.sqrt(x) * r_i
        exp_lower = np.exp((r + omega - q) * T + (mu + sigma ** 2 / 2) * x)
        X_price_lower = ((X_0 * exp_lower * norm.cdf(d_1_lower) - K_i * norm.cdf(d_2_lower)))
        S_lower = np.dot(X_price_lower, weight)

        var_upper_part1 = np.outer(X_0 * weight, X_0 * weight)
        var_upper_part2 = np.exp(2 * r * T)
        var_upper_part3 = np.outer(np.exp((weight - q) * T + mu * x + sigma ** 2 / 2 * x),np.exp((weight - q) * T + mu * x + sigma ** 2 / 2 * x))
        var_upper_part4 = np.exp(np.outer(sigma,sigma) * x) - 1
        var_upper = sum(sum(var_upper_part1 * var_upper_part2 * var_upper_part3 * var_upper_part4))

        var_lower_part1 = np.outer(X_0 * weight, X_0 * weight)
        var_lower_part2 = np.exp(2 * r * T)
        var_lower_part3 = np.outer(np.exp((weight - q) * T + mu * x + sigma ** 2 / 2 * x),np.exp((weight - q) * T + mu * x + sigma ** 2 / 2 * x))
        var_lower_part4 = np.exp(np.outer(sigma * r_i, sigma * r_i) * x) - 1
        var_lower = sum(sum(var_lower_part1 * var_lower_part2 * var_lower_part3 * var_lower_part4))

        var_part1 = np.outer(X_0 * weight, X_0 * weight)
        var_part2 = np.exp(2 * r * T)
        var_part3 = np.outer(np.exp((weight - q) * T + mu * x),np.exp((weight - q) * T + mu * x))
        var_part4_i_ne_j = np.exp((sigma ** 2 + (sigma ** 2).reshape(len(sigma),-1) + np.outer(sigma, sigma) * rho) * 2 * (1 - rho ** 2) * x/(4 - rho ** 2))
        np.fill_diagonal(var_part4_i_ne_j, np.outer(np.exp(x * sigma ** 2), np.exp(x * sigma ** 2 * 2))) # 生成E[XY] # TODO
        var_part4 = var_part4_i_ne_j - np.outer(np.exp(sigma ** 2 / 2 * x), np.exp(sigma ** 2 / 2 * x))
        var_Sy = sum(sum(var_part1 * var_part2 * var_part3 * var_part4))
        
        gamma_density = gamma(a=T/v, scale=v).pdf(x) # ((1/v) ** (T/v)) / (math.gamma(T/v)) * (x ** (T/v - 1)) * np.exp(-x / v)
        
        if var_upper - var_lower == 0:
            z_y = 0.5
        else:
            z_y = (var_upper - var_Sy)/(var_upper - var_lower)
        C_approximate = z_y * S_lower + (1 - z_y) * S_upper
        return C_approximate * gamma_density# , z_y
    return sci.quad(f, 1e-10, 100)[0] * np.exp(-r * T)
    # return sci.quad(f, 1e-10, np.inf)[0] * np.exp(-r * T)


# The multivariate Variance Gamma model basket option pricing and calibration 论文复刻
## table1
r = 0.03
q = 0

from tqdm import tqdm
# TODO: weight有问题，应该是价格加权
## Table 8
table8_data = np.array(["Alcoa Incorporated" , 36.26 , 0.5374 , -0.5072 , 0.61 , "American Express Company" , 45.53 , 0.3715 , -1.1845 , 1.99 , "American International group" , 48.23 , 0.4076 , -1.8592 , 4.69 , "Bank of America" , 38.56 , 0.4256 , -1.3081 , 2.63 , "Boeing Corporation" , 78.66 , 0.364 , -0.6805 , 3.7 , "Caterpillar" , 85.28 , 0.3731 , -0.7144 , 2.81 , "JP Morgan" , 45.76 , 0.349 , -0.6409 , 4.65 , "Chevron" , 93.18 , 0.2168 , -0.4838 , 1.53 , "Citigroup" , 25.11 , 0.4227 , -0.6585 , 6.17 , "Coca Cola Company" , 60.11 , 0.271 , -0.5272 , 6.55 , "Walt Disney Company" , 31.33 , 0.2962 , -0.5588 , 2.26 , "DuPont" , 52.02 , 0.3222 , -0.5008 , 1.14 , "Exxon Mobile" , 94 , 0.2646 , -0.597 , 8.08 , "General Electric" , 32.69 , 0.2327 , -0.2801 , 4.18 , "General Motors" , 20.13 , 0.6881 , -1.3389 , 4.01 , "Hewlet-Packard" , 48.18 , 0.3927 , -0.6216 , 1.01 , "Home Depot" , 28.68 , 0.4451 , -1.0861 , 0.5 , "Intel" , 22.55 , 0.3652 , -0.7617 , 1.82 , "IBM" , 124.4 , 0.2461 , -0.609 , 6.43 , "Johnson & Johnson" , 66.51 , 0.1775 , -0.2969 , 2.86 , "McDonald's" , 58.3 , 0.2122 , -0.4376 , 1.79 , "Merck & Company" , 39.76 , 0.416 , -0.9171 , 8.62 , "Microsoft" , 30 , 0.3407 , -0.6717 , 1.59 , "3M" , 82.9 , 0.2608 , -0.4586 , 1.12 , "Pfizer" , 20.47 , 0.2156 , 0.3303 , 2.84 , "Practer & Gamble" , 67.17 , 0.1916 , -0.4434 , 1.55 , "AT&T" , 37.51 , 0.3172 , -0.7123 , 0.72 , "United Technologies" , 72.51 , 0.3082 , -0.6888 , 3.06 , "Verizon" , 36.03 , 0.3141 , -0.6515 , 1.04 , "Wal-Mart Stores" , 56.31 , 0.2112 , -0.3738 , 1.17 ])
table8 = pd.DataFrame(data = table8_data.reshape(30,-1)).set_index(0)
table8.columns = ['X_0', 'sigma_i', 'mu_i', 'RMSE']
table8.index.name = ""
table8
## Table 9 
table9_data = np.array(["Day" , " 18 April 2008 " , " 22 May 2008 " , " 18 July 2008" , "Time to maturity" , " 64 days" , " 30 days" , " 29 days" , "v" , 0.076312 , 0.0438 , 0.03395 , "rho" , 0.064745 , 0.23293 , 0.21057 , "RMSE" , 0.0796 , 0.1311 , 0.0367 , "Relative error" , 0.0154 , 0.0574 , 0.0401])
table9 = pd.DataFrame(table9_data.reshape(6,-1))
table9.columns = table9.iloc[0]
table9.columns.name = table9.iloc[0,0]
table9 = table9.drop(table9.index[0])
table9.index = table9.iloc[:,0]
table9 = table9.drop(table9.columns[0], axis=1)
table9.index.name = ''
table9
## Table 10
table10_data = np.array(["Strikes" , "Market prices" , "Model price" , 97 , 31.50 , 31.40  , 99 , 29.50 , 29.43  , 100 , 28.55 , 28.44  , 101 , 27.55 , 27.46  , 102 , 26.58 , 26.48  , 103 , 25.60 , 25.50  , 104 , 24.63 , 24.52  , 105 , 23.65 , 23.55  , 106 , 22.70 , 22.58  , 107 , 21.73 , 21.62  , 108 , 20.78 , 20.66  , 109 , 19.83 , 19.70  , 110 , 18.85 , 18.76  , 111 , 17.95 , 17.82  , 112 , 17.00 , 16.88  , 113 , 16.05 , 15.96  , 114 , 15.15 , 15.04  , 115 , 14.25 , 14.14  , 116 , 13.33 , 13.25  , 117 , 12.50 , 12.38  , 118 , 11.58 , 11.52  , 119 , 10.73 , 10.67  , 120 , 9.93 , 9.85  , 121 , 9.13 , 9.04  , 122 , 8.33 , 8.26  , 123 , 7.55 , 7.50  , 124 , 6.80 , 6.77  , 125 , 6.10 , 6.07  , 126 , 5.40 , 5.40  , 127 , 4.75 , 4.76  , 128 , 4.13 , 4.16  , 129 , 3.55 , 3.59  , 130 , 3.02 , 3.07  , 131 , 2.53 , 2.58  , 132 , 2.08 , 2.14  , 133 , 1.68 , 1.74  , 134 , 1.33 , 1.38  , 135 , 1.04 , 1.08  , 136 , 0.78 , 0.81  , 137 , 0.59 , 0.59  , 138 , 0.42 , 0.41  , 139 , 0.30 , 0.27  , 140 , 0.20 , 0.17])
table10 = pd.DataFrame(table10_data.reshape(44,-1))
table10.columns = table10.iloc[0]
table10 = table10.drop(table10.index[0])
table10.index = table10.iloc[:,0]
table10 = table10.drop(table10.columns[0], axis=1)
table10.T

table8 = table8.astype(float)
table9.iloc[1:,:] = table9.iloc[1:,:].astype(float)
table10 = table10.astype(float)

## Table 11
T = 64/365
v = 0.076312
rho_i = 0.064745
rho = np.ones(30 * 30).reshape(30,30) * rho_i
np.fill_diagonal(rho, 1)
X_0 = np.array(table8.loc[:,'X_0'])
sigma = np.array(table8.loc[:,'sigma_i'])
mu = np.array(table8.loc[:,'mu_i'])

weight = np.ones(len(mu))/len(mu)
weight
from tqdm import tqdm
def monte_carlo_calculate_func(T, v, K, sigma, mu, r, q, power_num = 6):
    sim_times = 10 ** power_num
    np.random.seed(20240321) 
    Normal_values = multivariate_normal.rvs(mean=np.zeros(len(sigma)), cov=rho, size=sim_times)
    Gamma_values = np.random.gamma(shape = T/v, scale = v, size=sim_times)
    omega = 1/v * np.log(1 - 1/2 * (sigma ** 2) * v - mu * v)
    X_price_vector = X_0 * np.exp((r - q + omega) * T) * np.exp(np.outer(mu, Gamma_values)).T * np.exp(sigma * np.sqrt(Gamma_values.reshape(-1, 1)) * Normal_values)
    S_price_vector = np.dot(X_price_vector, weight)
    C_sim_vector_list = (S_price_vector - K) * (S_price_vector - K > 0)
    C_sim = sum(C_sim_vector_list) * np.exp(-r * T) / len(C_sim_vector_list) # 原论文是27.3230
    return C_sim

C_approximate_calculate_func(T = 1, v = 0.5, K = 125, sigma = sigma, mu = mu, r = r, q = 0)
monte_carlo_calculate_func(T = 1, v = 0.5, K = 125, sigma = sigma, mu = mu, r = r, q = 0)
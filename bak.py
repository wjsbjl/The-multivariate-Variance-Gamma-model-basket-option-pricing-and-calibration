# #### sim
# ##### 生成随机数
# ##### 蒙特卡洛（向量化，指令并行）
# def monte_carlo_calculate_func(T, v, K, sigma, mu, r, q, power_num = 6):
#     sim_times = 10 ** power_num
#     np.random.seed(20240321) 
#     Normal_values = multivariate_normal.rvs(mean=[0,0,0], cov=rho, size=sim_times)
#     Gamma_values = np.random.gamma(shape = T/v, scale = v, size=sim_times)
#     omega = 1/v * np.log(1 - 1/2 * (sigma ** 2) * v - mu * v)
#     X_price_vector = X_0 * np.exp((r - q + omega) * T) * np.exp(np.outer(mu, Gamma_values)).T * np.exp(sigma * np.sqrt(Gamma_values.reshape(-1, 1)) * Normal_values)
#     S_price_vector = np.dot(X_price_vector, weight)
#     C_sim_vector_list = (S_price_vector - K) * (S_price_vector - K > 0)
#     C_sim = sum(C_sim_vector_list) * np.exp(-r * T) / len(C_sim_vector_list) # 原论文是27.3230
#     return C_sim
# ### 带入参数
# # 表头
# from itertools import product
# # T, v, K, C_hat, C_sim, vare

# T = [1, 2]
# v = [0.5, 0.75, 0.9]
# K = [75, 90, 100, 110, 125]

# table2 = pd.DataFrame(list(product(T, v, K)), columns = ['T', 'v', 'K']).set_index(['T','v', 'K'])
# table2.T
# r = 0.03
# rho = np.eye(3)
# X_0 = np.array([100, 200, 300])
# q = np.array([0, 0, 0])
# mu = np.array([-0.1368, -0.056, -0.1984])
# sigma = np.array([0.1099, 0.1677, 0.0365])
# weight = np.array([1/3, 1/6, 1/9]) # w是权重，omega才是参数
# from tqdm import tqdm
# C_sim_result_list = list()
# C_approximate_list = list()
# power_num = 6

# for (T,v,K) in tqdm(table2.index.values):
#     C_approximate = C_approximate_calculate_func(T, v, K, sigma, mu, r, q)
#     C_approximate_list.append(C_approximate)
    
#     C_sim = monte_carlo_calculate_func(T, v, K, sigma, mu, r, q, power_num)
#     C_sim_result_list.append(C_sim)
# table2.loc[:,'$\overline{C}[K]$'] = C_approximate_list
# table2.loc[:,'$C^sim[K]$'] = C_sim_result_list
# table2.loc[:,r'$\varepsilon[K]$'] = abs(table2.loc[:,'$\overline{C}[K]$'] - table2.loc[:,'$C^sim[K]$'])/ C_sim_result_list
# table2 = table2.round(4)
# table2
# ## figure 1
# ### 试试逼近
# #### 函数定义
# import sys
# sys.path.append('./Users/zyz/Library/CloudStorage/OneDrive-uibe.edu.cn/Code/Python/mypackage')
# from mypackage.my_plot import my_plot
# def second_derivative_approx_calculation_func(T, v, K, sigma, mu, omega, r, q, power_num, h, cal_func):
#     K_plus_h = K + h
#     K_minus_h = K - h
#     C_sim_at_K_plus_h = cal_func(T, v, K_plus_h, sigma, mu, omega, r, q)
#     C_sim_at_K_minus_h = cal_func(T, v, K_minus_h, sigma, mu, omega, r, q)
#     C_sim_at_K = cal_func(T, v, K, sigma, mu, omega, r, q)

#     second_derivative_approx = (C_sim_at_K_plus_h - 2*C_sim_at_K + C_sim_at_K_minus_h) / (h ** 2)
#     return second_derivative_approx
# #### figure的函数
# def figure_calculate_func(T, v, K, sigma, mu, omega, r, q, power_num, approx_num = 200, h_times = 2, plot_name = ['sigma1 = 0.1', '1-1']):
#     second_derivative_approx_list1 = list()
#     second_derivative_approx_list2 = list()
#     power_num = 6
#     sigma[0] = 0.1
#     h = (np.linspace(70,130,approx_num)[1] - np.linspace(70,130,approx_num)[0]) * h_times

#     for K in tqdm(np.linspace(70,130,approx_num)):
#         second_derivative_approx1 = second_derivative_approx_calculation_func(T, v, K, sigma, mu, omega, r, q, power_num, h, C_approximate_calculate_func)
#         second_derivative_approx2 = second_derivative_approx_calculation_func(T, v, K, sigma, mu, omega, r, q, power_num, h, monte_carlo_calculate_func)
#         second_derivative_approx_list1.append(second_derivative_approx1)
#         second_derivative_approx_list2.append(second_derivative_approx2)
#     plot_df = pd.DataFrame(index = np.linspace(70,130,approx_num))
#     plot_df['Approximate density'] = second_derivative_approx_list1
#     plot_df['Empirical density'] = second_derivative_approx_list2
#     my_plot(plot_df, [f'Probability distribution function {plot_name[0]}', 'Basket terminal value', '', f'figure{plot_name[1]}']).line_plot()
#     return plot_df
# #### 模拟
# T, v, sigma, mu, power_num = 1, 0.7514, np.array([0.1099, 0.1677, 0.0365]), np.array([-0.1368, -0.056, -0.1984]), 6
# approx_num, h_times = 60, 8
# v = 0.3
# plot_df = figure_calculate_func(T, v, K, sigma, mu, omega, r, q, power_num, approx_num, h_times,
#                                 plot_name = ['v = 0.3', '1-1'])
# v = 0.5
# plot_df = figure_calculate_func(T, v, K, sigma, mu, omega, r, q, power_num, approx_num, h_times,
#                                 plot_name = ['v = 0.5', '1-2'])
# v = 0.7
# plot_df = figure_calculate_func(T, v, K, sigma, mu, omega, r, q, power_num, approx_num, h_times,
#                                 plot_name = ['v = 0.7', '1-3'])
# v = 0.9
# plot_df = figure_calculate_func(T, v, K, sigma, mu, omega, r, q, power_num, approx_num, h_times,
#                                 plot_name = ['v = 0.9', '1-4'])
# ## Figure 3
# r = 0.03
# rho = np.eye(3)
# X_0 = np.array([100, 200, 300])
# q = np.array([0, 0, 0])
# mu = np.array([-0.1368, -0.056, -0.1984])
# sigma = np.array([0.1099, 0.1677, 0.0365])
# weight = np.array([1/3, 1/6, 1/9]) # w是权重，omega才是参数
# T = 1
# K = 100
# varepsilon_list = list()
# sim_times = 100
# for sigma_1 in tqdm(np.linspace(0.1,1,sim_times)):
#     sigma[0] = sigma_1
#     C_approximate = C_approximate_calculate_func(T, v, K, sigma, mu, omega, r, q)
#     C_approximate_list.append(C_approximate)
    
#     C_sim = monte_carlo_calculate_func(T, v, K, sigma, mu, omega, r, q, power_num)
#     C_sim_result_list.append(C_sim)
    
#     varepsilon = abs(C_approximate - C_sim)/ C_sim
#     varepsilon_list.append(varepsilon)
# plot_df_varepsilon = pd.DataFrame(index = np.linspace(0,1,sim_times))
# plot_df_varepsilon['varepsilon'] = varepsilon_list
# my_plot(plot_df_varepsilon, [f'Relative error for different choices of sigma 1', 'sigma 1', '', f'figure 3']).line_plot()
# plot_df_varepsilon
# ## table 3
# ### 带入参数
# v = 0.7514
# r = 0.03
# rho = np.eye(3)
# X_0 = np.array([100, 200, 300])
# q = np.array([0, 0, 0])
# mu = np.array([-0.1368, -0.056, -0.1984])
# sigma = np.array([0.1099, 0.1677, 0.0365])
# weight = np.array([1/3, 1/6, 1/9]) # w是权重，omega才是参数
# # 表头
# from itertools import product
# # T, v, K, C_hat, C_sim, vare

# T = [1, 2]
# sigma_1 = [0.05, 0.25, 0.75]
# K = [75, 90, 100, 110, 125]

# table3 = pd.DataFrame(list(product(T, sigma_1, K)), columns = ['T', 'sigma_1', 'K']).set_index(['T','sigma_1', 'K'])
# table3.T
# from tqdm import tqdm
# C_sim_result_list = list()
# C_approximate_list = list()
# power_num = 6

# for (T,sigma_1,K) in tqdm(table3.index.values):
#     sigma[0] = sigma_1
#     C_approximate = C_approximate_calculate_func(T, v, K, sigma, mu, omega, r, q)
#     C_approximate_list.append(C_approximate)
    
#     C_sim = monte_carlo_calculate_func(T, v, K, sigma, mu, omega, r, q, power_num)
#     C_sim_result_list.append(C_sim)
# table3.loc[:,'$\overline{C}[K]$'] = C_approximate_list
# table3.loc[:,'$C^sim[K]$'] = C_sim_result_list
# table3.loc[:,r'$\varepsilon[K]$'] = abs(table3.loc[:,'$\overline{C}[K]$'] - table3.loc[:,'$C^sim[K]$'])/ C_sim_result_list
# table3 = table3.round(4)
# table3
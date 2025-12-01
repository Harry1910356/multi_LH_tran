import numpy as np
import pymc as pm
import arviz as az

# note these imports, thay may be useful in your own task

import matplotlib.pyplot as plt  # plotting library
import numpy as np  # work with numeric arrays without labeled axes
import xarray as xr  # work with arrays with labeled axes


import pandas as pd

xlsx_file = 'Final_full_dataset.xlsx'
df = pd.read_excel(xlsx_file)
#df_cleaned1 = df.rename(columns={'PlasmaCurrent_Measured_ND': 'I_p'})
df['pre_ts'] = df['pre_ts'] /100
df['PlasmaCurrent_Measured_ND'] = df['PlasmaCurrent_Measured_ND'] /(-50000)
df['W'] = df['W'] /1000
df['dw'] = (df['dw'] / 20)
df['H_alpha'] = df['H_alpha'] *(-1)
df['Average_triangularity'] = df['Average_triangularity'] *10
df['f_ELM'] = df['f_ELM'] *40
df['H_alpha'] = df['H_alpha'] *10
df['beta_n'] = df['beta_n'] *5
df['q95'] = df['q95'].abs()
df['NBI'] = (df['NBI'] /20)+0.01
df['resistance'] = (df['resistance'].abs())*100
df['ohmic_power'] =( df['ohmic_power'].abs() )/40000
df['B_phi_R_mag'] = df['B_phi_R_mag'].abs()*5
df = df.apply(pd.to_numeric, errors='coerce')
selected_columns = ['pre_ts', "pe_tem","pe_den",'PlasmaCurrent_Measured_ND', 'CorrectedDensity', 'W', 'dw', 'beta_n', 'q95', 'elongation_lcfs', 'f_ELM', 'B_phi_R_mag', 'H_alpha', 'Average_triangularity',"NBI","plasma_seconds_from_boronization","ohmic_power","resistance"]
log_df = df[selected_columns].apply(np.log)
log_df.columns = [f"log_{col}" for col in selected_columns]
name=['log_PlasmaCurrent_Measured_ND', 'log_CorrectedDensity', 'log_W', 'log_dw', 'log_beta_n', 'log_q95', 'log_elongation_lcfs', 'log_f_ELM', 'log_B_phi_R_mag', 'log_H_alpha', 'log_Average_triangularity',"log_NBI","log_plasma_seconds_from_boronization","log_ohmic_power","log_resistance"]
X = log_df[name]  # 自变量列
X['constant'] = 1
name2=['PlasmaCurrent_Measured_ND', 'CorrectedDensity', 'W', 'dw', 'beta_n', 'q95', 'elongation_lcfs', 'f_ELM', 'B_phi_R_mag', 'H_alpha', 'Average_triangularity',"NBI","plasma_seconds_from_boronization","ohmic_power","resistance"]
X2 = df[name2]  # 自变量列'log_W', 'log_PlasmaCurrent_Measured_ND',
y = log_df['log_pre_ts']
# 因变量列
y2 = df['pre_ts']


# Step 1: Generate simulated data
np.random.seed(0)
N = 5000  # Number of samples
D = 16  # Number of predictors
'''true_beta = np.array([2.5, 0.0, -1.5, 1.0, 5.0])  # True regression coefficients, sparse
sigma_true = 1.0  # True noise standard deviation

# Generate predictors
X = np.random.randn(N, D)

# Generate response variable y
noise = np.random.normal(0, sigma_true, N)
y = X @ true_beta + noise'''

# Step 2: Define the PyMC model
with pm.Model() as model:
    # Priors
    # Beta for regression coefficients, grouped into classes
    alpha = pm.HalfNormal("alpha", sigma=1.0)  # Dirichlet prior concentration parameter
    p = pm.Dirichlet("p", a=pm.math.ones(16) * alpha, shape=16)  # Class probabilities

    S = pm.Categorical("S", p=p, shape=D)  # Class assignment for predictors (0-9)

    eta = pm.InverseGamma("eta", alpha=1.0, beta=1.0)  # Variance for beta_m
    beta_m = pm.Normal("beta_m", mu=0.5, sigma=eta, shape=16)  # Regression coefficients for each class

    # Ensure S only indexes valid categories of beta_m
    beta = pm.Deterministic("beta", beta_m[S])  # Map beta based on S

    # Sparsity indicator gamma
    pi = pm.Beta("pi", alpha=1.0, beta=1.0)  # Probability of gamma=1
    gamma = pm.Bernoulli("gamma", p=pi, shape=D)

    # Noise variance (sigma^2)
    sigma2 = pm.InverseGamma("sigma2", alpha=0.1, beta=0.1)  # Prior for noise variance

    # Observed data likelihood
    mu = pm.math.dot(X, beta * gamma)  # Predicted mean
    y_obs = pm.Normal("y_obs", mu=mu, sigma=pm.math.sqrt(sigma2), observed=y)

    # Inference
    trace = pm.sample(N, tune=500, return_inferencedata=True)

# Step 3: Analyze and visualize the results
az.plot_trace(trace, var_names=["beta", "gamma", "sigma2", "eta", "pi", "p"])
plt.show()
summary=az.summary(trace, var_names=["beta", "gamma", "sigma2", "eta", "pi", "p"])
print(summary)
beta_post_mean = trace.posterior['beta'].mean(dim=["chain", "draw"]).values
gamma_post_mean = trace.posterior['gamma'].mean(dim=["chain", "draw"]).values
gamma_binary = (gamma_post_mean > 0.5).astype(int)
print(gamma_binary)
sigma2_post_mean = trace.posterior['sigma2'].mean().values

y_pred_mean = X @ (beta_post_mean * gamma_post_mean)  # 预测均值
print(y)
noise_samples = np.random.normal(0, np.sqrt(sigma2_post_mean.mean()), size=y.shape[0])
y_pred = y_pred_mean + noise_samples
print(y_pred)
# 计算 R²
SS_res = np.sum((y_pred - y) ** 2)
SS_tot = np.sum((y - np.mean(y)) ** 2)
R_squared = 1 - (SS_res / SS_tot)
print(f"模型的 R² 值为: {R_squared:.4f}")

# 绘制实际值与预测值的散点图
plt.figure(figsize=(8,6))
plt.scatter(y, y_pred, alpha=0.5)
plt.xlabel("实际值 y")
plt.ylabel("预测值 $\hat{y}$")
plt.title("实际值与预测值的关系图")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # 绘制 y=x 的参考线
plt.show()
# 从后验中随机抽取 10 个样本
num_samples = 1000
posterior_indices = np.random.choice(
    trace.posterior["beta"].stack(samples=["chain", "draw"]).shape[1],
    size=num_samples,
    replace=False
)

# 定义新数据
X_new =X  # 新数据 (10 个样本)

# 存储预测结果
predictions = []
S_new = []
gamma_new=[]
for idx in posterior_indices:
    beta_sample = trace.posterior["beta"].stack(samples=["chain", "draw"]).values[:, idx]
    gamma_sample = trace.posterior["gamma"].stack(samples=["chain", "draw"]).values[:, idx]
    S_sample = trace.posterior["S"].stack(samples=["chain", "draw"]).values[:, idx]
    sigma_sample = np.sqrt(trace.posterior["sigma2"].stack(samples=["chain", "draw"]).values[idx])
    #gamma_binary = (gamma_post_mean > 0.5).astype(int)
    # 计算预测值
    mu_pred = X @ (beta_sample * gamma_sample)  # 预测均值
    y_pred = np.random.normal(mu_pred, sigma_sample, size=X.shape[0])  # 从预测分布采样
    S_new.append(gamma_sample*beta_sample)
    gamma_new.append(gamma_sample)
    predictions.append(y_pred)
S_mean_new = np.array(S_new).mean(axis=0)
gamma_mean = np.array(gamma_new).mean(axis=0)
gamma_binary = (gamma_mean > 0.5).astype(int)
print(gamma_binary)
# 转置为 (新样本数 x 后验样本数)
predictions = np.array(predictions).T  # (num_samples x 新样本数)
pred_mean = predictions.mean(axis=1)
print(S_mean_new)
SS_res = np.sum((pred_mean - y) ** 2)
SS_tot = np.sum((y - np.mean(y)) ** 2)
R_squared = 1 - (SS_res / SS_tot)
print(f"模型的 R² 值为: {R_squared:.4f}")

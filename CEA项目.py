import sympy as sp
import numpy
# 定义未知量 R, B 以及其它参数
R, B = sp.symbols('R B', positive=True)
# 预先定义那些数值参数
alpha_I = 0.93
alpha_epsilon = 0.58
alpha_R = 1.97
alpha_B = 0.15
val_alpha_n = 0.41  # 用数值替换符号 alpha_n

# 预先计算 gamma_I, gamma_R, gamma_B, gamma_epsilon 的数值
gamma_I_val = 1 + val_alpha_n + alpha_I  # 1 + 0.41 + 0.93 = 2.34
gamma_R_val = alpha_R + alpha_I - val_alpha_n  # 1.97 + 0.93 - 0.41 = 2.49
gamma_B_val = alpha_B + val_alpha_n + alpha_I + 2  # 0.15 + 0.41 + 0.93 + 2 = 4.49
gamma_epsilon_val = 1 + alpha_epsilon + 2 * alpha_I  # 1 + 0.58 + 0.82 = 2.4

# 构造参数字典，注意这里将依赖符号计算的参数都转为数值
Q = 10
gamma_rad = 0.7
lam = 4.94
alpha_P = 0.69
C_SL = 0.056
C_n = 3.183
alpha_n = val_alpha_n
C_I = 13.144
gamma_I = gamma_I_val
gamma_R = gamma_R_val
gamma_B = gamma_B_val
C_fus = 1.45e-3
C_beta = 0.741
C_tr = 0.087
P_fus_hat = 500
M = 2.554
alpha_M = 0.19
kappa = 1.7
alpha_kappa = 0.78
epsilon = 0.333
gamma_epsilon = gamma_epsilon_val
q = 3
beta_N = 1.6
n_N = 10

# 定义第一个方程左右两边
A_expr = (Q / (gamma_rad * (1 + Q / lam))) ** (1 + (-alpha_P))
K1_expr = (C_SL * C_n ** alpha_n * C_I ** gamma_I * C_fus) / (C_beta * C_tr) * P_fus_hat ** (
    -alpha_P) * M ** alpha_M * kappa ** alpha_kappa * epsilon ** gamma_epsilon * q ** (
              -gamma_I) * n_N ** alpha_n * beta_N * R ** (gamma_R) * B ** (gamma_B)

eq1 = sp.Eq(A_expr, K1_expr)

# 定义第二个方程
K2_expr = (C_fus * C_I ** 2) / (C_beta ** 2) * (kappa * epsilon ** 4 / q ** 2) * beta_N ** 2 * R ** 3 * B ** 4
eq2 = sp.Eq(P_fus_hat, K2_expr)

# -------------------------------


# 初始猜测，根据实际情况调整
initial_guess = (6, 5)

# 使用 nsolve 进行数值求解
solution = sp.nsolve([eq1, eq2], (R, B), initial_guess)

print("数值求解得到的 R 和 B 为：")
sp.pprint(solution)

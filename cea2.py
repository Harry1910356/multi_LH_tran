import sympy as sp

# 定义未知量 R, B 以及其它符号变量
R, B = sp.symbols('R B', positive=True)
Q, gamma_rad, lam, alpha_P = sp.symbols('Q gamma_rad lam alpha_P', positive=True)
C_SL, C_n, alpha_n, C_I, gamma_I, gamma_R, gamma_B, C_fus, C_beta, C_tr = sp.symbols(
    'C_SL C_n alpha_n C_I gamma_I gamma_R gamma_B C_fus C_beta C_tr', positive=True)
P_fus_hat, M, alpha_M, kappa, alpha_kappa, epsilon, gamma_epsilon, q, beta_N, n_N = sp.symbols(
    'P_fus_hat M alpha_M kappa alpha_kappa epsilon gamma_epsilon q beta_N n_N', positive=True)

# 构造第一个方程
# eq1:  (Q/(gamma_rad*(1+Q/lam)))^(1+alpha_P) = K1_expr
# 其中 K1_expr = (C_SL * C_n^alpha_n * C_I^gamma_I * C_fus)/(C_beta * C_tr) *
#                P_fus_hat^alpha_P * M^alpha_M * kappa^alpha_kappa * epsilon^gamma_epsilon *
#                q^(-gamma_I) * n_N^alpha_n * beta_N * R^(gamma_R) * B^(gamma_B)
A_expr = (Q / (gamma_rad * (1 + Q / lam))) ** (1 + alpha_P)
K1_expr = (C_SL * C_n**alpha_n * C_I**gamma_I * C_fus) / (C_beta * C_tr) \
          * P_fus_hat**alpha_P * M**alpha_M * kappa**alpha_kappa * epsilon**gamma_epsilon \
          * q**(-gamma_I) * n_N**alpha_n * beta_N * R**gamma_R * B**gamma_B
eq1 = sp.Eq(A_expr, K1_expr)

# 构造第二个方程
# eq2:  P_fus_hat = K2_expr, 其中
# K2_expr = (C_fus * C_I^2)/(C_beta^2) * (kappa * epsilon^4/q^2) * beta_N^2 * R^3 * B^4
K2_expr = (C_fus * C_I**2) / (C_beta**2) * (kappa * epsilon**4 / q**2) * beta_N**2 * R**3 * B**4
eq2 = sp.Eq(P_fus_hat, K2_expr)

# -------------------------------
# 预先定义数值参数
alpha_I = 0.93
alpha_epsilon = 0.58
alpha_R = 1.97
alpha_B = 0.15
val_alpha_n = 0.41  # 数值化的 alpha_n

# 根据题意预先计算 gamma_I, gamma_R, gamma_B, gamma_epsilon 的数值
gamma_I_val = 1 + val_alpha_n + alpha_I         # 1 + 0.41 + 0.93 = 2.34
gamma_R_val = alpha_R + alpha_I - val_alpha_n      # 1.97 + 0.93 - 0.41 = 2.49
gamma_B_val = alpha_B + val_alpha_n + alpha_I + 2    # 0.15 + 0.41 + 0.93 + 2 = 4.49
# 注意：这里根据之前的定义，gamma_epsilon = 1 + alpha_epsilon + 2*alpha_n
gamma_epsilon_val = 1 + alpha_epsilon + 2 * val_alpha_n  # 1 + 0.58 + 0.82 = 2.4

# 构造完整参数字典（所有参数都数值化）
params = {
    Q: 10,
    gamma_rad: 0.7,
    lam: 4.94,
    alpha_P: -0.69,
    C_SL: 0.056,
    C_n: 3.183,
    alpha_n: val_alpha_n,
    C_I: 13.144,
    gamma_I: gamma_I_val,
    gamma_R: gamma_R_val,
    gamma_B: gamma_B_val,
    C_fus: 1.45e-3,
    C_beta: 0.741,
    C_tr: 0.087,
    P_fus_hat: 500,
    M: 2.554,
    alpha_M: 0.19,
    kappa: 1.7,
    alpha_kappa: 0.78,
    epsilon: 0.333,
    gamma_epsilon: gamma_epsilon_val,
    q: 3,
    beta_N: 1.6,
    n_N: 2,
}

# -------------------------------
# 采用代入法
# 根据 eq2 求 R 关于 B 的表达式：
# 从 eq2 得到: R^3 = P_fus_hat / ( K2_no_R * B^4 )
K2_no_R = (C_fus * C_I**2) / (C_beta**2) * (kappa * epsilon**4 / q**2) * beta_N**2
R_expr = (P_fus_hat / (K2_no_R * B**4))**(sp.Rational(1,3))

# 将 R_expr 代入 eq1，得到关于 B 的方程
eq1_sub = eq1.subs(R, R_expr)

# 先替换参数，变成全数值表达式
eq1_sub_num = eq1_sub.subs(params)

# 对单变量方程 eq1_sub_num 用 nsolve 求 B（初始猜测根据实际情况调整，这里以 5 为初始猜测）
B_sol = sp.nsolve(eq1_sub_num, B, 5)
# 然后利用 R_expr 求出 R
R_sol = R_expr.subs(params).subs(B, B_sol)

print("数值求解得到的 B =", B_sol)
print("数值求解得到的 R =", R_sol)

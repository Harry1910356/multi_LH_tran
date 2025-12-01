import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 创建相关系数矩阵的 DataFrame
data = {
    "BT": [1.000000, 0.863060, -0.118966, 0.803632, 0.868909, 0.173681],
    "BP": [0.863060, 1.000000, 0.262335, 0.824304, 0.775428, 0.129850],
    "IP": [-0.118966, 0.262335, 1.000000, -0.096937, -0.329181, 0.070885],
    "PLTH/S/2": [0.803632, 0.824304, -0.096937, 1.000000, 0.865607, -0.241761],
    "NEL": [0.868909, 0.775428, -0.329181, 0.865607, 1.000000, 0.089726],
    "PGASA": [0.173681, 0.129850, 0.070885, -0.241761, 0.089726, 1.000000]
}
index = ["BT", "BP", "IP", "PLTH/S/2", "NEL", "PGASA"]
correlation_matrix = pd.DataFrame(data, index=index)

# 打印 DataFrame 确认数据
print(correlation_matrix)
plt.figure(figsize=(10, 8))  # 设置图形大小
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix Heatmap')  # 添加标题
plt.show()
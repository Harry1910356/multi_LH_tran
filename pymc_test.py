import pandas as pd

# 读取CSV文件为DataFrame
df = pd.read_csv('LH_metal_DB_v1.12.csv')  # 替换为你的文件路径

# 筛选 STAND 为 1 的行并统计 DIVCON 中的唯一值及其数量
stand_1 = df[df['SELEC2024'] == 1]
divcon_1_counts = stand_1['DIVCON'].value_counts()

# 筛选 STAND 为 0 的行并统计 DIVCON 中的唯一值及其数量
stand_0 = df[df['SELEC2024'] == 0]
divcon_0_counts = stand_0['DIVCON'].value_counts()

# 打印结果
print("SELEC2024 == 1 时的 DIVCON 分布：")
print(divcon_1_counts)
print("\nSELEC2024 == 0 时的 DIVCON 分布：")
print(divcon_0_counts)

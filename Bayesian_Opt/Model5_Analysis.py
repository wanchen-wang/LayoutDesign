import pandas as pd
import numpy as np
from scipy import stats
import os

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 读取CSV文件
file_path = r"D:\PYTHON\layout design\Analysis_A_UGstandard_Data\analysis_results_swA_lagrangian_30cut.csv"
df = pd.read_csv(file_path)

print("=== Model5 贝叶斯优化对比分析 ===")
print(f"总数据行数: {len(df)}")

# 取前100条数据
df_100 = df.head(100)
print(f"取前100条数据: {len(df_100)}")

# 从前100条中随机抽取20条
sample_indices = np.random.choice(df_100.index, size=20, replace=False)
df_sample = df_100.loc[sample_indices]

print(f"随机抽取20条数据 (种子42): {len(df_sample)}")

# 计算相对误差的统计量
relative_errors = df_sample['error_pct']

# 基本统计量
mean_error = relative_errors.mean()
std_error = relative_errors.std()
n = len(relative_errors)

print("\n=== 相对误差统计分析 ===")
print(f"平均相对误差: {mean_error:.4f}%")
print(f"标准差: {std_error:.4f}%")
print(f"样本数量: {n}")

# 计算95%置信区间 (t分布)
confidence_level = 0.95
degrees_of_freedom = n - 1
t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
margin_of_error = t_value * (std_error / np.sqrt(n))

confidence_interval_lower = mean_error - margin_of_error
confidence_interval_upper = mean_error + margin_of_error

print("\n=== 95% 置信区间 ===")
print(f"置信区间: [{confidence_interval_lower:.4f}%, {confidence_interval_upper:.4f}%]")
print(f"置信区间宽度: {confidence_interval_upper - confidence_interval_lower:.4f}%")

# 输出抽样数据的详细信息
print("\n=== 抽样数据详情 ===")
print("排名 | 相对误差(%) | 绝对误差 | 误差密度")
print("-" * 50)
for i, (idx, row) in enumerate(df_sample.iterrows(), 1):
    print(f"{i:2d}   | {row['error_pct']:10.4f}   | {row['abs_error']:8.4f}   | {row['error_density']:10.6f}")

# 保存结果到CSV
output_file = r"D:\PYTHON\layout design\Analysis_A_UGstandard_Data\Model5_Sample_Analysis.csv"
result_df = pd.DataFrame({
    'metric': ['平均相对误差(%)', '标准差(%)', '样本数量', '95%置信区间下限(%)', '95%置信区间上限(%)'],
    'value': [mean_error, std_error, n, confidence_interval_lower, confidence_interval_upper]
})

result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n✓ 分析结果已保存到: {output_file}")

# 同时保存抽样数据
sample_output_file = r"D:\PYTHON\layout design\Analysis_A_UGstandard_Data\Model5_Sample_Data.csv"
df_sample.to_csv(sample_output_file, index=False, encoding='utf-8-sig')
print(f"✓ 抽样数据已保存到: {sample_output_file}")
import pandas as pd
import os

# 读取现有的分析结果
input_file = r"D:\PYTHON\layout design\Analysis_A_UGstandard_Data\analysis_results_swA_lagrangian_30cut.csv"
output_dir = r"D:\PYTHON\layout design\Analysis_A_Bayesian_Opt"
output_file = os.path.join(output_dir, "model3_baseline_analysis.csv")

# 读取数据
df = pd.read_csv(input_file)

print(f"读取到 {len(df)} 条数据记录")

# 取前100条数据
df_100 = df.head(100)
print(f"取前100条数据进行分析")

# 重命名列以匹配Model3的格式
df_model3 = df_100.copy()
df_model3 = df_model3.rename(columns={
    'wave_id': 'wave_id',
    'abs_error': 'abs_error',
    'error_pct': 'error_pct',
    'true_h0': 'Delta_Z_true',
    'dh': 'Delta_Z_calc',
    'error_density': 'error_density'
})

# 添加Model3特有的列
df_model3['J'] = df_model3['abs_error']  # 目标函数值等于绝对误差
df_model3['rel_error'] = df_model3['error_pct'] / 100  # 相对误差（小数形式）

# 重新排列列顺序
columns_order = ['wave_id', 'J', 'abs_error', 'rel_error', 'error_pct', 'Delta_Z_calc',
                 'Delta_Z_true', 'error_density']
df_model3 = df_model3[columns_order]

# 保存为CSV
df_model3.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"Model3基准分析结果已保存到: {output_file}")

# 显示统计信息
print("\n统计信息:")
print(f"平均绝对误差: {df_model3['abs_error'].mean():.4f} m")
print(f"平均相对误差: {df_model3['error_pct'].mean():.2f} %")
print(f"最大相对误差: {df_model3['error_pct'].max():.2f} %")
print(f"最小相对误差: {df_model3['error_pct'].min():.2f} %")
print(f"标准差: {df_model3['error_pct'].std():.2f} %")

print("\n前10条数据预览:")
print(df_model3.head(10).to_string(index=False))
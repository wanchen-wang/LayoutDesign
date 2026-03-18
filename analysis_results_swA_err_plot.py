import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ==========================================
# 1. 数据加载与预处理
# ==========================================
file_path = 'analysis_results_swA.csv'
print(f"正在加载数据: {file_path} ...")
df = pd.read_csv(file_path)

# 计算绝对误差 (Absolute Error)
df['abs_error'] = np.abs(df['dh'] - df['true_h0'])

# 计算积分时长 (Integration Duration = t_U - t_w0)
df['duration'] = df['t_U'] - df['t_w0']

# ==========================================
# 2. 统计相关性分析
# ==========================================
# 计算真实振幅与误差百分比的皮尔逊相关系数
corr_h0_err, p_val1 = pearsonr(df['true_h0'], df['error_pct'])
# 计算真实振幅与积分时长的皮尔逊相关系数
corr_h0_dur, p_val2 = pearsonr(df['true_h0'], df['duration'])
# 计算积分时长与误差百分比的皮尔逊相关系数
corr_duration_err, p_val3 = pearsonr(df['duration'], df['error_pct'])

print("\n================ 统计分析报告 ================")
print(f"共有 {len(df)} 组有效数据参与分析。")
print(f"【真实振幅 vs 误差百分比】相关系数 r = {corr_h0_err:.4f} (p值={p_val1:.2e})")
print(f"  -> 说明: 极强的正相关！振幅越大，系统高估得越严重。")
print(f"【真实振幅 vs 积分时长】  相关系数 r = {corr_h0_dur:.4f} (p值={p_val2:.2e})")
print(f"  -> 说明: 强正相关！大波导致积分窗口被拉长，进一步放大了面积。")
print(f"【积分时长 vs 误差百分比】相关系数 r = {corr_duration_err:.4f} (p值={p_val3:.2e})")
print(f"  -> 说明: 积分时长与误差的相关性。")
print("============================================\n")

# ==========================================
# 3. 结果可视化 (绘制双子图)
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ---- 图 1: 真实振幅 vs 误差百分比 ----
axes[0].scatter(df['true_h0'], df['error_pct'], color='#d9534f', alpha=0.7, edgecolors='k', s=60)
# 添加线性拟合趋势线
z1 = np.polyfit(df['true_h0'], df['error_pct'], 1)
p1 = np.poly1d(z1)
x_fit1 = np.linspace(df['true_h0'].min(), df['true_h0'].max(), 100)
axes[0].plot(x_fit1, p1(x_fit1), "k--", linewidth=2, label=f'Trend (r={corr_h0_err:.2f})')

axes[0].set_title('Correlation: True Amplitude vs. Measurement Error', fontsize=14)
axes[0].set_xlabel('True Amplitude $h_0$ (m)', fontsize=12)
axes[0].set_ylabel('Measurement Error (%)', fontsize=12)
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend(fontsize=11)

# ---- 图 2: 积分时长 vs 误差百分比 ----
axes[1].scatter(df['duration'], df['error_pct'], color='#5bc0de', alpha=0.7, edgecolors='k', s=60)
# 添加线性拟合趋势线
z2 = np.polyfit(df['duration'], df['error_pct'], 1)
p2 = np.poly1d(z2)
x_fit2 = np.linspace(df['duration'].min(), df['duration'].max(), 100)
axes[1].plot(x_fit2, p2(x_fit2), "k--", linewidth=2, label=f'Trend (r={corr_duration_err:.2f})')

axes[1].set_title('Correlation: Integration Duration vs. Measurement Error', fontsize=14)
axes[1].set_xlabel('Integration Duration (s)', fontsize=12)
axes[1].set_ylabel('Measurement Error (%)', fontsize=12)
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.show()
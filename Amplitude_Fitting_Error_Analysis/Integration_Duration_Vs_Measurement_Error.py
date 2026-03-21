import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

def analyze_integration_duration(file_path):
    if not os.path.exists(file_path):
        print(f"⚠️ 找不到文件: {file_path}")
        return

    print(f"\n{'='*60}")
    print(f"🚀 开始执行积分时长敏感性与全局截断误差分析: {file_path}")
    print(f"{'='*60}")

    # 1. 数据加载与预处理
    df = pd.read_csv(file_path)
    
    # 核心计算：积分区间长度 L = t_U - t_w0
    df['duration'] = df['t_U'] - df['t_w0']
    
    # 计算绝对误差与误差百分比 (确保数据列存在)
    df['abs_error'] = np.abs(df['dh'] - df['true_h0'])
    if 'error_pct' not in df.columns:
        df['error_pct'] = df['abs_error'] / df['true_h0'] * 100

    # 2. 定量计算 Pearson 相关系数与 p 值
    r_dur, p_dur = pearsonr(df['duration'], df['error_pct'])

    print(f"【分析指标: 积分区间长度 (Duration)】")
    print(f"  -> 与测量误差的 Pearson 相关系数 r = {r_dur:.4f} (p值={p_dur:.2e})")
    print(f"  -> 理论对标: 验证是否因长尾积分导致“全局截断误差”与“随机噪声”累积。")
    print(f"{'='*60}\n")

    # 3. 生成残差诊断与趋势散点图
    plt.figure(figsize=(8, 6))

    # 绘制 积分时长 vs 误差百分比 的散点
    plt.scatter(df['duration'], df['error_pct'], color='#d9534f', alpha=0.7, edgecolors='k', s=60)
    
    # 添加线性拟合趋势线
    z1 = np.polyfit(df['duration'], df['error_pct'], 1)
    p1 = np.poly1d(z1)
    x_fit1 = np.linspace(df['duration'].min(), df['duration'].max(), 100)
    plt.plot(x_fit1, p1(x_fit1), "k--", linewidth=2, label=f'Linear Trend (r={r_dur:.2f})')
    
    plt.title('Sensitivity: Integration Duration vs. Measurement Error', fontsize=14)
    plt.xlabel('Integration Duration $L = t_U - t_{w0}$ (s)', fontsize=12)
    plt.ylabel('Measurement Error (%)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.show()

# 运行代码：请替换为你实际生成的拉格朗日数据文件
if __name__ == "__main__":
    analyze_integration_duration('analysis_results_swA_lagrangian.csv')
    pass
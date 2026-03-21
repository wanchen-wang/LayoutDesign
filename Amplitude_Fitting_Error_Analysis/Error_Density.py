import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_comprehensive_error_density(file_path='Analysis_Results_SwA_Lagrangian_Cut_Data/analysis_results_swA_lagrangian_30cut.csv'):
    print(f"\n{'='*60}")
    print(f"🚀 开始执行: 局部误差密度全维度诊断分析 (起点、终点、时长)")
    print(f"   理论依据: 剥离全局截断误差，评估边界二阶导数惩罚")
    print(f"{'='*60}")

    if not os.path.exists(file_path):
        print(f"⚠️ 找不到文件: {file_path}，请确保路径正确。")
        return

    # 1. 加载数据与指标计算
    df = pd.read_csv(file_path)
    
    if 'duration' not in df.columns:
        df['duration'] = df['t_U'] - df['t_w0']
    
    if 'error_pct' not in df.columns:
        if 'abs_error' not in df.columns:
            df['abs_error'] = np.abs(df['dh'] - df['true_h0'])
        df['error_pct'] = df['abs_error'] / df['true_h0'] * 100
        
    # ⭐ 计算局部误差密度 (%/s)
    df['error_density'] = df['error_pct'] / df['duration']

    # 2. 绘制 1x3 综合诊断仪表盘
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # ---------------- 图1：起点 t_w0 vs 误差密度 ----------------
    axes[0].scatter(df['t_w0'], df['error_density'], color='#3498db', alpha=0.7, edgecolors='k', s=50)
    try:
        z1 = np.polyfit(df['t_w0'], df['error_density'], 2)
        p1 = np.poly1d(z1)
        x_fit1 = np.linspace(df['t_w0'].min(), df['t_w0'].max(), 100)
        axes[0].plot(x_fit1, p1(x_fit1), "r--", linewidth=2, label='2nd-order Fit')
    except:
        pass
    axes[0].set_title('Start Time ($t_{w0}$) vs. Error Density', fontsize=12)
    axes[0].set_xlabel('Integration Start Time $t_{w0}$ (s)', fontsize=11)
    axes[0].set_ylabel('Local Error Density (% / s)', fontsize=11)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()

    # ---------------- 图2：终点 t_U vs 误差密度 ----------------
    axes[1].scatter(df['t_U'], df['error_density'], color='#e67e22', alpha=0.7, edgecolors='k', s=50)
    try:
        z2 = np.polyfit(df['t_U'], df['error_density'], 2)
        p2 = np.poly1d(z2)
        x_fit2 = np.linspace(df['t_U'].min(), df['t_U'].max(), 100)
        axes[1].plot(x_fit2, p2(x_fit2), "r--", linewidth=2, label='2nd-order Fit')
    except:
        pass
    axes[1].set_title('End Time ($t_U$) vs. Error Density', fontsize=12)
    axes[1].set_xlabel('Integration End Time $t_U$ (s)', fontsize=11)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend()

    # ---------------- 图3：积分时长 Duration vs 误差密度 ----------------
    axes[2].scatter(df['duration'], df['error_density'], color='#2ecc71', alpha=0.7, edgecolors='k', s=50)
    try:
        # 时长一般看线性趋势，检查是否彻底剥离了长度相关性
        z3 = np.polyfit(df['duration'], df['error_density'], 1)
        p3 = np.poly1d(z3)
        x_fit3 = np.linspace(df['duration'].min(), df['duration'].max(), 100)
        axes[2].plot(x_fit3, p3(x_fit3), "k-.", linewidth=2, label='Linear Trend')
    except:
        pass
    axes[2].set_title('Integration Duration vs. Error Density', fontsize=12)
    axes[2].set_xlabel('Duration $L = t_U - t_{w0}$ (s)', fontsize=11)
    axes[2].grid(True, linestyle=':', alpha=0.6)
    axes[2].legend()

    plt.suptitle('Comprehensive Diagnostic Dashboard for Local Error Density', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    # 3. 打印核心统计报告
    print(f"【多维度敏感性报告】")
    print(f"  -> 平均局部误差密度: {df['error_density'].mean():.5f} %/s")
    print(f"  -> 时长与密度的 Pearson 相关系数: {df['duration'].corr(df['error_density']):.4f}")
    print(f"============================================================")

if __name__ == "__main__":
    analyze_comprehensive_error_density()
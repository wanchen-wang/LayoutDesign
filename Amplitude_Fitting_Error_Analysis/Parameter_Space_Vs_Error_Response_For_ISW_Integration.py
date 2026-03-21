import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_scatter_with_error_annotations(file_path='D:\\PYTHON\\layout design\\Analysis_Results_SwA_Lagrangian_Cut_Data\\analysis_results_swA_lagrangian_0cut.csv'):
    print(f"\n{'='*60}")
    print(f"🚀 开始生成: 积分参数空间与相对误差响应散点图")
    print(f"{'='*60}")

    if not os.path.exists(file_path):
        print(f"⚠️ 找不到文件: {file_path}，请确保路径正确。")
        return

    # 1. 读取数据与指标准备
    df = pd.read_csv(file_path)
    
    if 'duration' not in df.columns:
        df['duration'] = df['t_U'] - df['t_w0']
        
    if 'error_pct' not in df.columns:
        if 'abs_error' not in df.columns:
            df['abs_error'] = np.abs(df['dh'] - df['true_h0'])
        df['error_pct'] = df['abs_error'] / df['true_h0'] * 100

    # 2. 创建 1x2 的画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 颜色映射：根据误差百分比渐变 (蓝 -> 黄 -> 红)
    cmap = 'jet'
    
    # ==========================================
    # 图 1：X=起始时间 (t_w0), Y=终止时间 (t_U)
    # ==========================================
    scatter1 = ax1.scatter(df['t_w0'], df['t_U'], c=df['error_pct'], cmap=cmap, 
                           s=100, edgecolors='black', alpha=0.8)
    ax1.set_title('Scatter Plot 1: Start Time vs. End Time', fontsize=14, pad=15)
    ax1.set_xlabel('Integration Start Time $t_{w0}$ (s)', fontsize=12)
    ax1.set_ylabel('Integration End Time $t_U$ (s)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 在每个点上标注相对误差
    for i in range(len(df)):
        ax1.annotate(f"{df['error_pct'].iloc[i]:.1f}%", 
                     (df['t_w0'].iloc[i], df['t_U'].iloc[i]),
                     textcoords="offset points", xytext=(8, 4), ha='left', fontsize=9, color='darkred', weight='bold')

    # ==========================================
    # 图 2：X=起始时间 (t_w0), Y=持续时间 (Duration)
    # ==========================================
    scatter2 = ax2.scatter(df['t_w0'], df['duration'], c=df['error_pct'], cmap=cmap, 
                           s=100, edgecolors='black', alpha=0.8)
    ax2.set_title('Scatter Plot 2: Start Time vs. Duration', fontsize=14, pad=15)
    ax2.set_xlabel('Integration Start Time $t_{w0}$ (s)', fontsize=12)
    ax2.set_ylabel('Integration Duration $L$ (s)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # 在每个点上标注相对误差
    for i in range(len(df)):
        ax2.annotate(f"{df['error_pct'].iloc[i]:.1f}%", 
                     (df['t_w0'].iloc[i], df['duration'].iloc[i]),
                     textcoords="offset points", xytext=(8, 4), ha='left', fontsize=9, color='darkred', weight='bold')

    # ==========================================
    # 共享颜色条与整体布局
    # ==========================================
    cbar = fig.colorbar(scatter2, ax=[ax1, ax2], fraction=0.03, pad=0.03)
    cbar.set_label('Relative Error (%)', fontsize=12)
    
    plt.suptitle('Parameter Space vs. Error Response for ISW Integration', fontsize=18, y=1.02, weight='bold')
    plt.show()

# 执行绘图
if __name__ == "__main__":
    plot_scatter_with_error_annotations()
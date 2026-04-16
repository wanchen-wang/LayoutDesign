import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# ==========================================
# 1. 字体与路径配置
# ==========================================
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Microsoft YaHei", rc={"axes.unicode_minus": False})

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = Path(r"D:\PYTHON\layout design\Analysis_C_Data\TDOA_Metrics_Summary.csv")

def plot_comprehensive_dashboard():
    if not CSV_PATH.exists():
        print(f"❌ 找不到数据文件: {CSV_PATH}")
        return

    # 读取完全匹配您列名的数据
    df = pd.read_csv(CSV_PATH)
    print(f"✅ 成功加载 {len(df)} 组 OSSE 蒙特卡洛实验数据。")

    # 创建 1x2 的画板
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"水下滑翔机内孤立波组网观测综合误差分析 (N={len(df)}组)", fontsize=18, fontweight='bold', y=1.02)

    # ==================== 子图 1: 相速度误差分布 (C_p_error) ====================
    ax1 = axes[0]
    sns.histplot(data=df, x='C_p_error', kde=True, ax=ax1, color='#2b59c3', bins=20, alpha=0.6)
    ax1.set_title("1. 组网反演：相速度误差分布 (C_p_error)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("相速度误差绝对值 (m/s)", fontsize=12)
    ax1.set_ylabel("频数", fontsize=12)
    ax1.axvline(0, color='k', linestyle='--', linewidth=1.5, alpha=0.8)
    
    cp_mean, cp_std = df['C_p_error'].mean(), df['C_p_error'].std()
    ax1.text(0.05, 0.95, f"Mean: {cp_mean:.4f} m/s\nStd: {cp_std:.4f}", 
             transform=ax1.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # ==================== 子图 2: 传播偏角误差分布 (theta_error) ====================
    ax2 = axes[1]
    sns.histplot(data=df, x='theta_error', kde=True, ax=ax2, color='#d1495b', bins=20, alpha=0.6)
    ax2.set_title("2. 组网反演：传播偏角误差分布 (theta_error)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("偏角误差绝对值 (Degree)", fontsize=12)
    ax2.set_ylabel("频数", fontsize=12)
    ax2.axvline(0, color='k', linestyle='--', linewidth=1.5, alpha=0.8)
    
    th_mean, th_std = df['theta_error'].mean(), df['theta_error'].std()
    ax2.text(0.05, 0.95, f"Mean: {th_mean:.2f}°\nStd: {th_std:.2f}°", 
             transform=ax2.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_comprehensive_dashboard()
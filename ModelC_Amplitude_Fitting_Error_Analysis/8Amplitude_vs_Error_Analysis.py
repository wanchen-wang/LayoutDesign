"""
分析：振幅 (X) 与 相对误差 (Y) 的关系
读取 0cut 和 30cut 的 CSV 文件，生成两张散点图
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 文件路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELB_DIR = PROJECT_ROOT / "ModelB_Simulated_Sampling_And_Amplitude_Fitting"
DATA_DIR = MODELB_DIR / "Analysis_Results_SwA_Lagrangian_Cut_Data"

CSV_0CUT = DATA_DIR / "analysis_results_swA_lagrangian_0cut.csv"
CSV_30CUT = DATA_DIR / "analysis_results_swA_lagrangian_30cut.csv"

# 图片输出目录
OUTPUT_DIR = Path("D:/PYTHON/layout design/Pic/A_analysis/Amplitude_vs_Error_Plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_and_analyze(csv_path, title_suffix=""):
    """读取 CSV 并返回数据"""
    if not csv_path.exists():
        print(f"错误：文件不存在 {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"✓ 读取文件: {csv_path.name} ({len(df)} 条记录)")
    return df

def plot_amplitude_vs_error(df, cut_type, output_dir):
    """绘制振幅 vs 相对误差的散点图"""
    if df is None or df.empty:
        print(f"数据为空，无法绘制 {cut_type} 图")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 提取数据
    x = df['true_h0'].values    # X 轴：真实振幅
    y = df['error_pct'].values  # Y 轴：相对误差
    
    # 散点图
    ax.scatter(x, y, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    # 添加趋势线（可选）
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f"趋势线 (y={z[0]:.3f}x+{z[1]:.3f})")
    
    # 图表设置
    ax.set_xlabel('振幅 (真实 ISW 深度差, m)', fontsize=12)
    ax.set_ylabel('相对误差 (%)', fontsize=12)
    ax.set_title(f'振幅 vs 相对误差 ({cut_type})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 统计信息
    corr = np.corrcoef(x, y)[0, 1]
    textstr = f'样本数: {len(df)}\n'
    textstr += f'相关系数: {corr:.4f}\n'
    textstr += f'振幅范围: [{x.min():.2f}, {x.max():.2f}] m\n'
    textstr += f'相对误差范围: [{y.min():.2f}, {y.max():.2f}]%'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图片
    output_file = output_dir / f"amplitude_vs_error_{cut_type}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存: {output_file}")
    
    plt.close()

def main():
    print("="*70)
    print("振幅 vs 计算误差分析")
    print("="*70)
    
    # 读取 0cut 数据
    print("\n【0cut 数据】")
    df_0cut = load_and_analyze(CSV_0CUT)
    
    # 读取 30cut 数据
    print("\n【30cut 数据】")
    df_30cut = load_and_analyze(CSV_30CUT)
    
    # 生成图表
    print("\n【生成图表】")
    if df_0cut is not None:
        plot_amplitude_vs_error(df_0cut, "0cut", OUTPUT_DIR)
    
    if df_30cut is not None:
        plot_amplitude_vs_error(df_30cut, "30cut", OUTPUT_DIR)
    
    print("\n" + "="*70)
    print(f"✓ 所有操作完成！输出目录: {OUTPUT_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()

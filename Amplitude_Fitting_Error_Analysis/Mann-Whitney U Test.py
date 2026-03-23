import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

def calculate_cohens_d(group1, group2):
    """计算效应量 Cohen's d"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    # 计算合并标准差
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def mw_u_test_and_violin_plot(base_dir):
    print(f"\n{'='*65}")
    print(f"🚀 启动推断统计诊断: 曼-惠特尼 U 检验与小提琴图分布分析")
    print(f"   目标: 突破离群值干扰，评估优化截断前后的显著性差异与效应量")
    print(f"{'='*65}")

    # 1. 加载数据 (此处对比 0% 截断与 4% 截断)
    file_baseline = f"{base_dir}\\analysis_results_swA_lagrangian_0cut.csv"
    file_optimized = f"{base_dir}\\analysis_results_swA_lagrangian_30cut.csv"
    
    try:
        df_base = pd.read_csv(file_baseline)
        df_opt = pd.read_csv(file_optimized)
    except FileNotFoundError as e:
        print(f"❌ 找不到文件: {e}")
        return

    # 提取绝对误差百分比
    # 注意：曼-惠特尼U检验针对的是绝对误差的大小分布，考察误差强度的整体下降
    err_base = df_base['error_pct'].dropna()
    err_opt = df_opt['error_pct'].dropna()

    # 2. 执行曼-惠特尼 U 检验 (双侧检验)
    stat, p_value = mannwhitneyu(err_base, err_opt, alternative='two-sided')
    
    # 3. 计算效应量 Cohen's d
    cohens_d = calculate_cohens_d(err_base, err_opt)

    # 4. 数据合并以便于 Seaborn 绘图
    df_base['Group'] = 'Baseline (0% Cut)\n(High Variance & Long Tail)'
    df_opt['Group'] = 'Optimized (30% Cut)\n(Suppressed Outliers)'
    df_combined = pd.concat([df_base[['error_pct', 'Group']], df_opt[['error_pct', 'Group']]])

    # 5. 绘制小提琴图结合内部箱线图
    plt.figure(figsize=(10, 7))
    
    # 使用 seaborn 的 violinplot，内部自带箱线图(box)，并通过 cut=0 限制核密度估计超出数据范围
    ax = sns.violinplot(x='Group', y='error_pct', data=df_combined, 
                        palette=['#e74c3c', '#2ecc71'], inner='box', cut=0, alpha=0.8)

    # 图表装饰与统计标注
    plt.title('Error Distribution Comparison: Baseline vs. Optimized Truncation\n(Violin Plot with Kernel Density Estimation)', fontsize=14, pad=15)
    plt.ylabel('Absolute Percentage Error (%)', fontsize=12)
    plt.xlabel('Integration Parameter Setting', fontsize=12)
    plt.grid(True, axis='y', linestyle=':', alpha=0.6)

    # 在图表中添加统计检验结果文本框
    stats_text = (f"Mann-Whitney U Test:\n"
                  f"U-Statistic = {stat:.1f}\n"
                  f"p-value = {p_value:.3e}\n"
                  f"Cohen's d = {cohens_d:.2f}")
    
    # 根据 p 值给出显著性结论
    if p_value < 0.05:
        stats_text += "\n\nResult: Significant Difference (p < 0.05) *"
    else:
        stats_text += "\n\nResult: No Significant Difference"

    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.5, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='center', bbox=props, weight='bold')

    plt.tight_layout()
    plt.show()

    # 6. 终端输出详细报告
    print(f"【推断统计检验报告】")
    print(f"  -> Baseline 组 (0%): 样本量 N={len(err_base)}, 中位数={np.median(err_base):.2f}%, 最大值={np.max(err_base):.2f}%")
    print(f"  -> Optimized 组 (30%): 样本量 N={len(err_opt)}, 中位数={np.median(err_opt):.2f}%, 最大值={np.max(err_opt):.2f}%")
    print(f"  -> 曼-惠特尼 U 统计量: {stat:.2f}")
    print(f"  -> p 值: {p_value:.5e} ", end="")
    if p_value < 0.05:
        print("(统计学上存在显著的误差分布差异！)")
    else:
        print("(两组误差分布无显著差异。)")
    print(f"  -> Cohen's d 效应量: {cohens_d:.2f} (量化了改善的实际物理强度)")
    print(f"{'='*65}\n")

if __name__ == "__main__":
    # 请确保此处指向你存放 0cut 和 4cut csv 文件的绝对路径
    target_path = r"D:\PYTHON\layout design\Analysis_Results_SwA_Lagrangian_Cut_Data"
    mw_u_test_and_violin_plot(target_path)
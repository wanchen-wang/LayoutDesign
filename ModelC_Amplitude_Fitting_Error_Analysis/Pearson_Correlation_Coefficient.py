import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

def calculate_correlation_metrics(file_path):
    print(f"\n{'='*65}")
    print(f"🚀 启动诊断: 积分区间长度与绝对误差的统计相关性分析")
    print(f"{'='*65}")

    # 1. 加载数据
    df = pd.read_csv(file_path)
    
    # 提取我们需要分析的两个变量：积分时长(Duration) 和 绝对误差(Abs Error)
    # 在你的数据中，duration 相当于报告中的步长/区间长度 h，abs_error 相当于 E
    h = df['duration'].values
    E = df['abs_error'].values

    # ==========================================
    # 诊断步聚 1: 斯皮尔曼等级相关系数 (Spearman's Rank Correlation)
    # 理论依据：衡量变量间的单调关系，对原始数据的幂律关系和极端离群点具有极高的鲁棒性
    # 数学公式：r_s = 1 - (6 * ΣD_i^2) / (n * (n^2 - 1))
    # ==========================================
    spearman_corr, spearman_p = spearmanr(h, E)
    
    # ==========================================
    # 诊断步骤 2: 对数变换与皮尔逊相关系数 (Pearson Correlation in Log Space)
    # 理论依据：因为 E ≈ C * h^k，取对数后 log(E) ≈ k*log(h) + log(C) 变为线性模型。
    # 数学公式：r = Σ((x_i - x_mean)*(y_i - y_mean)) / sqrt(Σ(x_i - x_mean)^2 * Σ(y_i - y_mean)^2)
    # ==========================================
    # 过滤掉可能的 0 值以避免 log(0) 报错
    valid_idx = (h > 0) & (E > 0)
    log_h = np.log10(h[valid_idx])
    log_E = np.log10(E[valid_idx])
    
    pearson_corr, pearson_p = pearsonr(log_h, log_E)

    # 计算经验收敛阶 k (即 Log-Log 空间下线性回归的斜率)
    slope, intercept = np.polyfit(log_h, log_E, 1)

    # ==========================================
    # 打印诊断报告
    # ==========================================
    print("\n【1】斯皮尔曼等级相关 (非参数单调性检验):")
    print(f"  -> Spearman r_s = {spearman_corr:.4f}")
    print(f"  -> P-Value      = {spearman_p:.3e}")
    if spearman_p < 0.05:
        print("  -> 结论: 显著相关。随着积分时间增加，误差呈现出统计学意义上的显著单调递增。")
    else:
        print("  -> 结论: 无显著单调关系 (可能由舍入误差主导)。")

    print("\n【2】对数空间皮尔逊相关 (线性与渐近行为检验):")
    print(f"  -> Pearson r = {pearson_corr:.4f}")
    print(f"  -> P-Value   = {pearson_p:.3e}")
    print(f"  -> 观测斜率 k = {slope:.4f} (对应全局截断误差的非线性放大倍率)")
    
    # 根据报告中的定性标准解释相关性强度
    abs_r = abs(pearson_corr)
    if abs_r >= 0.90: strength = "极高相关 (Very Strong) - 处于理想渐近区"
    elif abs_r >= 0.70: strength = "高相关 (Strong) - 趋势显著，存在微量噪声"
    elif abs_r >= 0.50: strength = "中度相关 (Moderate) - 稳定性受到挑战"
    elif abs_r >= 0.30: strength = "低相关 (Weak) - 算法可能进入非收敛区"
    else: strength = "极弱或无相关 (Negligible) - 舍入误差主导"
    print(f"  -> 强度评估: {strength}")

    # ==========================================
    # 绘制 Log-Log 诊断图
    # ==========================================
    plt.figure(figsize=(8, 6))
    plt.scatter(log_h, log_E, color='royalblue', alpha=0.7, edgecolors='k', label='Observed Data')
    
    # 绘制回归线
    x_fit = np.linspace(min(log_h), max(log_h), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, 'r--', linewidth=2, label=f'Linear Fit (Slope k={slope:.2f})')
    
    plt.title('Log-Log Pearson Correlation Diagnostic\nIntegration Duration vs. Absolute Error', fontsize=14)
    plt.xlabel('Log10(Integration Duration $L$)', fontsize=12)
    plt.ylabel('Log10(Absolute Error $E$)', fontsize=12)
    
    # 在图中标注统计系数
    stats_text = (f"Spearman $r_s$: {spearman_corr:.3f}\n"
                  f"Pearson $r$: {pearson_corr:.3f}\n"
                  f"Empirical Order $k$: {slope:.3f}")
    plt.text(0.05, 0.85, stats_text, transform=plt.gca().transAxes, 
             fontsize=11, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
             
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 你可以将这里替换为你的 0cut 或 4cut 的 CSV 路径进行对比
    calculate_correlation_metrics(r'D:\PYTHON\layout design\Analysis_Results_SwA_Lagrangian_Cut_Data\analysis_results_swA_lagrangian_30cut.csv')
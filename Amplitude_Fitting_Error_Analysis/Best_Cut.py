import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_truncation_sensitivity(data_dir, max_pct=40):
    print(f"\n{'='*65}")
    print(f"🚀 启动深度诊断: 截断阈值 (0%-{max_pct}%) 参数敏感性与误差分解")
    print(f"   目标路径: {data_dir}")
    print(f"   理论支撑: 偏差-方差权衡 (Bias-Variance Tradeoff) 及系统性漂移分析")
    print(f"{'='*65}")

    if not os.path.exists(data_dir):
        print(f"❌ 找不到指定的文件夹路径：{data_dir}")
        return

    thresholds = []
    rmse_list = []
    mbe_list = []
    mae_list = []

    # 1. 遍历读取指定路径下 0 到 40 的所有 CSV 文件
    for pct in range(max_pct + 1):
        file_name = f'analysis_results_swA_lagrangian_{pct}cut.csv'
        file_path = os.path.join(data_dir, file_name)
        
        if not os.path.exists(file_path):
            print(f"   ⚠️ 找不到文件 {file_name}，已跳过。")
            continue
            
        # 读取当前阈值的子组数据
        df = pd.read_csv(file_path)
        
        # 确保包含我们计算所需的基准列
        if 'dh' not in df.columns or 'true_h0' not in df.columns:
            print(f"   ⚠️ {file_name} 中缺失 'dh' 或 'true_h0' 列，已跳过。")
            continue
            
        # ⭐ 核心物理校准：计算带符号的相对误差百分比 (Percentage Error)
        # 注意：CSV 自带的 error_pct 是绝对值。为了计算 MBE 揭示系统性漂移，我们必须保留正负号！
        # 正值代表高估 (把长尾噪声当成了面积)，负值代表低估 (提前截断漏掉了真实动能)
        pe = (df['dh'] - df['true_h0']) / df['true_h0'] * 100
        
        # 2. 计算三大核心指标
        # MAE (平均绝对误差百分比): 反映基础物理精度，避免正负抵消
        mae = np.mean(np.abs(pe))
        
        # RMSE (均方根误差百分比): 探测极端拟合失效，对离群大误差施加高惩罚
        rmse = np.sqrt(np.mean(pe**2))
        
        # MBE (平均偏差误差百分比): 揭示系统性漂移方向
        mbe = np.mean(pe)
        
        thresholds.append(pct)
        mae_list.append(mae)
        rmse_list.append(rmse)
        mbe_list.append(mbe)

    if not thresholds:
        print("❌ 在该路径下未找到任何有效数据，请检查路径中是否包含生成的 CSV 文件。")
        return

    # 寻找最佳理论截断点
    best_mbe_idx = np.argmin(np.abs(mbe_list))
    best_mae_idx = np.argmin(mae_list)
    
    best_pct_mbe = thresholds[best_mbe_idx]
    best_pct_mae = thresholds[best_mae_idx]

    # 3. 绘制多维误差全景折线图
    plt.figure(figsize=(12, 7))
    
    # 绘制 MAE 曲线 (U型基础精度)
    plt.plot(thresholds, mae_list, marker='o', markersize=6, linewidth=2.5, 
             color='#2ecc71', label='MAE (Mean Absolute Error) - Accuracy Baseline')
             
    # 绘制 RMSE 曲线 (U型波动惩罚)
    plt.plot(thresholds, rmse_list, marker='s', markersize=6, linewidth=2.5, 
             color='#e74c3c', label='RMSE (Root Mean Square Error) - Outlier Penalty')
             
    # 绘制 MBE 曲线 (单调穿零系统偏差)
    plt.plot(thresholds, mbe_list, marker='^', markersize=6, linewidth=2.5, 
             color='#3498db', label='MBE (Mean Bias Error) - Systematic Drift')

    # 添加零轴辅助线 (绝对物理平衡线)
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.8, label='Zero Bias Line (Perfect Balance)')

    # 标注最优点
    plt.axvline(best_pct_mae, color='gray', linestyle=':', alpha=0.6)
    plt.scatter([best_pct_mae], [mae_list[best_mae_idx]], color='gold', s=150, zorder=5, edgecolors='k')
    plt.annotate(f'Lowest MAE\n({best_pct_mae}%, {mae_list[best_mae_idx]:.1f}%)', 
                 xy=(best_pct_mae, mae_list[best_mae_idx]), xytext=(best_pct_mae+1, mae_list[best_mae_idx]+5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5), fontsize=10)

    plt.axvline(best_pct_mbe, color='purple', linestyle=':', alpha=0.6)
    plt.scatter([best_pct_mbe], [mbe_list[best_mbe_idx]], color='magenta', s=150, zorder=5, edgecolors='k')
    plt.annotate(f'Zero MBE Crossing\n({best_pct_mbe}%, {mbe_list[best_mbe_idx]:.1f}%)', 
                 xy=(best_pct_mbe, mbe_list[best_mbe_idx]), xytext=(best_pct_mbe+1, mbe_list[best_mbe_idx]-5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5), fontsize=10)

    # 图表装饰
    plt.title('Sensitivity Analysis of Integration Truncation Threshold (0% - 40%)\nBias-Variance Tradeoff & Error Decomposition', fontsize=15, pad=15)
    plt.xlabel('Truncation Threshold (% of Maximum Velocity)', fontsize=12)
    plt.ylabel('Integration Error / Bias (%)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best', fontsize=11)
    
    # 划分理论区域
    plt.axvspan(0, max(0, best_pct_mae-2), color='red', alpha=0.05, label='Overfitting (Noise Accumulation)')
    plt.axvspan(min(40, best_pct_mae+2), 40, color='blue', alpha=0.05, label='Underfitting (Energy Loss)')
    
    plt.tight_layout()
    plt.show()

    # 4. 打印统计结论
    print(f"\n【诊断结论与参数优选】")
    print(f"  🔹 [稳定性] 最低 RMSE 出现于 {thresholds[np.argmin(rmse_list)]}% 截断处，说明此时极端失效点被最大程度压制。")
    print(f"  🔹 [准确度] 最低 MAE 出现于 {best_pct_mae}% 截断处，基础物理重建精度最高。")
    print(f"  🔹 [系统差] MBE 在 {best_pct_mbe}% 截断处最接近 0，意味着此时长尾带来的“正向面积虚增”与提前截断带来的“负向能量丢失”达到完美抵消。")
    print(f"  💡 综合建议: 如果 MAE 谷底与 MBE 穿零点非常接近，该区间即为最无懈可击的【黄金截断阈值】！")
    print(f"{'='*65}\n")

if __name__ == "__main__":
    # 使用你指定的绝对路径，前面加 'r' 防止转义字符报错
    target_path = r"D:\PYTHON\layout design\Analysis_Results_SwA_Lagrangian_Cut_Data"
    analyze_truncation_sensitivity(target_path, max_pct=40)
"""
对比脚本：用相同的 train_dirs 30个样本，对比：
1. 直接运行 process_30cut 的结果
2. 从 Model3_Unprovoke_Comparison.csv 读取对应行的结果
两种方式是否计算结果一致
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from Model3_Lagrange_Dynamic_Sampling_and_Error_Calculation import process_30cut

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = Path(__file__).resolve().parent
BASE_DATA_DIR = PROJECT_ROOT / "ModelA_Virtual_Internal_Solitary_Wave_Data_Generation" / "V_Wave_Data"
MODEL3_ANALYSIS_DIR = MODULE_DIR / "Analysis_Bayesian_Opt_Model3_Data"

# Model3 参数（与 Model4 相同）
w_c_threshold = 9999
V_target = 0.276
zeta_target = -37.2
f_s = 0.2

# 数据分割（与 Model4 相同）
DATA_SPLIT_SEED = 42
TRAIN_SAMPLE_SIZE = 30

all_run_dirs = sorted([
    os.path.join(BASE_DATA_DIR, d)
    for d in os.listdir(BASE_DATA_DIR)
    if os.path.isdir(os.path.join(BASE_DATA_DIR, d))
])

train_pool = all_run_dirs[:100]
rng_split = np.random.default_rng(DATA_SPLIT_SEED)
train_pick = rng_split.choice(100, size=TRAIN_SAMPLE_SIZE, replace=False)
train_dirs = [train_pool[i] for i in sorted(train_pick)]

print("="*70)
print("对比 Model4 中用的 train_dirs（30个样本）")
print("="*70)
print(f"\n选中的 train_dirs 索引（前 100 个中）: {sorted(train_pick)}")
print(f"首个数据目录: {os.path.basename(train_dirs[0])}")
print(f"末个数据目录: {os.path.basename(train_dirs[-1])}\n")

# 方式1：直接运行 process_30cut
print("方式1：直接运行 process_30cut")
print("-" * 70)
direct_results = []
for idx, run_dir in enumerate(train_dirs, 1):
    data_dir_name = os.path.basename(run_dir)
    try:
        result_dict = process_30cut(w_c_threshold, V_target, zeta_target, f_s, run_dir, return_diagnostic_data=True)
        rel_error = result_dict['rel_error']
        direct_results.append(rel_error)
    except Exception as e:
        print(f"[{idx}/{len(train_dirs)}] {data_dir_name}: 错误 {e}")

direct_mean = np.mean(direct_results)
direct_std = np.std(direct_results, ddof=1)
print(f"✓ 处理完成：{len(direct_results)} 个样本")
print(f"  平均相对误差: {direct_mean:.6f}%")
print(f"  标准差: {direct_std:.6f}%")

# 方式2：从 CSV 读取
print("\n方式2：从 Model3_Unprovoke_Comparison.csv 读取")
print("-" * 70)
csv_path = MODEL3_ANALYSIS_DIR / "Model3_Unprovoke_Comparison.csv"
if not csv_path.exists():
    print(f"✗ CSV 文件不存在: {csv_path}")
else:
    df_csv = pd.read_csv(csv_path)
    print(f"✓ 读取 CSV: 共 {len(df_csv)} 行")
    
    # 检查 CSV 的 wave_id 顺序和 train_dirs 的对应关系
    csv_wave_ids = df_csv['wave_id'].values
    train_wave_ids = [os.path.basename(d) for d in train_dirs]
    
    # 找出 train_wave_ids 在 CSV 中的行号
    csv_indices = []
    csv_values = []
    for wave_id in train_wave_ids:
        if wave_id in csv_wave_ids:
            csv_idx = np.where(csv_wave_ids == wave_id)[0][0]
            csv_indices.append(csv_idx)
            csv_values.append(df_csv.iloc[csv_idx]['error_pct'])
    
    if len(csv_values) == len(train_dirs):
        csv_mean = np.mean(csv_values)
        csv_std = np.std(csv_values, ddof=1)
        print(f"✓ 从 CSV 读取：{len(csv_values)} 个样本")
        print(f"  平均相对误差: {csv_mean:.6f}%")
        print(f"  标准差: {csv_std:.6f}%")
        
        # 对比两种方式
        print("\n" + "="*70)
        print("对比分析")
        print("="*70)
        print(f"直接运行平均值: {direct_mean:.6f}%")
        print(f"CSV 读取平均值: {csv_mean:.6f}%")
        print(f"差异: {abs(direct_mean - csv_mean):.6f}%")
        
        # 逐个对比
        print("\n逐个样本对比:")
        max_diff = 0
        for i, (direct_val, csv_val) in enumerate(zip(direct_results, csv_values)):
            diff = abs(direct_val - csv_val)
            if diff > max_diff:
                max_diff = diff
            if diff > 0.001:
                print(f"  [{i+1}] {train_wave_ids[i]}: 直接={direct_val:.6f}%, CSV={csv_val:.6f}%, 差异={diff:.6f}%")
        
        print(f"\n最大样本差异: {max_diff:.6f}%")
    else:
        print(f"✗ CSV 中只找到 {len(csv_values)} 个样本（期望 {len(train_dirs)} 个）")

# 方式3：Model3_Analysis 的方式（从 CSV 取前 100 行，再随机抽 30 行）
print("\n方式3：Model3_Analysis 的方式（从 CSV 前 100 行随机抽 30 行）")
print("-" * 70)
if csv_path.exists():
    df_100 = df_csv.head(100)
    np.random.seed(42)
    sample_indices = np.random.choice(df_100.index, size=30, replace=False)
    df_sample = df_100.loc[sample_indices]
    
    analysis_mean = df_sample['error_pct'].mean()
    analysis_std = df_sample['error_pct'].std(ddof=1)
    print(f"✓ 从 CSV 前 100 行随机抽 30 行: ")
    print(f"  平均相对误差: {analysis_mean:.6f}%")
    print(f"  标准差: {analysis_std:.6f}%")
    print(f"\n这与之前你看到的 5.044% 相符吗？")

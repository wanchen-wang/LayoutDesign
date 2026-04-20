"""
查看 Model3_Analysis 从 CSV 中随机选择的具体行，
与 Model4 的 train_dirs 的对应关系
"""
import pandas as pd
import numpy as np
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
csv_path = MODULE_DIR / "Analysis_Bayesian_Opt_Model3_Data" / "Model3_Unprovoke_Comparison.csv"

df = pd.read_csv(csv_path)
df_100 = df.head(100)

# Model3_Analysis 的随机抽样（种子 42）
np.random.seed(42)
sample_indices = np.random.choice(df_100.index, size=30, replace=False)

print("Model3_Analysis 从 CSV 前 100 行中随机选择的行索引：")
print(sorted(sample_indices))

print("\n这 30 行对应的 wave_id：")
selected_wave_ids = df_100.loc[sample_indices, 'wave_id'].values
selected_errors = df_100.loc[sample_indices, 'error_pct'].values

for idx, (wid, err) in enumerate(zip(selected_wave_ids, selected_errors)):
    print(f"  [{idx+1:2d}] {wid}: {err:.4f}%")

print(f"\nCSV 第 0-29 行的 wave_id：")
for i in range(min(30, len(df))):
    print(f"  [{i+1:2d}] {df.iloc[i]['wave_id']}: {df.iloc[i]['error_pct']:.4f}%")

# 查看 Model4 的 train_dirs 对应的 wave_id
import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DATA_DIR = PROJECT_ROOT / "ModelA_Virtual_Internal_Solitary_Wave_Data_Generation" / "V_Wave_Data"

all_run_dirs = sorted([
    os.path.join(BASE_DATA_DIR, d)
    for d in os.listdir(BASE_DATA_DIR)
    if os.path.isdir(os.path.join(BASE_DATA_DIR, d))
])

train_pool = all_run_dirs[:100]
DATA_SPLIT_SEED = 42
rng_split = np.random.default_rng(DATA_SPLIT_SEED)
train_pick = rng_split.choice(100, size=30, replace=False)
train_dirs = [train_pool[i] for i in sorted(train_pick)]
train_wave_ids = [os.path.basename(d) for d in train_dirs]

print(f"\nModel4 的 train_dirs 选择的 wave_id：")
for idx, wid in enumerate(train_wave_ids, 1):
    print(f"  [{idx:2d}] {wid}")

# 对比：这两个列表是否一样？
print(f"\nModel3_Analysis 选择的 wave_ids 列表（按 df_100 行索引排序后）：")
selected_wave_ids_sorted = df_100.loc[sorted(sample_indices), 'wave_id'].values
for idx, wid in enumerate(selected_wave_ids_sorted, 1):
    print(f"  [{idx:2d}] {wid}")

print(f"\n\n对比结果：")
print(f"Model3_Analysis 选择的 30 行是否与 Model4 的 train_dirs 相同？")
if list(selected_wave_ids_sorted) == train_wave_ids:
    print("✓ 是的，完全相同！")
else:
    print("✗ 否，不同！")
    print("\n差异分析：")
    set_m3 = set(selected_wave_ids_sorted)
    set_m4 = set(train_wave_ids)
    print(f"  只在 Model3 中: {set_m3 - set_m4}")
    print(f"  只在 Model4 中: {set_m4 - set_m3}")

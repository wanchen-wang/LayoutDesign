import pandas as pd
import glob
import os
from tabulate import tabulate
from pathlib import Path

# 读取 Model4优化参数 的 loss 数据，找到 loss 最小的 10 条记录，并进行对比分析

# 指定目录路径
data_dir = Path(__file__).resolve().parent / "Analysis_Bayesian_Opt_Model4_Data"

# 读取所有 CSV 文件
csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
print(f"找到的 CSV 文件数: {len(csv_files)}\n")

# 合并所有 CSV 文件
df_list = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    df_list.append(df)
    print(f"✓ {os.path.basename(csv_file):40} 行数: {len(df):3}")

# 合并所有数据
all_data = pd.concat(df_list, ignore_index=True)
print(f"\n总行数: {len(all_data)}\n")

# 按 loss 排序，找到最小的 10 条
top_10_min_loss = all_data.nsmallest(10, 'loss').reset_index(drop=True)

# 添加排名列
top_10_min_loss.insert(0, '排名', range(1, 11))

print("=" * 120)
print("Loss 最小的 10 条数据对比表")
print("=" * 120)

# 核心列的对比表格
core_cols = ['排名', 'eval_id', 'loss', 'MAE_pct', 'CI_width', 'w_c_threshold', 'zeta_target', 'V_ratio', 'f_s', 'V_target']
display_df = top_10_min_loss[core_cols].copy()

# 格式化数值，保留合适的小数位
display_df_formatted = display_df.copy()
for col in display_df.columns:
    if col not in ['排名', 'eval_id', 'f_s']:
        display_df_formatted[col] = display_df_formatted[col].apply(lambda x: f"{x:.6f}")

print(tabulate(display_df_formatted, headers='keys', tablefmt='grid', showindex=False))

print("\n" + "=" * 120)
print("完整数据对比表")
print("=" * 120)

# 格式化所有列
full_df_formatted = top_10_min_loss.copy()
for col in full_df_formatted.columns:
    if col not in ['排名', 'eval_id', 'f_s', 'w_MAE', 'w_CI', 'is_penalty']:
        full_df_formatted[col] = full_df_formatted[col].apply(lambda x: f"{x:.6f}")

print(tabulate(full_df_formatted, headers='keys', tablefmt='grid', showindex=False))

# 导出到 Excel
output_dir = Path(__file__).resolve().parent / "Analysis_Bayesian_Opt_Model5_Data"
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "model5_Top_10_Min_Loss.xlsx"
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    top_10_min_loss.to_excel(writer, index=False, sheet_name='Top_10_Min_Loss')
    
    # 调整列宽
    worksheet = writer.sheets['Top_10_Min_Loss']
    for column in worksheet.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 50)
        worksheet.column_dimensions[column_letter].width = adjusted_width

print(f"\n✓ 已保存结果到: {output_file}")

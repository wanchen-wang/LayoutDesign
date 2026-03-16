"""
Batch executor for single_w_A analysis over multiple data groups.

Allows selection of start/end folder indices from the `v_wave_data`
directory. Folders are assumed to be lexicographically ordered
timestamps. Default behavior runs only the first folder.

Usage:
    python Execute_single_w_A.py            # prompt for start/end or use defaults
    python Execute_single_w_A.py 2 5       # run analyses for groups 2 through 5

Indices are 1-based.
"""

import os
import sys
import pandas as pd

# ensure current directory is on path so that we can import local modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from single_w_A import run_single


def list_groups(base_dir="v_wave_data"):
    if not os.path.isdir(base_dir):
        return []
    items = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    items.sort()
    return items


def execute_range(start_idx=1, end_idx=1, base_dir="v_wave_data", output_file="analysis_results_swA.xlsx"):
    groups = list_groups(base_dir)
    n = len(groups)
    if n == 0:
        print(f"未找到任何数据组 (目录 {base_dir} 为空)")
        return

    # clamp indices
    start = max(1, start_idx)
    end = min(end_idx, n)
    if start > end:
        print("起始索引大于结束索引，取消执行")
        return

    # Load existing results if file exists
    if os.path.exists(output_file):
        try:
            df_existing = pd.read_excel(output_file)
            existing_groups = set(df_existing['group'].tolist())
        except Exception as e:
            print(f"读取现有 Excel 文件失败: {e}")
            existing_groups = set()
    else:
        existing_groups = set()

    results = []

    for idx in range(start, end + 1):
        group = groups[idx - 1]
        if group in existing_groups:
            print(f"组 {group} 已经分析过，跳过")
            continue
        path = os.path.join(base_dir, group)
        print(f"\n*** 运行组 {idx}/{n}: {group} ***")
        try:
            result = run_single(path)
            result['group'] = group
            results.append(result)
        except Exception as e:
            print(f"组 {group} 处理失败: {e}")

    if results:
        df_new = pd.DataFrame(results)
        # Reorder columns for better readability
        columns = ['group', 'dh', 'true_h0', 'error_pct', 't_w0', 't_U']
        df_new = df_new[columns]
        
        if os.path.exists(output_file):
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        
        df_combined.to_excel(output_file, index=False)
        print(f"\n结果已保存到 {output_file}")
    else:
        print("没有新的结果需要保存")


if __name__ == "__main__":
    groups = list_groups()
    total = len(groups)

    if total == 0:
        print("没有可用的数据组，请先运行 v_wave 生成数据")
        sys.exit(1)

    # parse command-line args
    if len(sys.argv) >= 3:
        try:
            s = int(sys.argv[1])
            e = int(sys.argv[2])
        except ValueError:
            print("参数必须为整数，格式: start end")
            sys.exit(1)
    else:
        # interactive prompt with defaults
        s = 1
        e = 1
        inp = input(f"起始组索引 (1-{total}) [{s}]: ")
        if inp.strip():
            s = int(inp)
        inp = input(f"结束组索引 (1-{total}) [{e}]: ")
        if inp.strip():
            e = int(inp)

    execute_range(s, e)

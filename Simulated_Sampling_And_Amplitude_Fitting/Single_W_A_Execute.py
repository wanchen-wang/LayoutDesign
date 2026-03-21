"""
Batch executor for analysis over multiple data groups.

Allows selection of analysis method and start/end folder indices from the `V_Wave_Data`
directory. Folders are assumed to be lexicographically ordered timestamps.

Usage:
    python Single_W_A_Execute.py            # interactive prompt for method and range
    python Single_W_A_Execute.py 2 5       # run analyses for groups 2-5 (prompts for method)

Indices are 1-based.
Available methods:
    1: Single_W_A
    2: Single_W_A_Lagrangian
"""

import os
import sys
import pandas as pd

# ensure current directory is on path so that we can import local modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from Single_W_A import run_single as run_single_swA
from Single_W_A_Lagrangian import run_single as run_single_lagrangian


def list_groups(base_dir="D:\\PYTHON\\layout design\\V_Wave_Data"):
    if not os.path.isdir(base_dir):
        return []
    items = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    items.sort()
    return items


def execute_range(start_idx=1, end_idx=1, method='single_w_A', base_dir="D:\\PYTHON\\layout design\\V_Wave_Data", output_file=None):
    """
    Execute batch analysis over a range of data groups.
    
    Parameters
    ----------
    start_idx : int
        Starting group index (1-based)
    end_idx : int
        Ending group index (1-based)
    method : str
        Analysis method: 'single_w_A' or 'single_w_A_lagrangian'
    base_dir : str
        Base directory containing data groups
    output_file : str, optional
        Output CSV file path. If None, auto-generated based on method.
    """
    # Auto-generate output filename if not provided
    if output_file is None:
        if method == 'Single_W_A':
            output_file = "D:\\PYTHON\\layout design\\Analysis_Results_SwA_Lagrangian_Cut_Data\\analysis_results_swA.csv"
        elif method == 'Single_W_A_Lagrangian':
            output_file = "D:\\PYTHON\\layout design\\Analysis_Results_SwA_Lagrangian_Cut_Data\\analysis_results_swA_lagrangian_0cut.csv"
        else:
            output_file = f"analysis_results_{method}.csv"
    
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
    df_existing = None
    if os.path.exists(output_file):
        try:
            df_existing = pd.read_csv(output_file)
            existing_groups = set(df_existing['group'].tolist())
        except Exception as e:
            print(f"读取现有 CSV 文件失败: {e}")
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
        print(f"\n*** 运行组 {idx}/{n}: {group} (方法: {method}) ***")
        try:
            # 根据方法调用对应的分析函数
            if method == 'Single_W_A':
                result = run_single_swA(path)
            elif method == 'Single_W_A_Lagrangian':
                result = run_single_lagrangian(path)
            else:
                raise ValueError(f"未知方法: {method}")
            
            result['group'] = group
            results.append(result)
        except Exception as e:
            print(f"组 {group} 处理失败: {e}")
            import traceback
            traceback.print_exc()

    if results:
        df_new = pd.DataFrame(results)
        # Only keep specified columns
        columns_to_save = ['group', 'dh', 'true_h0', 'error_pct', 't_w0', 't_U']
        # Only include columns that exist
        columns_to_save = [col for col in columns_to_save if col in df_new.columns]
        df_new = df_new[columns_to_save]
        
        if df_existing is not None:
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        
        df_combined.to_csv(output_file, index=False)
        print(f"\n结果已保存到 {output_file}")
    else:
        print("没有新的结果需要保存")


if __name__ == "__main__":
    groups = list_groups()
    total = len(groups)

    if total == 0:
        print("没有可用的数据组，请先运行 v_wave 生成数据")
        sys.exit(1)

    # 可用的分析方法
    methods = ['Single_W_A', 'Single_W_A_Lagrangian']
    
    # 选择分析方法
    print("\n可用的分析方法:")
    for i, m in enumerate(methods, 1):
        print(f"{i}: {m}")
    
    method = 'Single_W_A'
    while True:
        try:
            method_inp = input(f"\n请选择分析方法 (1-{len(methods)}) [1]: ").strip()
            if not method_inp:
                method = methods[0]
                break
            method_idx = int(method_inp) - 1
            if 0 <= method_idx < len(methods):
                method = methods[method_idx]
                break
            else:
                print(f"请输入 1-{len(methods)} 之间的数字")
        except ValueError:
            print("请输入有效的整数")
    
    print(f"已选择方法: {method}\n")

    # parse command-line args for start/end indices
    if len(sys.argv) >= 3:
        try:
            s = int(sys.argv[1])
            e = int(sys.argv[2])
        except ValueError:
            print("参数必须为整数，格式: python single_w_A_Execute.py start end")
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

    execute_range(s, e, method=method)

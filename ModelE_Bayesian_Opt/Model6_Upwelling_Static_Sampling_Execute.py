"""
Model5 Upwelling Static Sampling Execute
批量执行 Model5_Upwelling_Static_Sampling 分析，参考 Single_W_A_Execute.py 的框架
"""

import os
import sys
import pandas as pd
from pathlib import Path

# ensure current directory is on path so that we can import local modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from Model6_Upwelling_Static_Sampling import run_single as run_single_model5


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V_WAVE_DATA_DIR = PROJECT_ROOT / "ModelA_Virtual_Internal_Solitary_Wave_Data_Generation" / "V_Wave_Data"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "Analysis_Bayesian_Opt_Model5_Data"


def list_groups(base_dir=DEFAULT_V_WAVE_DATA_DIR):
    if not os.path.isdir(base_dir):
        return []
    items = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    items.sort()
    return items


def execute_range(start_idx=1, end_idx=1, base_dir=DEFAULT_V_WAVE_DATA_DIR, output_file=None):
    """
    批量执行 Model5 分析。

    Parameters
    ----------
    start_idx : int
        起始组索引（1-based）
    end_idx : int
        结束组索引（1-based）
    base_dir : str
        包含数据组的基础目录
    output_file : str, optional
        输出 CSV 文件路径，若为 None 则自动生成
    """
    # 自动生成输出文件名
    if output_file is None:
        output_file = DEFAULT_RESULTS_DIR / "model6_upwelling_static_sampling_results.csv"
    
    groups = list_groups(base_dir)
    n = len(groups)
    if n == 0:
        print(f"未找到任何数据组 (目录 {base_dir} 为空)")
        return

    # 限制索引范围
    start = max(1, start_idx)
    end = min(end_idx, n)
    if start > end:
        print("起始索引大于结束索引，取消执行")
        return

    # 加载现有结果（若有）
    df_existing = None
    if os.path.exists(output_file):
        try:
            df_existing = pd.read_csv(output_file)
            id_col = 'wave_id' if 'wave_id' in df_existing.columns else 'group'
            existing_groups = set(df_existing[id_col].tolist())
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
        print(f"\n*** 运行组 {idx}/{n}: {group} (Model5 Upwelling Static Sampling) ***")
        try:
            result = run_single_model5(path)
            result['group'] = group
            results.append(result)
        except Exception as e:
            print(f"组 {group} 处理失败: {e}")
            import traceback
            traceback.print_exc()

    if results:
        df_new = pd.DataFrame(results)
        
        # 重命名 'group' 列为 'wave_id'
        if 'group' in df_new.columns:
            df_new = df_new.rename(columns={'group': 'wave_id'})
        
        # 只保留指定列
        columns_to_save = [
            'wave_id', 't_w0', 't_U', 'duration', 'dh', 'true_h0', 
            'error_pct', 'upwelling_detected', 'upwelling_depth'
        ]
        columns_to_save = [col for col in columns_to_save if col in df_new.columns]
        df_new = df_new[columns_to_save]
        
        if df_existing is not None:
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
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

    print("="*70)
    print("Model6 Upwelling Static Sampling - 批量执行器")
    print("="*70)
    print(f"发现 {total} 组数据\n")

    # 从命令行参数解析起始和结束索引
    if len(sys.argv) >= 3:
        try:
            s = int(sys.argv[1])
            e = int(sys.argv[2])
        except ValueError:
            print("参数必须为整数，格式: python Model6_Execute.py start end")
            sys.exit(1)
    else:
        # 交互式提示
        s = 1
        e = 1
        inp = input(f"起始组索引 (1-{total}) [{s}]: ")
        if inp.strip():
            try:
                s = int(inp)
            except ValueError:
                print("输入无效，使用默认值 1")
                s = 1
        
        inp = input(f"结束组索引 (1-{total}) [{e}]: ")
        if inp.strip():
            try:
                e = int(inp)
            except ValueError:
                print("输入无效，使用默认值 1")
                e = 1

    print(f"\n执行范围: {s} 到 {e}\n")
    execute_range(s, e)
    print("\n=" * 70)
    print("✅ 执行完成")
    print("=" * 70)

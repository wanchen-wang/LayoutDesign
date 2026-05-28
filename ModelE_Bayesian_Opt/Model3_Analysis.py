import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
# 读取 Model4 优化参数的 loss 数据，分析相对误差的统计特性，特别是前100条数据的随机抽样分析
MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = MODULE_DIR / "Analysis_Bayesian_Opt_Model3_Hor_Data"


def _select_csv_file(csv_paths):
    print("=== 可选的数据文件 ===")
    for idx, csv_path in enumerate(csv_paths, start=1):
        print(f"{idx}: {csv_path.name}")
        print(f"   路径: {csv_path}")

    while True:
        try:
            choice = int(input(f"\n请选择要分析的文件 (1-{len(csv_paths)}): ").strip())
            if 1 <= choice <= len(csv_paths):
                return csv_paths[choice - 1]
            print(f"请输入 1 到 {len(csv_paths)} 之间的数字。")
        except ValueError:
            print("请输入有效数字。")


def _load_relative_errors(df):
    if "error_pct" in df.columns:
        return df["error_pct"].astype(float)
    if "rel_error" in df.columns:
        return df["rel_error"].astype(float)
    if "abs_error" in df.columns and "true_h0" in df.columns:
        return (df["abs_error"].astype(float) / df["true_h0"].astype(float)) * 100.0
    raise KeyError(
        "CSV 文件必须包含 'error_pct'、'rel_error' 或 ('abs_error' 和 'true_h0') 其中之一。"
    )


def main():
    if not DATA_DIR.exists() or not DATA_DIR.is_dir():
        print(f"错误：目录不存在: {DATA_DIR}")
        return

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print(f"错误：目录中未找到任何 CSV 文件: {DATA_DIR}")
        return

    selected_csv = _select_csv_file(csv_files)
    df = pd.read_csv(selected_csv)

    try:
        relative_errors = _load_relative_errors(df)
    except KeyError as exc:
        print(f"错误：{exc}")
        return

    print(f"\n=== {selected_csv.name} 相对误差统计分析 ===")
    print(f"总数据行数: {len(relative_errors)}")

    df_100 = relative_errors.head(100)
    print(f"取前100条数据: {len(df_100)}")

    sample_size = min(30, len(df_100))
    if sample_size == 0:
        print("错误：可用数据不足，无法抽样。")
        return

    # 使用与 Model4 完全相同的新版 Generator 和种子
    rng = np.random.default_rng(42)

    # 抽取样本
    sample_indices = rng.choice(df_100.index, size=sample_size, replace=False)
    df_sample = df_100.loc[sample_indices]

    print(f"随机抽取{sample_size}条数据 (种子42): {len(df_sample)}")

    mean_error = df_sample.mean()
    std_error = df_sample.std(ddof=1)
    n = len(df_sample)

    print("\n=== 相对误差统计分析 ===")
    print(f"平均相对误差: {mean_error:.4f}%")
    print(f"标准差: {std_error:.4f}%")
    print(f"样本数量: {n}")

    if n > 1:
        confidence_level = 0.95
        degrees_of_freedom = n - 1
        t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
        margin_of_error = t_value * (std_error / np.sqrt(n))
        ci_lower = mean_error - margin_of_error
        ci_upper = mean_error + margin_of_error
        print("\n=== 95% 置信区间 ===")
        print(f"置信区间: [{ci_lower:.4f}%, {ci_upper:.4f}%]")
        print(f"置信区间宽度: {ci_upper - ci_lower:.4f}%")
    else:
        print("\n样本数量不足，无法计算 95% 置信区间。")


if __name__ == "__main__":
    main()

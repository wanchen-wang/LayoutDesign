import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELB_DIR = PROJECT_ROOT / "ModelB_Simulated_Sampling_And_Amplitude_Fitting"


def find_csv_files(base_dir: Path) -> list[Path]:
    """查找指定目录下所有的 CSV 文件，包括子目录。"""
    csv_files = []
    for csv_path in base_dir.rglob("*.csv"):
        # 过滤掉一些临时文件或隐藏文件
        if not csv_path.name.startswith(".") and not csv_path.name.startswith("__"):
            csv_files.append(csv_path)
    return sorted(csv_files)


def load_relative_error_data(csv_path: Path) -> pd.DataFrame:
    """读取 CSV，并统一生成相对误差百分比列。"""
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV 没有数据: {csv_path}")

    if "error_pct" in df.columns:
        df["relative_error_pct"] = pd.to_numeric(df["error_pct"], errors="coerce")
    elif {"abs_error", "true_h0"}.issubset(df.columns):
        abs_error = pd.to_numeric(df["abs_error"], errors="coerce")
        true_h0 = pd.to_numeric(df["true_h0"], errors="coerce")
        df["relative_error_pct"] = abs_error / true_h0 * 100.0
    else:
        raise KeyError("CSV 中既没有 error_pct，也没有 abs_error/true_h0，无法计算相对误差。")

    df = df.dropna(subset=["relative_error_pct"]).copy()
    if df.empty:
        raise ValueError("相对误差列全部为空，无法绘图。")

    df = df.reset_index(drop=True)
    df["sample_index"] = df.index + 1
    return df


def plot_relative_error_scatter(csv_path: Path, output_path: Path | None = None, show: bool = True):
    """绘制相对误差散点图。"""
    df = load_relative_error_data(csv_path)

    fig, ax = plt.subplots(figsize=(12, 6.8))

    scatter = ax.scatter(
        df["sample_index"],
        df["relative_error_pct"],
        c=df["relative_error_pct"],
        cmap="YlOrRd",
        s=34,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.35,
    )

    mean_error = df["relative_error_pct"].mean()
    min_row = df.loc[df["relative_error_pct"].idxmin()]
    max_row = df.loc[df["relative_error_pct"].idxmax()]

    ax.axhline(mean_error, color="#1565c0", linestyle="--", linewidth=1.3, label=f"平均相对误差 = {mean_error:.2f}%")
    ax.scatter(
        [max_row["sample_index"]],
        [max_row["relative_error_pct"]],
        color="#7b2cbf",
        s=70,
        marker="D",
        label=f"最大误差 = {max_row['relative_error_pct']:.2f}%",
        zorder=4,
    )
    ax.scatter(
        [min_row["sample_index"]],
        [min_row["relative_error_pct"]],
        color="#2a9d8f",
        s=70,
        marker="s",
        label=f"最小误差 = {min_row['relative_error_pct']:.2f}%",
        zorder=4,
    )

    ax.set_title(f"相对误差散点图\n{csv_path.name}", fontsize=14)
    ax.set_xlabel("样本序号", fontsize=11)
    ax.set_ylabel("相对误差 (%)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(loc="upper right", fontsize=9)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("相对误差 (%)", fontsize=10)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        print(f"已保存散点图: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="根据分析结果 CSV 绘制相对误差散点图")
    parser.add_argument("--csv", type=str, default=None, help="输入 CSV 路径；不指定则进入交互式选择模式")
    parser.add_argument("--output", type=str, default=None, help="可选：输出 PNG 路径；不传则不保存")
    parser.add_argument("--no-show", action="store_true", help="不弹出窗口；通常与 --output 一起使用")
    return parser


def interactive_select_csv() -> Path:
    """交互式选择要分析的 CSV 文件。"""
    print(f"\n{'='*60}")
    print("🚀 相对误差散点图分析工具")
    print(f"{'='*60}")
    print(f"\n正在扫描 ModelB 文件夹: {MODELB_DIR}")

    csv_files = find_csv_files(MODELB_DIR)

    if not csv_files:
        print("\n❌ 未找到任何 CSV 文件！")
        raise FileNotFoundError("在 ModelB 文件夹中未找到任何 CSV 文件")

    print(f"\n✅ 找到 {len(csv_files)} 个 CSV 文件:\n")

    # 按目录分组显示
    dir_groups = {}
    for csv_file in csv_files:
        parent_dir = csv_file.parent.name
        if parent_dir not in dir_groups:
            dir_groups[parent_dir] = []
        dir_groups[parent_dir].append(csv_file)

    # 显示分组后的文件
    all_files = []
    idx = 1
    for dir_name, files in dir_groups.items():
        print(f"📂 {dir_name}:")
        for csv_file in files:
            print(f"   {idx:2d}. {csv_file.name}")
            all_files.append(csv_file)
            idx += 1
        print()

    # 让用户选择
    while True:
        try:
            choice = input(f"请选择要分析的 CSV 文件 (1-{len(all_files)}): ")
            idx = int(choice) - 1
            if 0 <= idx < len(all_files):
                selected_csv = all_files[idx]
                print(f"\n✓ 已选择: {selected_csv}")
                return selected_csv
            else:
                print(f"请输入 1-{len(all_files)} 之间的数字")
        except ValueError:
            print("请输入有效的数字")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"找不到 CSV 文件: {csv_path}")
    else:
        csv_path = interactive_select_csv()

    output_path = Path(args.output) if args.output else None
    plot_relative_error_scatter(csv_path=csv_path, output_path=output_path, show=not args.no_show)


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "Analysis_Results_SwA_Lagrangian_Cut_Data" / "analysis_results_swA_lagrangian_30cut.csv"


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
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV), help="输入 CSV 路径")
    parser.add_argument("--output", type=str, default=None, help="可选：输出 PNG 路径；不传则不保存")
    parser.add_argument("--no-show", action="store_true", help="不弹出窗口；通常与 --output 一起使用")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV 文件: {csv_path}")

    output_path = Path(args.output) if args.output else None
    plot_relative_error_scatter(csv_path=csv_path, output_path=output_path, show=not args.no_show)


if __name__ == "__main__":
    main()
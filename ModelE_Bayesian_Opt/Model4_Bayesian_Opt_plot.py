import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PENALTY_LOSS = 9999.0
WC_NO_TRIGGER = 9999.0
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_ANALYSIS_DIR = MODULE_DIR / "Analysis_Bayesian_Opt_Model4_Data"
DEFAULT_PIC_DIR = PROJECT_ROOT / "Pic" / "Bayesian_opt"
HISTORY_CSV_NAME = "model4_bayesopt_history_Continuous_4.csv"
OUTPUT_PNG_NAME = "model4_metrics_vs_5features_Continuous_4.png"
TIME_LOSS_PNG_NAME = "model4_loss_vs_eval_Continuous_4.png"
FEATURE_COLS = ["w_c_threshold", "zeta_target", "V_ratio", "V_target", "f_s"]
EMPTY_HINT = "无数据（已排除 w_c=9999）"


def _filter_feature_data(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """按特征列过滤用于绘图的数据。

    目前仅对 w_c_threshold 子图过滤掉“永不触发”值（9999）。
    """
    if col == "w_c_threshold":
        return df[~np.isclose(df["w_c_threshold"].astype(float), WC_NO_TRIGGER, rtol=0, atol=1e-6)].copy()
    return df.copy()


def _draw_single_subplot(ax, df: pd.DataFrame, col: str) -> None:
    """绘制单个特征与 loss 的散点子图。"""
    dfx = _filter_feature_data(df, col)
    if dfx.empty:
        ax.text(
            0.5,
            0.5,
            EMPTY_HINT,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.set_xlabel(col)
        ax.set_ylabel("loss")
        ax.set_title(f"{col} vs loss")
        return

    ax.scatter(
        dfx[col].values,
        dfx["loss"].values,
        s=45,
        alpha=0.9,
        color="tab:red",
        edgecolors="none",
        label="loss",
    )
    ax.set_xlabel(col)
    ax.set_ylabel("loss")
    ax.set_title(f"{col} vs loss")
    ax.grid(alpha=0.3)
    ax.legend()


def plot_metrics_vs_features(df: pd.DataFrame, output_png: str) -> None:
    """将五个特征分别与 loss 关系绘制到 3x2 子图中。"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 14))
    axes = axes.flatten()

    for idx, col in enumerate(FEATURE_COLS):
        _draw_single_subplot(axes[idx], df, col)

    axes[5].axis("off")
    fig.suptitle("Model4: Loss vs Five Features", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_loss_vs_eval(df: pd.DataFrame, output_png: str) -> None:
    """绘制排序后的 Loss 分布图：从大到小排列。"""
    # 1. 提取所有 loss 
    all_losses = df["loss"].values
    
    # 2. 核心逻辑：进行从大到小的排序
    # np.sort 默认是从小到大，[::-1] 实现反转，变为从大到小
    sorted_losses = np.sort(all_losses)[::-1]
    
    # 3. 横坐标改为简单的序号 (1, 2, 3...)
    x = np.arange(1, len(sorted_losses) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制散点，展示每一个点的分布
    ax.scatter(
        x,
        sorted_losses,
        s=45,
        alpha=0.8,
        color="tab:blue",
        edgecolors="none",
        label="Sorted Trials",
    )
    
    # 绘制辅助线，连接所有点，更清晰地观察“下降坡度”
    ax.plot(x, sorted_losses, color="tab:blue", alpha=0.4, linestyle='--')

    # 设置坐标轴标签
    ax.set_xlabel("Trial Rank (Sorted by Loss)")
    ax.set_ylabel("Loss")
    ax.set_title("Model4: Sorted Loss Distribution (Max to Min)")
    
    # 在图上标注出最小值（最右侧的点）
    best_loss = sorted_losses[-1]
    ax.annotate(f'Best Loss: {best_loss:.4f}', 
                xy=(x[-1], best_loss), 
                xytext=(x[-1] - 20, best_loss + 0.5))
                # arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    ax.grid(alpha=0.3, linestyle=':')
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_history_csv(analysis_dir: str) -> pd.DataFrame:
    """读取优化历史 CSV。"""
    csv_path = os.path.join(analysis_dir, HISTORY_CSV_NAME)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"未找到优化历史文件：{csv_path}\n"
            "请先运行 Model4_Model4_Bayesian_Opt.py 生成结果。"
        )
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError("未获取到有效优化记录，无法绘图。")
    return df


def _filter_valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """过滤惩罚样本，只保留有效 loss 数据。"""
    if "is_penalty" in df.columns:
        df_plot = df[df["is_penalty"] != 1].copy()
    else:
        df_plot = df[df["loss"] < PENALTY_LOSS].copy()

    if df_plot.empty:
        raise RuntimeError("全部记录均为惩罚值（9999），无法绘制有效 loss 图。")
    return df_plot


def parse_args():
    parser = argparse.ArgumentParser(
        description="读取已有的 Model4 贝叶斯优化结果并可视化。"
    )
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default=DEFAULT_ANALYSIS_DIR,
        help="Model4 优化结果目录（包含 model4_bayesopt_history.csv）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_PIC_DIR,
        help="图像输出目录",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    analysis_dir = args.analysis_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("\n================= 读取优化结果并开始绘图 =================")
    df = _load_history_csv(analysis_dir)
    df_plot = _filter_valid_rows(df)

    csv_path = os.path.join(analysis_dir, HISTORY_CSV_NAME)
    trend_png = os.path.join(output_dir, OUTPUT_PNG_NAME)
    time_loss_png = os.path.join(output_dir, TIME_LOSS_PNG_NAME)
    plot_metrics_vs_features(df_plot, trend_png)
    plot_loss_vs_eval(df_plot, time_loss_png)

    best_row = df_plot.loc[df_plot["loss"].idxmin()]
    print("\n[+] 历史记录中的最优参数：")
    print(f"  w_c_threshold : {best_row['w_c_threshold']:.3f}")
    print(f"  zeta_target   : {best_row['zeta_target']:.3f}")
    print(f"  V_ratio       : {best_row['V_ratio']:.3f}")
    print(f"  f_s           : {best_row['f_s']:.3f}")
    print(f"  best loss     : {best_row['loss']:.4f}")
    print(f"  有效样本数     : {len(df_plot)} / {len(df)}")

    print("\n[+] 可视化输出文件：")
    print(f"  - {csv_path}")
    print(f"  - {trend_png}")
    print(f"  - {time_loss_png}")


if __name__ == "__main__":
    main()

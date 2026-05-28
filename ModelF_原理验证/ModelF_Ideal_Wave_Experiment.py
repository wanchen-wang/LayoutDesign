"""Run ModelF ideal-wave sampling experiment.

Generates 20 self-consistent ideal internal solitary waves, applies A/B1/B2/B3
sampling strategies, and writes the error table, summary, and distribution
plot.
"""

import matplotlib.pyplot as plt

from ModelF_Ideal_Wave_Generate import OUT_DIR, generate_batch
from ModelF_Sampling_Strategies import RESULTS_DIR, run_batch, summarize_results


def main():
    generate_batch(count=20)
    df = run_batch(limit=20, data_dir=OUT_DIR)
    summary = summarize_results(df)

    results_path = RESULTS_DIR / "ModelF_Ideal_Wave_Strategy_Errors_20.csv"
    summary_path = RESULTS_DIR / "ModelF_Ideal_Wave_Strategy_Summary.csv"
    plot_path = RESULTS_DIR / "ModelF_Ideal_Wave_Strategy_Distribution.png"

    df.to_csv(results_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    order = ["A", "B1", "B2", "B3"]
    data = [df[df["strategy"] == strategy]["rel_error"].values for strategy in order]

    fig, ax = plt.subplots(figsize=(8.5, 5.4), dpi=160)
    box = ax.boxplot(data, patch_artist=True, showfliers=False)
    colors = ["#5275b8", "#d98c42", "#7aa95c", "#9a6fb0"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    ax.set_xticklabels(order)
    ax.set_xlabel("Sampling strategy")
    ax.set_ylabel("Relative error (%)")
    ax.set_title("ModelF ideal material-layer waves: 20-wave sampling check")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    print(OUT_DIR)
    print(results_path)
    print(summary_path)
    print(plot_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

"""
Model4B_Convergence_Analysis.py
================================
功能：读取 Model4A_Batch_Run.py 生成的 Model4A_batch_results.csv，
      对 n = 1, 2, ..., N 进行随机抽样收敛分析，输出图表和收敛结果 CSV。

运行前需先执行 Model4A_Batch_Run.py 生成数据文件。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib

matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# =====================================================================
# 参数
# =====================================================================
OUT_DIR = r"D:\PYTHON\layout design\Analysis_A_Bayesian_Opt"

# 「选项 A」直接指定某个 CSV 文件（设为 None 则自动通过 PARAMS 定位）
BATCH_CSV_OVERRIDE = None
# 示例: BATCH_CSV_OVERRIDE = r"D:\PYTHON\layout design\Analysis_A_Bayesian_Opt\batch_wct0.05_V0.300_zeta-40.0_fs2.csv"

# 「选项 B」自动通过四参数定位（BATCH_CSV_OVERRIDE 为 None 时生效）
PARAMS = dict(
    w_c_threshold = 0.10,
    V_target      = 0.2,
    zeta_target   = -25,
    f_s           = 1,
)

N_REPEAT    = 300
RANDOM_SEED = 42
MAX_N       = 150
MARK_NS     = [20, 50]
PIC_DIR     = r"D:\PYTHON\layout design\Pic\Bayesian_opt"


def _param_tag(params):
    return (f"wct{params['w_c_threshold']:.2f}"
            f"_V{params['V_target']:.3f}"
            f"_zeta{params['zeta_target']:+.1f}"
            f"_fs{params['f_s']}")


def _find_batch_csv(params):
    name = f"batch_{_param_tag(params)}.csv"
    return os.path.join(OUT_DIR, name)


def _list_available_csvs():
    """列出 OUT_DIR 下所有 batch_*.csv"""
    if not os.path.isdir(OUT_DIR):
        return []
    return sorted(f for f in os.listdir(OUT_DIR) if f.startswith('batch_') and f.endswith('.csv'))


def main():
    # =====================================================================
    # 1. 确定读取的 CSV 文件
    # =====================================================================
    if BATCH_CSV_OVERRIDE:
        batch_csv = BATCH_CSV_OVERRIDE
    else:
        batch_csv = _find_batch_csv(PARAMS)

    # 列出可用的 CSV 供参考
    available = _list_available_csvs()
    if available:
        print("Analysis_A_Bayesian_Opt 中可用的批量结果：")
        for f in available:
            print(f"  {'[*]' if os.path.join(OUT_DIR, f) == batch_csv else '   '} {f}")
    print(f"\n当前使用: {os.path.basename(batch_csv)}\n")

    if not os.path.exists(batch_csv):
        raise FileNotFoundError(f"找不到批量结果文件: {batch_csv}\n"
                                "请先运行 Model4A_Batch_Run.py")

    df = pd.read_csv(batch_csv)
    valid = df[df['status'] == 'ok'].copy()
    N_VALID = len(valid)
    print(f"读取到有效数据: {N_VALID} 组（共 {len(df)} 行）")

    if N_VALID < 2:
        raise RuntimeError("有效数据不足，无法继续分析。")

    err_values = valid['rel_error'].values   # 相对误差（0~1）

    print(f"全量均值相对误差: {err_values.mean()*100:.2f}%  ±  {err_values.std()*100:.2f}%")
    print(f"全量均值绝对误差: {valid['abs_error'].mean():.4f} m  "
          f"±  {valid['abs_error'].std():.4f} m")

    # 截断到 MAX_N
    N_ANALYZE = min(MAX_N, N_VALID)
    err_values = err_values[:N_VALID]   # 保持全量均值不变

    # =====================================================================
    # 2. 随机抽样收敛分析
    # =====================================================================
    print(f"\n[Step 1] 随机抽样收敛分析（n=1~{N_ANALYZE}，每个 n 重复 {N_REPEAT} 次）...")

    rng          = np.random.default_rng(RANDOM_SEED)
    ns           = np.arange(1, N_ANALYZE + 1)
    mean_of_mean = np.zeros(N_ANALYZE)
    std_of_mean  = np.zeros(N_ANALYZE)

    for n in ns:
        samples = [
            rng.choice(err_values, size=n, replace=False).mean()
            for _ in range(N_REPEAT)
        ]
        mean_of_mean[n - 1] = np.mean(samples)
        std_of_mean[n - 1]  = np.std(samples)

    ci2 = 2 * std_of_mean   # 95% 置信区间半宽（2σ）

    # =====================================================================
    # 3. 保存收敛结果 CSV
    # =====================================================================
    out_csv = os.path.join(OUT_DIR,
                           "Model4B_convergence_result.csv")
    df_out = pd.DataFrame({
        'n_samples':           ns,
        'mean_rel_error_pct':  mean_of_mean * 100,
        'std_rel_error_pct':   std_of_mean  * 100,
        'ci95_half_pct':       ci2          * 100,   # 2σ，半宽
        'ci95_width_pct':      ci2 * 2      * 100,   # 全宽（上下各2σ）
        'ci95_lower_pct':      (mean_of_mean - ci2)  * 100,
        'ci95_upper_pct':      (mean_of_mean + ci2)  * 100,
    })
    df_out.to_csv(out_csv, index=False)
    print(f"收敛结果已保存: {out_csv}")

    # 打印关键节点
    marks = sorted(set(MARK_NS + [N_ANALYZE]))
    print("\n关键节点均值相对误差（95% 置信区间 = 均值 ± 2σ）:")
    print(f"  {'n':>5}  {'均值(%)':>8}  {'2σ(%)':>8}  {'CI下限(%)':>10}  {'CI上限(%)':>10}")
    for m in marks:
        if 1 <= m <= N_ANALYZE:
            mu  = mean_of_mean[m-1] * 100
            ci  = ci2[m-1] * 100
            print(f"  {m:>5}  {mu:>8.2f}  {ci:>8.2f}  {mu-ci:>10.2f}  {mu+ci:>10.2f}")

    # =====================================================================
    # 4. 绘图
    # =====================================================================
    full_mean_pct = err_values.mean() * 100

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(ns,
                    mean_of_mean * 100 - ci2 * 100,
                    mean_of_mean * 100 + ci2 * 100,
                    alpha=0.25, color='steelblue', label=r'95% 置信区间 ($\pm 2\sigma$)')

    ax.plot(ns, mean_of_mean * 100, color='steelblue', linewidth=1.8,
            label='随机抽样均值相对误差')

    ax.axhline(full_mean_pct, color='black', linestyle='--', linewidth=1.2,
               label=f'全量均值 ({full_mean_pct:.2f}%)')

    # ±2% 和 ±1% 收敛阈值线，找第一个置信区间半宽落入其内的 n
    threshold_marks = []   # (n_first, mu_first, color_mark, label)
    for threshold, color_band, color_mark in [(2.0, '#f7c6c6', 'crimson'),
                                               (1.0, '#c6f7c6', 'seagreen')]:
        ax.axhline(full_mean_pct + threshold, color=color_mark,
                   linestyle='-.', linewidth=0.9, alpha=0.7)
        ax.axhline(full_mean_pct - threshold, color=color_mark,
                   linestyle='-.', linewidth=0.9, alpha=0.7,
                   label=f'±{threshold:.0f}% 收敛带')
        within = np.where(ci2 * 100 <= threshold)[0]
        if len(within) > 0:
            n_first  = int(ns[within[0]])
            mu_first = mean_of_mean[within[0]] * 100
            ax.axvline(n_first, color=color_mark, linestyle='--', linewidth=1.0)
            ax.scatter([n_first], [mu_first], color=color_mark, zorder=6, s=70, marker='D')
            threshold_marks.append((n_first, mu_first, color_mark,
                                    f'n={n_first}\n首次进入±{threshold:.0f}%'))

    colors = ['tomato', 'darkorange', 'seagreen', 'purple']
    # 收集所有标注点，统一绘制在图的底部，避免重叠
    all_marks_info = []
    for i, m in enumerate(marks):
        if 1 <= m <= N_ANALYZE:
            val = mean_of_mean[m - 1] * 100
            c   = colors[i % len(colors)]
            ax.axvline(m, color=c, linestyle=':', linewidth=1.2)
            ax.scatter([m], [val], color=c, zorder=5, s=60)
            all_marks_info.append((m, val, c, f'n={m}\n{val:.2f}%'))

    # 合并 threshold 首次进入的标注，按 x 排序
    all_marks_info += threshold_marks
    all_marks_info.sort(key=lambda t: t[0])

    ax.set_xlabel('随机抽取组数 n', fontsize=12)
    ax.set_ylabel('均值相对误差 (%)', fontsize=12)
    ax.set_title('随机抽样组数 vs 均值相对误差收敛性分析', fontsize=13)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))

    # 每隔 10 个标注一次 95% 置信区间宽度（2σ×2）
    for n in range(10, N_ANALYZE + 1, 10):
        idx   = n - 1
        mu    = mean_of_mean[idx] * 100
        half  = ci2[idx] * 100
        width = 2 * half
        top   = mu + half
        bot   = mu - half
        ax.annotate('', xy=(n, top), xytext=(n, bot),
                    arrowprops=dict(arrowstyle='<->', color='gray',
                                    lw=0.8, shrinkA=0, shrinkB=0))
        ax.text(n, top + 0.05, f'{width:.2f}%',
                fontsize=7, color='gray', ha='center', va='bottom')

    ax.legend(fontsize=10)
    ax.grid(True, which='major', linestyle='--', alpha=0.4)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    plt.tight_layout()
    plt.draw()   # 确保坐标轴范围已确定

    # 统一将所有标注文字放到图下方，均匀散开，箭头指向数据点
    if all_marks_info:
        y_bot, y_top = ax.get_ylim()
        y_range  = y_top - y_bot
        y_text   = y_bot - y_range * 0.10   # 文字基线：图下方留白处
        x_left, x_right = ax.get_xlim()
        n_total  = len(all_marks_info)
        x_positions = np.linspace(x_left + (x_right - x_left) * 0.04,
                                  x_right - (x_right - x_left) * 0.04,
                                  n_total)
        for j, (m, val, c, label) in enumerate(all_marks_info):
            ax.annotate(label,
                        xy=(m, val),
                        xytext=(x_positions[j], y_text),
                        fontsize=8, color=c, ha='center', va='top',
                        annotation_clip=False,
                        arrowprops=dict(arrowstyle='->', color=c, lw=0.9,
                                        connectionstyle='arc3,rad=0.0'))
        ax.set_ylim(bottom=y_bot - y_range * 0.20)

    # 从文件名提取 tag（去掉 batch_ 前缀和 .csv 后缀）
    csv_stem = os.path.splitext(os.path.basename(batch_csv))[0]  # e.g. batch_wct0.10_V0.276...
    param_tag = csv_stem[len('batch_'):] if csv_stem.startswith('batch_') else csv_stem

    os.makedirs(PIC_DIR, exist_ok=True)
    fig_path = os.path.join(PIC_DIR, f"Model4B_convergence_{param_tag}.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\n图片已保存: {fig_path}")
    plt.show()
    print("完成。")


if __name__ == '__main__':
    main()

"""
Model5_Single_Parameter_Sensitivity.py
========================================
目的：单参数敏感性分析——固定其余三参数为默认值，逐一扫描每个参数的取值范围，
      对每组取值跑全部数据文件夹，统计均值绝对误差和均值相对误差。

四个独立（不嵌套）的扫描循环：
  1. w_c_threshold : [9999, 0.1, 0.2, 0.3, 0.4]  (9999 = 永不触发)
  2. f_s           : [0.5, 1, 3, 5, 7, 9]
  3. zeta_target   : [-23, -33, -43]
  4. V_target      : [0.1, 0.2, 0.3, 0.4]

默认值：w_c_threshold=0.1, V_target=0.3, zeta_target=-30, f_s=1
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Model3_Lagrange_Dynamic_Sampling_and_Error_Calculation import process_30cut

# =====================================================================
# 默认参数
# =====================================================================
DEFAULT = dict(
    w_c_threshold = 0.1,
    V_target      = 0.3,
    zeta_target   = -30.0,
    f_s           = 1,
)

# =====================================================================
# 扫描范围
# =====================================================================
SWEEPS = [
    ('w_c_threshold', [9999, 0.1, 0.2, 0.3, 0.4]),
    ('f_s',           [0.5, 1, 3, 5, 7, 9]),
    ('zeta_target',   [-23, -33, -43]),
    ('V_target',      [0.1, 0.2, 0.3, 0.4]),
]

# =====================================================================
# 路径
# =====================================================================
V_WAVE_DATA_DIR = r"D:\PYTHON\layout design\V_Wave_Data"
OUT_DIR         = r"D:\PYTHON\layout design\Analysis_A_Bayesian_Opt"
PIC_DIR         = r"D:\PYTHON\layout design\Pic\Bayesian_opt"


# ── 顶层 worker（多进程必须定义在模块顶层）──
def _run_one(args):
    """args = (run_dir, params_dict)"""
    run_dir, params = args
    try:
        result = process_30cut(run_data_dir=run_dir, **params)
        if isinstance(result, dict) and result.get('rel_error') is not None:
            return result['abs_error'], result['rel_error']
        return None, None
    except Exception:
        return None, None


def run_batch(all_dirs, params):
    """对所有文件夹跑一次批量，返回 (mean_abs, mean_rel, n_valid)"""
    n_workers = max(1, multiprocessing.cpu_count() - 1)
    abs_errors, rel_errors = [], []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_run_one, (d, params)) for d in all_dirs]
        for fut in as_completed(futures):
            ae, re = fut.result()
            if ae is not None:
                abs_errors.append(ae)
                rel_errors.append(re)

    if not abs_errors:
        return float('nan'), float('nan'), 0
    return float(np.mean(abs_errors)), float(np.mean(rel_errors)), len(abs_errors)


def main():
    # ── 扫描数据文件夹 ──
    all_dirs = sorted([
        os.path.join(V_WAVE_DATA_DIR, d)
        for d in os.listdir(V_WAVE_DATA_DIR)
        if os.path.isdir(os.path.join(V_WAVE_DATA_DIR, d))
           and os.path.exists(os.path.join(V_WAVE_DATA_DIR, d, 'params.json'))
    ])
    N_TOTAL = len(all_dirs)
    print(f"共发现 {N_TOTAL} 个数据文件夹")
    print(f"默认参数: {DEFAULT}\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PIC_DIR, exist_ok=True)

    # ── 结果存储 ──
    all_results = {}   # param_name -> DataFrame

    for param_name, values in SWEEPS:
        print(f"{'='*60}")
        print(f"扫描参数: {param_name}  取值: {values}")
        print(f"{'='*60}")

        rows = []
        for val in values:
            params = dict(DEFAULT)
            params[param_name] = val

            label = str(val) if val != 9999 else '9999(不触发)'
            print(f"\n  {param_name} = {label} ...")

            mean_abs, mean_rel, n_valid = run_batch(all_dirs, params)
            print(f"  有效组数={n_valid}  均值绝对误差={mean_abs:.4f}m  "
                  f"均值相对误差={mean_rel*100:.2f}%")
            rows.append({
                'param_name':   param_name,
                'param_value':  val,
                'n_valid':      n_valid,
                'mean_abs_err': mean_abs,
                'mean_rel_err': mean_rel,
            })

        df = pd.DataFrame(rows)
        all_results[param_name] = df

    # ── 保存 CSV ──
    csv_path = os.path.join(OUT_DIR, "Model5_sensitivity_results.csv")
    combined = pd.concat(all_results.values(), ignore_index=True)
    combined.to_csv(csv_path, index=False)
    print(f"\n全部结果已保存: {csv_path}")

    # ── 绘图 ──
    param_labels = {
        'w_c_threshold': '触发阈值 $w_c^{threshold}$ (m/s)',
        'f_s':           '采样频率 $f_s$ (Hz)',
        'zeta_target':   '滑翔角 $\\zeta$ (°)',
        'V_target':      '目标速度 $V_{target}$ (m/s)',
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, (param_name, values) in zip(axes, SWEEPS):
        df = all_results[param_name]

        # x 轴标签（9999 显示为 "∞"）
        x_labels = [('∞' if v == 9999 else str(v)) for v in df['param_value']]
        x = np.arange(len(x_labels))

        color_abs = '#2878b5'
        color_rel = '#d62728'

        ax2 = ax.twinx()

        bars_abs = ax.bar(x - 0.18, df['mean_abs_err'], width=0.35,
                          color=color_abs, alpha=0.75, label='均值绝对误差 (m)')
        bars_rel = ax2.bar(x + 0.18, df['mean_rel_err'] * 100, width=0.35,
                           color=color_rel, alpha=0.75, label='均值相对误差 (%)')

        # 数值标签
        for bar in bars_abs:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7.5, color=color_abs)
        for bar in bars_rel:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                     f'{h:.2f}%', ha='center', va='bottom', fontsize=7.5, color=color_rel)

        # 标注默认值
        default_val = DEFAULT[param_name]
        for xi, v in enumerate(df['param_value']):
            if v == default_val:
                ax.axvline(xi, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
                ax.text(xi, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1,
                        '默认', ha='center', va='bottom', fontsize=7.5, color='gray')

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_xlabel(param_labels.get(param_name, param_name), fontsize=10)
        ax.set_ylabel('均值绝对误差 (m)', color=color_abs, fontsize=9)
        ax2.set_ylabel('均值相对误差 (%)', color=color_rel, fontsize=9)
        ax.tick_params(axis='y', labelcolor=color_abs)
        ax2.tick_params(axis='y', labelcolor=color_rel)
        ax.set_title(f'参数敏感性：{param_name}', fontsize=11)

        # 合并图例
        lines = [bars_abs, bars_rel]
        labels = ['均值绝对误差 (m)', '均值相对误差 (%)']
        ax.legend(lines, labels, fontsize=8, loc='upper left')

    plt.suptitle('单参数敏感性分析\n'
                 f'（默认: w_c_threshold={DEFAULT["w_c_threshold"]}, '
                 f'V_target={DEFAULT["V_target"]}, '
                 f'zeta_target={DEFAULT["zeta_target"]}, '
                 f'f_s={DEFAULT["f_s"]}）',
                 fontsize=12)
    plt.tight_layout()

    fig_path = os.path.join(PIC_DIR, "Model5_sensitivity.png")
    plt.savefig(fig_path, dpi=150)
    print(f"图片已保存: {fig_path}")
    plt.show()
    print("完成。")


if __name__ == '__main__':
    main()

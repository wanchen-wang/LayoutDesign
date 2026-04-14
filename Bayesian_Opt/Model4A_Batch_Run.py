"""
Model4A_Batch_Run.py
=====================
功能：对 V_Wave_Data 下的全部数据文件夹各跑一次 process_30cut（固定参数），
      将每组 process_30cut 返回的完整数据保存为 CSV（文件名体现四参数）。
      结果存放于 Analysis_A_Bayesian_Opt 文件夹。

运行后生成的 CSV 供 Model4B_Convergence_Analysis.py 直接使用。
"""

import os
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Model3_Lagrange_Dynamic_Sampling_and_Error_Calculation import process_30cut

# =====================================================================
# 超参数
# =====================================================================
PARAMS = dict(
    w_c_threshold = 0.2,
    V_target      = 0.2,
    zeta_target   = -40,
    f_s           = 1,
)

V_WAVE_DATA_DIR = r"D:\PYTHON\layout design\V_Wave_Data"
OUT_DIR         = r"D:\PYTHON\layout design\Analysis_A_Bayesian_Opt"

# =====================================================================
# 范围选择：设置想处理第几组到第几组（1-based，含两端）
# 设为 None 表示不限制（处理全部）
# =====================================================================
IDX_START = 1     # 从第几组开始（1 = 第一个文件夹）
IDX_END   = 35  # 到第几组结束（None = 到最后一组）

# CSV 文件名体现四参数，方便日后对比不同参数的批量结果
def _make_out_csv(params):
    name = (f"batch"
            f"_wct{params['w_c_threshold']:.2f}"
            f"_V{params['V_target']:.3f}"
            f"_zeta{params['zeta_target']:+.1f}"
            f"_fs{params['f_s']}.csv")
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, name)


# ── 顶层 worker（多进程必须定义在模块顶层）──
def _run_one(run_dir):
    try:
        result = process_30cut(run_data_dir=run_dir, **PARAMS)
        if isinstance(result, dict) and result.get('rel_error') is not None:
            return {
                'run_dir':        run_dir,
                'tag':            os.path.basename(run_dir),
                'Delta_Z_calc':   result['Delta_Z_calc'],
                'Delta_Z_true':   result['Delta_Z_true'],
                'abs_error':      result['abs_error'],
                'rel_error':      result['rel_error'],
                'J':              result['J'],
                'w_max':          result['w_max'],
                'dh_raw':         result['dh_raw'],
                'doppler_factor': result['doppler_factor'],
                'W_z_meet':       result['W_z_meet'],
                'status':         'ok',
            }
        return {'run_dir': run_dir, 'tag': os.path.basename(run_dir), 'status': 'invalid'}
    except Exception as e:
        return {'run_dir': run_dir, 'tag': os.path.basename(run_dir),
                'status': f'error: {e}'}


def main():
    all_dirs = sorted([
        os.path.join(V_WAVE_DATA_DIR, d)
        for d in os.listdir(V_WAVE_DATA_DIR)
        if os.path.isdir(os.path.join(V_WAVE_DATA_DIR, d))
           and os.path.exists(os.path.join(V_WAVE_DATA_DIR, d, 'params.json'))
    ])
    N_TOTAL = len(all_dirs)

    # 按 1-based 索引截取范围
    i_start = (IDX_START - 1) if IDX_START else 0
    i_end   = IDX_END if IDX_END else N_TOTAL
    i_start = max(0, i_start)
    i_end   = min(N_TOTAL, i_end)
    selected_dirs = all_dirs[i_start:i_end]
    N_SEL = len(selected_dirs)

    print(f"共发现 {N_TOTAL} 个数据文件夹，本次处理第 {i_start+1} ~ {i_end} 组（共 {N_SEL} 组）")
    for idx, d in enumerate(selected_dirs, i_start + 1):
        print(f"  [{idx:3d}] {os.path.basename(d)}")

    n_workers = max(1, multiprocessing.cpu_count() - 1)
    out_csv   = _make_out_csv(PARAMS)
    print(f"\n使用 {n_workers} 进程并行，结果将保存至: {out_csv}")

    rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_one, d): d for d in selected_dirs}
        for fut in as_completed(futures):
            done += 1
            row = fut.result()
            rows.append(row)
            if row['status'] == 'ok':
                print(f"  [{done:3d}/{N_SEL}] {row['tag']}  "
                      f"推算={row['Delta_Z_calc']:.2f}m  真实={row['Delta_Z_true']:.2f}m  "
                      f"绝对误差={row['abs_error']:.4f}m  相对误差={row['rel_error']*100:.2f}%")
            else:
                print(f"  [{done:3d}/{N_SEL}] {row['tag']}  *** {row['status']}")

    df = pd.DataFrame(rows)
    df = df.sort_values('tag').reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"\n全部完成。结果已保存至: {out_csv}")

    valid = df[df['status'] == 'ok']
    if not valid.empty:
        print(f"有效组数: {len(valid)} / {N_SEL}")
        print(f"均值相对误差: {valid['rel_error'].mean()*100:.2f}%  "
              f"±  {valid['rel_error'].std()*100:.2f}%")
        print(f"均值绝对误差: {valid['abs_error'].mean():.4f} m  "
              f"±  {valid['abs_error'].std():.4f} m")


if __name__ == '__main__':
    main()

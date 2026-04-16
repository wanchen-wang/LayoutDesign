import os
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _annotate_point(ax, x, y, text, dx, dy, color="black", fontsize=8):
    dx = max(-24, min(24, dx))
    dy = max(-24, min(24, dy))
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=fontsize,
        color=color,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.85),
    )


def list_groups(base_dir=r"D:\PYTHON\layout design\V_Wave_Data"):
    if not os.path.isdir(base_dir):
        return []
    items = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    items.sort()
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Core simulation (mirrors process_30cut but returns full diagnostic data)
# ─────────────────────────────────────────────────────────────────────────────

def run_30cut_detailed(w_c_threshold, V_target, zeta_target, f_s, run_data_dir):
    """
    与 process_30cut 逻辑完全相同，但返回完整诊断字典用于可视化。
    """
    # ── 参数读取 ──────────────────────────────────────────────────────────────
    with open(os.path.join(run_data_dir, 'params.json'), 'r') as f:
        params = json.load(f)
    Delta_Z_true      = params.get('h0', 105.0)
    Cp                = params.get('c0')
    thermocline_depth = params.get('thermocline_depth')
    D                 = params.get('D', 1000.0)

    # ── 流场加载 ──────────────────────────────────────────────────────────────
    x_grid    = np.load(os.path.join(run_data_dir, 'x_grid.npy'))
    y_grid    = np.load(os.path.join(run_data_dir, 'y_grid.npy'))
    z_grid    = np.load(os.path.join(run_data_dir, 'z.npy'))
    W_Vel_3D  = np.load(os.path.join(run_data_dir, 'W_Vel_3D.npy'))
    W_profile = np.load(os.path.join(run_data_dir, 'W_profile.npy'))

    if z_grid[0] > z_grid[-1]:
        z_grid    = np.flip(z_grid)
        W_Vel_3D  = np.flip(W_Vel_3D, axis=2)
        W_profile = np.flip(W_profile)

    interpolator_w = RegularGridInterpolator(
        (x_grid, y_grid, z_grid), W_Vel_3D, bounds_error=False, fill_value=0.0
    )
    y_center = 0.0

    def get_flow(X, Z):
        w_c = interpolator_w([[X, y_center, Z]])[0]
        return float(w_c)

    # ── 反算初始位置 ──────────────────────────────────────────────────────────
    dt    = 0.05
    v_g   = 0.22
    V_rel = Cp + v_g
    t_meet = thermocline_depth * (6000.0 / 1000.0)
    x_init = (v_g + Cp) * t_meet

    half_window_time = max(4000.0, (8.0 * D) / V_rel)
    start_time = max(0.0, t_meet - half_window_time)
    end_time   = t_meet + half_window_time

    X = v_g * start_time - (x_init - Cp * start_time)
    X = float(np.clip(X, x_grid[0], x_grid[-1]))

    t_mod_start = start_time % 12000.0
    if t_mod_start < 6000.0:
        Z = t_mod_start * 1000.0 / 6000.0
    else:
        Z = 1000.0 - (t_mod_start - 6000.0) * 1000.0 / 6000.0
    Z = float(np.clip(Z, z_grid[0], z_grid[-1]))
    t = start_time

    w_stdy_norm = 1000.0 / 6000.0
    V_norm      = float(np.hypot(v_g, w_stdy_norm))
    zeta_norm   = float(np.degrees(np.arcsin(-w_stdy_norm / V_norm)))
    f_norm      = 0.2

    sampled_data      = []
    time_since_sample = 0.0

    # ── 拉格朗日主循环 ────────────────────────────────────────────────────────
    while t < end_time:
        x_g            = v_g * t
        X_wave_current = x_init - Cp * t
        x_eff          = x_g - X_wave_current

        w_c = get_flow(x_eff, Z)

        if abs(w_c) >= w_c_threshold:
            current_V    = V_target
            current_zeta = zeta_target
            current_fs   = f_s
        else:
            current_V    = V_norm
            current_zeta = zeta_norm
            current_fs   = f_norm

        zeta_rad = np.radians(current_zeta)
        w_g      = -current_V * np.sin(zeta_rad)
        # W_Vel_3D 中 w_c 向上为正，Z 轴向下为正，故减去
        w_abs    = w_g - w_c

        X += v_g * dt
        Z += w_abs * dt
        Z  = float(np.clip(Z, 0.0, 1000.0))
        t += dt

        interval           = 1.0 / current_fs
        time_since_sample += dt
        if time_since_sample >= interval:
            sampled_data.append({'Time': t, 'Z': Z, 'w_c': w_c, 'f_s': current_fs})
            time_since_sample -= interval

    # ── 30% 截断评估 ──────────────────────────────────────────────────────────
    df = pd.DataFrame(sampled_data)
    if df.empty:
        raise RuntimeError("仿真轨迹为空，请检查初始参数或流场数据。")

    # 只找正向峰値（与参考程序一致）
    w_series   = df['w_c']
    idx_max    = w_series.idxmax()
    w_max      = float(w_series.iloc[idx_max])

    if w_max <= 0:
        raise RuntimeError("未检测到上涌正向波办，请检查参数。")

    cutoff_val = 0.30 * w_max

    left_side   = w_series.iloc[:idx_max + 1]
    valid_left  = left_side[left_side > cutoff_val]
    idx_start   = int(valid_left.index[0])  if not valid_left.empty  else int(idx_max)

    right_side  = w_series.iloc[idx_max:]
    valid_right = right_side[right_side > cutoff_val]
    idx_end     = int(valid_right.index[-1]) if not valid_right.empty else int(idx_max)

    df_cut     = df.loc[idx_start:idx_end].copy()
    t_integral = df_cut['Time'].values
    w_integral = df_cut['w_c'].values
    dh_raw     = np.trapezoid(w_integral, x=t_integral)

    z_idx      = np.argmin(np.abs(z_grid - thermocline_depth))
    W_z_meet   = float(W_profile[z_idx])
    doppler_factor = V_rel / Cp
    Delta_Z_calc   = abs(dh_raw * doppler_factor / W_z_meet)
    J              = abs(Delta_Z_calc - Delta_Z_true)

    print(f"[*] {os.path.basename(run_data_dir)} | |w_max|={w_max:.3f}m/s | "
          f"dh_raw={dh_raw:.2f}m·s | doppler={doppler_factor:.3f} | "
          f"W_z_meet={W_z_meet:.4f} | 推算振幅={Delta_Z_calc:.2f}m | "
          f"真实={Delta_Z_true}m | J={J:.4f}")

    return {
        "params":          params,
        "df":              df,
        "idx_max":         idx_max,
        "w_max":           w_max,
        "cutoff_val":      cutoff_val,
        "idx_start":       idx_start,
        "idx_end":         idx_end,
        "t_integral":      t_integral,
        "w_integral":      w_integral,
        "dh_raw":          dh_raw,
        "Delta_Z_calc":    Delta_Z_calc,
        "Delta_Z_true":    Delta_Z_true,
        "J":               J,
        "doppler_factor":  doppler_factor,
        "W_z_meet":        W_z_meet,
        "thermocline_depth": thermocline_depth,
        "w_c_threshold":   w_c_threshold,
        "V_target":        V_target,
        "zeta_target":     zeta_target,
        "f_s":             f_s,
        "V_norm":          V_norm,
        "zeta_norm":       zeta_norm,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_30cut_result(result, group_name):
    df         = result["df"]
    idx_max    = result["idx_max"]
    idx_start  = result["idx_start"]
    idx_end    = result["idx_end"]
    cutoff_val = result["cutoff_val"]

    t_all   = df['Time'].values
    w_all   = df['w_c'].values
    z_all   = df['Z'].values
    fs_all  = df['f_s'].values
    idx_arr = df.index.values

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # ── 主信号 ────────────────────────────────────────────────────────────────
    ax1.plot(t_all, w_all, color="#005b96", linewidth=1.8, label="$w_c$ (sampled)")
    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)

    # 找正向波瓣过零点（与参考程序 base_left/base_right 逻辑一致）
    w_arr = w_all
    pos_max = int(np.where(idx_arr == idx_max)[0][0])
    base_left_pos = pos_max
    while base_left_pos > 0 and w_arr[base_left_pos] > 0:
        base_left_pos -= 1
    base_right_pos = pos_max
    while base_right_pos < len(w_arr) - 1 and w_arr[base_right_pos] > 0:
        base_right_pos += 1
    base_left_idx  = idx_arr[base_left_pos]
    base_right_idx = idx_arr[base_right_pos]

    # 蓝色填充：负速度区域（与参考程序一致）
    ax1.fill_between(
        t_all, 0.0, w_all,
        where=(w_all < 0),
        color="#6b9ac4", alpha=0.45,
        label="Negative velocity area"
    )

    # 30% 阈值线
    ax1.axhline(
        cutoff_val, color="green", linestyle=":", linewidth=1.5, alpha=0.85,
        label=f"30% cut threshold ({cutoff_val:.4f} m/s)"
    )

    # 粉红填充：保留积分区域
    keep_mask = (idx_arr >= idx_start) & (idx_arr <= idx_end)
    ax1.fill_between(
        t_all, 0.0, w_all,
        where=keep_mask,
        color="#f4a1c1", alpha=0.55,
        label="Kept for integration (30% cut)"
    )

    # 绿色填充：正向波瓣内、积分区之外的截去部分（与参考程序 cut_mask 一致）
    cut_mask = (
        (idx_arr >= base_left_idx) & (idx_arr <= base_right_idx)
        & ~keep_mask
        & (w_all > 0)
    )
    ax1.fill_between(
        t_all, 0.0, w_all,
        where=cut_mask,
        color="green", alpha=0.35,
        label="Cut-off area"
    )

    # ── 采样频率切换标记（散点颜色） ─────────────────────────────────────────
    fs_norm_mask   = fs_all == result["V_norm"] / result["V_norm"]  # placeholder
    high_fs_mask   = fs_all > 1.5   # f_s > 1.5 Hz 视为高频模式
    ax1.scatter(
        t_all[high_fs_mask], w_all[high_fs_mask],
        s=6, color="red", alpha=0.5, zorder=4, label=f"High-freq mode (f_s={result['f_s']:.0f} Hz)"
    )
    ax1.scatter(
        t_all[~high_fs_mask], w_all[~high_fs_mask],
        s=4, color="#005b96", alpha=0.25, zorder=3
    )

    # ── 关键点标注 ────────────────────────────────────────────────────────────
    # 峰值
    peak_t = t_all[idx_max]
    peak_w = w_all[idx_max]
    ax1.scatter([peak_t], [peak_w], color="red", s=50, zorder=7)
    _annotate_point(ax1, peak_t, peak_w,
                    f"Peak\n({peak_t:.1f}s, {peak_w:.4f}m/s)",
                    18, 18, color="red")

    # 截断左边界
    c0_t = t_all[idx_start]
    c0_w = w_all[idx_start]
    ax1.scatter([c0_t], [c0_w], color="green", s=36, zorder=7)
    _annotate_point(ax1, c0_t, c0_w,
                    f"Cut L\n({c0_t:.1f}s, {c0_w:.4f})",
                    -95, -42, color="green")

    # 截断右边界
    c1_t = t_all[idx_end]
    c1_w = w_all[idx_end]
    ax1.scatter([c1_t], [c1_w], color="green", s=36, zorder=7)
    _annotate_point(ax1, c1_t, c1_w,
                    f"Cut R\n({c1_t:.1f}s, {c1_w:.4f})",
                    18, -42, color="green")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Vertical Water Velocity $w_c$ (m/s)", color="#005b96")
    ax1.tick_params(axis="y", labelcolor="#005b96")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # ── 深度（副轴） ──────────────────────────────────────────────────────────
    ax2 = ax1.twinx()
    ax2.plot(t_all, z_all, color="darkorange", linewidth=1.5,
             linestyle="--", label="Glider depth")
    ax2.set_ylabel("Depth (m)", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax2.set_ylim(z_all.max() * 1.15, 0)

    # 温跃层深度水平虚线
    ax2.axhline(result["thermocline_depth"], color="darkorange", linestyle="-.",
                linewidth=1.0, alpha=0.6)
    ax2.text(t_all[0], result["thermocline_depth"] - 15,
             f"Thermocline {result['thermocline_depth']:.0f}m",
             color="darkorange", fontsize=8, alpha=0.85)

    # 峰值处深度
    peak_z = z_all[idx_max]
    ax2.scatter([peak_t], [peak_z], color="darkorange", s=28, zorder=6)
    _annotate_point(ax2, peak_t, peak_z,
                    f"Depth@Peak\n({peak_t:.1f}s, {peak_z:.1f}m)",
                    24, -38, color="darkorange")

    # ── 标题与图例 ────────────────────────────────────────────────────────────
    title = (
        f"30%-Cut Lagrangian Sampling | Group: {group_name}\n"
        f"w_thresh={result['w_c_threshold']:.3f}m/s  V_target={result['V_target']:.2f}m/s  "
        f"ζ_target={result['zeta_target']:.1f}°  f_s={result['f_s']:.1f}Hz\n"
        f"Δh_calc={result['Delta_Z_calc']:.2f}m  true h0={result['Delta_Z_true']:.2f}m  "
        f"J={result['J']:.4f}m  (doppler×{result['doppler_factor']:.3f})"
    )
    ax1.set_title(title)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2,
               loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base_dir = r"D:\PYTHON\layout design\V_Wave_Data"
    groups   = list_groups(base_dir)

    if not groups:
        print("未在 V_Wave_Data 下找到任何子文件夹。")
        sys.exit(1)

    print("可用数据组：")
    for i, name in enumerate(groups, 1):
        print(f"  {i}: {name}")

    while True:
        try:
            idx = int(input(f"选择数据组 (1-{len(groups)}): ").strip()) - 1
            if 0 <= idx < len(groups):
                break
            print("序号超出范围，请重新输入。")
        except ValueError:
            print("请输入整数。")

    def _get_float(prompt, lo=None, hi=None):
        while True:
            try:
                v = float(input(prompt).strip())
                if (lo is None or v >= lo) and (hi is None or v <= hi):
                    return v
                print(f"请输入 [{lo}, {hi}] 范围内的值。")
            except ValueError:
                print("请输入数值。")

    w_c_threshold = _get_float("w_c_threshold (m/s, 如 0.05): ", lo=0.0)
    V_target      = _get_float("V_target (m/s, 如 0.45): ",      lo=0.0)
    zeta_target   = _get_float("zeta_target (度, 如 -35.0): ")
    f_s           = _get_float("f_s (Hz, 如 10.0): ",            lo=0.0)

    group_name = groups[idx]
    group_path = os.path.join(base_dir, group_name)

    print(f"\n正在运行仿真: {group_name} ...")
    try:
        result = run_30cut_detailed(w_c_threshold, V_target, zeta_target, f_s, group_path)
        plot_30cut_result(result, group_name)
    except Exception as exc:
        print(f"运行失败: {exc}")
        import traceback
        traceback.print_exc()

"""
批量生成100组30cut Lagrangian采样的图表(初始设定的滑翔机，非优化参数后)
- 相对误差 > 15% 的图放在一张多子图页面上
- 相对误差 <= 15% 的图按每20个分组一页
输出到 Pic/Bayesian_opt 文件夹下
"""
import json
import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELB_DIR = PROJECT_ROOT / "ModelB_Simulated_Sampling_And_Amplitude_Fitting"
MODELE_DIR = PROJECT_ROOT / "ModelE_Bayesian_Opt"
V_WAVE_DATA_DIR = PROJECT_ROOT / "ModelA_Virtual_Internal_Solitary_Wave_Data_Generation" / "V_Wave_Data"
CSV_30CUT = MODELB_DIR / "Analysis_Results_SwA_Lagrangian_Cut_Data" / "analysis_results_swA_lagrangian_30cut.csv"

OUTPUT_DIR = Path("D:/PYTHON/layout design/Pic/Bayesian_opt/Model5_No_Big_Error")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CUT_PCT = 30.0


def run_single_cut(data_dir, cut_pct):
    """Run one Lagrangian sampling and apply dynamic cut threshold."""
    z = np.load(os.path.join(data_dir, "z.npy"))
    x_grid = np.load(os.path.join(data_dir, "x_grid.npy"))
    y_grid = np.load(os.path.join(data_dir, "y_grid.npy"))
    W_Vel_3D = np.load(os.path.join(data_dir, "W_Vel_3D.npy"))
    W_profile = np.load(os.path.join(data_dir, "W_profile.npy"))

    with open(os.path.join(data_dir, "params.json"), "r") as f:
        params = json.load(f)

    Cp = params["c0"]
    thermocline_depth = params["thermocline_depth"]
    true_h0 = params["h0"]
    D = params.get("D", 1000.0)

    if z[0] > z[-1]:
        z = np.flip(z)
        W_Vel_3D = np.flip(W_Vel_3D, axis=2)
        W_profile = np.flip(W_profile)

    interp_w = RegularGridInterpolator(
        (x_grid, y_grid, z), W_Vel_3D, bounds_error=False, fill_value=0.0
    )

    v_g = 0.22
    V_rel = Cp + v_g
    t_meet = thermocline_depth * (6000.0 / 1000.0)
    x_init = v_g * t_meet + Cp * t_meet

    half_window_time = max(4000.0, (8.0 * D) / V_rel)
    start_time = max(0.0, t_meet - half_window_time)
    end_time = t_meet + half_window_time
    dt = 5.0

    t_array = np.arange(start_time, end_time, dt)
    w_isw_array = np.zeros_like(t_array, dtype=float)
    depth_obs = np.zeros_like(t_array, dtype=float)

    t_mod_start = start_time % 12000
    if t_mod_start < 6000:
        z_g = t_mod_start * 1000.0 / 6000.0
    else:
        z_g = 1000.0 - (t_mod_start - 6000.0) * 1000.0 / 6000.0

    for i, t in enumerate(t_array):
        t_mod = t % 12000
        w_stdy = -1000.0 / 6000.0 if t_mod < 6000 else 1000.0 / 6000.0

        x_g = v_g * t
        X_wave_current = x_init - Cp * t
        x_eff = x_g - X_wave_current

        w_isw = interp_w((x_eff, 0.0, z_g))
        w_obs_real = w_stdy + w_isw

        w_isw_array[i] = w_isw
        depth_obs[i] = z_g

        z_g = z_g - w_obs_real * dt
        z_g = np.clip(z_g, 0.0, 1000.0)

    peak_idx = int(np.argmax(w_isw_array))
    w_max = float(w_isw_array[peak_idx])
    threshold_ratio = max(0.0, min(40.0, float(cut_pct))) / 100.0
    w_threshold = threshold_ratio * w_max

    # Baseline interval: no-cut positive lobe (threshold = 0)
    base_left = peak_idx
    while base_left > 0 and w_isw_array[base_left] > 0.0:
        base_left -= 1
    base_right = peak_idx
    while base_right < len(w_isw_array) - 1 and w_isw_array[base_right] > 0.0:
        base_right += 1

    # Kept interval after dynamic cut
    keep_left = peak_idx
    while keep_left > 0 and w_isw_array[keep_left] > w_threshold:
        keep_left -= 1
    keep_right = peak_idx
    while keep_right < len(w_isw_array) - 1 and w_isw_array[keep_right] > w_threshold:
        keep_right += 1

    t_integral = t_array[keep_left:keep_right]
    w_integral = w_isw_array[keep_left:keep_right]
    dh_raw = np.trapezoid(w_integral, x=t_integral)

    z_idx = int(np.argmin(np.abs(z - thermocline_depth)))
    W_z_meet = float(W_profile[z_idx])
    doppler_factor = V_rel / Cp
    h0_corrected = dh_raw * doppler_factor / W_z_meet

    error_abs = abs(h0_corrected - true_h0)
    error_pct = error_abs / true_h0 * 100.0

    upwelling_idx = None
    for i in range(1, len(w_isw_array)):
        if w_isw_array[i-1] <= 0 and w_isw_array[i] > 0:
            upwelling_idx = i
            break

    upwelling_depth = depth_obs[upwelling_idx] if upwelling_idx is not None else None

    return {
        "t_array": t_array,
        "w_isw_array": w_isw_array,
        "depth_obs": depth_obs,
        "peak_idx": peak_idx,
        "w_max": w_max,
        "w_threshold": w_threshold,
        "base_left": base_left,
        "base_right": base_right,
        "keep_left": keep_left,
        "keep_right": keep_right,
        "dh_raw": dh_raw,
        "dh": h0_corrected,
        "true_h0": true_h0,
        "error_pct": error_pct,
        "upwelling_idx": upwelling_idx,
        "upwelling_depth": upwelling_depth,
        "thermocline_depth": thermocline_depth,
    }


def plot_single_cut_on_ax(ax, result, group_name):
    """绘制单个30cut结果到指定的 ax 上（左轴：w_isw，右轴：深度）"""
    t_array = result["t_array"]
    w_isw_array = result["w_isw_array"]
    depth_obs = result["depth_obs"]

    base_left = result["base_left"]
    base_right = result["base_right"]
    keep_left = result["keep_left"]
    keep_right = result["keep_right"]

    idx_array = np.arange(len(t_array))

    # 左轴：垂直流速
    ax.plot(t_array, w_isw_array, color="#005b96", linewidth=1.5, label="w_isw")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)

    # Negative velocity area in blue
    ax.fill_between(
        t_array,
        0.0,
        w_isw_array,
        where=(w_isw_array < 0),
        color="#6b9ac4",
        alpha=0.35,
        label="Negative velocity",
    )

    # Kept integration area
    ax.fill_between(
        t_array,
        0.0,
        w_isw_array,
        where=(idx_array >= keep_left) & (idx_array <= keep_right),
        color="#f4a1c1",
        alpha=0.45,
        label="Kept for integration",
    )

    # Cut-off area
    cut_mask = (
        (idx_array >= base_left)
        & (idx_array <= base_right)
        & ~((idx_array >= keep_left) & (idx_array <= keep_right))
        & (w_isw_array > 0)
    )
    ax.fill_between(
        t_array,
        0.0,
        w_isw_array,
        where=cut_mask,
        color="green",
        alpha=0.25,
        label="Cut-off area",
    )

    # Threshold line
    ax.axhline(
        result["w_threshold"],
        color="green",
        linestyle=":",
        linewidth=1.0,
        alpha=0.7,
    )

    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("w (m/s)", fontsize=9, color="#005b96")
    ax.tick_params(axis='y', labelcolor="#005b96", labelsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(True, linestyle=":", alpha=0.3)

    # 右轴：滑翔机深度
    ax_depth = ax.twinx()
    ax_depth.plot(t_array, depth_obs, color="darkorange", linewidth=1.5, linestyle="--", label="Glider depth")
    ax_depth.set_ylabel("Depth (m)", fontsize=9, color="darkorange")
    ax_depth.tick_params(axis='y', labelcolor="darkorange", labelsize=8)
    ax_depth.set_ylim(1000, 0)  # 深度轴反向（深度向下）

    upwelling_idx = result.get("upwelling_idx")
    upwelling_depth = result.get("upwelling_depth")
    thermocline_depth = result.get("thermocline_depth")
    
    if upwelling_idx is not None and upwelling_depth is not None:
        t_up = result["t_array"][upwelling_idx]
        diff_depth = upwelling_depth - thermocline_depth
        
        ax_depth.scatter([t_up], [upwelling_depth], color='purple', s=30, zorder=7, marker='D')
        
        text = f"Upwelling\nZ={upwelling_depth:.1f}m\nΔZ={diff_depth:+.1f}m"
        ax_depth.annotate(
            text,
            xy=(t_up, upwelling_depth),
            xytext=(10, -25),
            textcoords='offset points',
            fontsize=7,
            color='purple',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='purple', alpha=0.8)
        )

    title = (
        f"{group_name} | Cut 30%\n"
        f"dh={result['dh']:.2f}m, true_h0={result['true_h0']:.2f}m, "
        f"error={result['error_pct']:.2f}%"
    )
    ax.set_title(title, fontsize=9)

    # 合并两个轴的图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_depth.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)


def main():
    print("="*80)
    print("批量生成100组30cut Lagrangian采样图表")
    print("="*80)

    # 读取CSV
    if not CSV_30CUT.exists():
        print(f"❌ CSV文件不存在: {CSV_30CUT}")
        return

    df = pd.read_csv(CSV_30CUT)
    df = df.head(100)  # 取前100组
    print(f"✓ 读取CSV: {len(df)} 组数据\n")

    # 分类：误差 > 15% 和 <= 15%
    high_error_data = []  # (index, wave_id, error_pct)
    low_error_data = []

    for idx, row in df.iterrows():
        wave_id = row['wave_id']
        error_pct = float(row['error_pct'])
        if error_pct > 15.0:
            high_error_data.append((idx, wave_id, error_pct))
        else:
            low_error_data.append((idx, wave_id, error_pct))

    print(f"高误差（>15%）: {len(high_error_data)} 组")
    print(f"低误差（<=15%）: {len(low_error_data)} 组\n")

    # 处理高误差数据：放在多行多列子图中
    if high_error_data:
        print("【生成高误差页面】")
        n_high = len(high_error_data)
        n_cols = 4
        n_rows = (n_high + n_cols - 1) // n_cols
        figsize = (16, 3 * n_rows)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        for subplot_idx, (csv_idx, wave_id, error_pct) in enumerate(high_error_data):
            data_dir = V_WAVE_DATA_DIR / wave_id
            if not data_dir.exists():
                print(f"  ⚠ 数据目录不存在: {wave_id}")
                continue

            try:
                result = run_single_cut(str(data_dir), CUT_PCT)
                plot_single_cut_on_ax(axes[subplot_idx], result, wave_id)
                print(f"  ✓ {wave_id} (error={error_pct:.2f}%)")
            except Exception as e:
                print(f"  ❌ {wave_id} 处理失败: {e}")

        # 隐藏多余的子图
        for idx in range(len(high_error_data), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("相对误差 > 15% 的所有结果", fontsize=14, fontweight='bold')
        plt.tight_layout()
        high_output = OUTPUT_DIR / "01_High_Error_Cases.png"
        plt.savefig(high_output, dpi=150, bbox_inches='tight')
        print(f"  ✓ 已保存: {high_output}\n")
        plt.close()

    # 处理低误差数据：每20个分组一页
    if low_error_data:
        print("【生成低误差页面（每页20个）】")
        page_size = 20
        n_pages = (len(low_error_data) + page_size - 1) // page_size

        for page_num in range(n_pages):
            start_idx = page_num * page_size
            end_idx = min(start_idx + page_size, len(low_error_data))
            page_data = low_error_data[start_idx:end_idx]

            n_items = len(page_data)
            n_cols = 5
            n_rows = (n_items + n_cols - 1) // n_cols
            figsize = (18, 3 * n_rows)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()

            for subplot_idx, (csv_idx, wave_id, error_pct) in enumerate(page_data):
                data_dir = V_WAVE_DATA_DIR / wave_id
                if not data_dir.exists():
                    print(f"  ⚠ 数据目录不存在: {wave_id}")
                    continue

                try:
                    result = run_single_cut(str(data_dir), CUT_PCT)
                    plot_single_cut_on_ax(axes[subplot_idx], result, wave_id)
                    print(f"  ✓ 第{page_num+1}页 - {wave_id} (error={error_pct:.2f}%)")
                except Exception as e:
                    print(f"  ❌ {wave_id} 处理失败: {e}")

            # 隐藏多余的子图
            for idx in range(n_items, len(axes)):
                axes[idx].set_visible(False)

            fig.suptitle(
                f"相对误差 ≤ 15% 的结果 - 第 {page_num+1}/{n_pages} 页（样本 {start_idx+1}-{end_idx}）",
                fontsize=14,
                fontweight='bold'
            )
            plt.tight_layout()
            page_output = OUTPUT_DIR / f"{page_num+2:02d}_Low_Error_Page_{page_num+1:02d}.png"
            plt.savefig(page_output, dpi=150, bbox_inches='tight')
            print(f"  ✓ 已保存: {page_output}\n")
            plt.close()

    print("="*80)
    print(f"✅ 所有操作完成！输出目录: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _annotate_point(ax, x, y, text, dx, dy, color="black", fontsize=8):
    """Annotate a point with readable boxed text and arrow."""
    # Keep labels close to the point for better readability.
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


def list_groups(base_dir="D:\\PYTHON\\layout design\\V_Wave_Data"):
    """List all available wave-data subdirectories."""
    if not os.path.isdir(base_dir):
        return []
    items = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    items.sort()
    return items


def _find_interval_by_threshold(w_array, peak_idx, threshold):
    """Find contiguous interval around the peak where w_array > threshold."""
    left = peak_idx
    while left > 0 and w_array[left] > threshold:
        left -= 1

    right = peak_idx
    while right < len(w_array) - 1 and w_array[right] > threshold:
        right += 1

    return left, right


def run_single_cut(data_dir, cut_pct):
    """Run one Lagrangian sampling and apply dynamic cut threshold.

    Parameters
    ----------
    data_dir : str
        Path to one subdirectory in V_Wave_Data.
    cut_pct : float
        Cut percentage in [0, 40]. 10 means threshold = 10% of peak w.
    """
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
    base_left, base_right = _find_interval_by_threshold(w_isw_array, peak_idx, 0.0)

    # Kept interval after dynamic cut
    keep_left, keep_right = _find_interval_by_threshold(w_isw_array, peak_idx, w_threshold)

    t_integral = t_array[keep_left:keep_right]
    w_integral = w_isw_array[keep_left:keep_right]
    dh_raw = np.trapezoid(w_integral, x=t_integral)

    z_idx = int(np.argmin(np.abs(z - thermocline_depth)))
    W_z_meet = float(W_profile[z_idx])
    doppler_factor = V_rel / Cp
    h0_corrected = dh_raw * doppler_factor / W_z_meet

    error_abs = abs(h0_corrected - true_h0)
    error_pct = error_abs / true_h0 * 100.0

    return {
        "params": params,
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
        "t_integral": t_integral,
        "w_integral": w_integral,
        "dh_raw": dh_raw,
        "dh": h0_corrected,
        "true_h0": true_h0,
        "error_pct": error_pct,
    }


def plot_cut_result(result, cut_pct, group_name):
    """Plot w_isw with cut-off region shaded in green."""
    t_array = result["t_array"]
    w_isw_array = result["w_isw_array"]
    depth_obs = result["depth_obs"]

    base_left = result["base_left"]
    base_right = result["base_right"]
    keep_left = result["keep_left"]
    keep_right = result["keep_right"]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    idx_array = np.arange(len(t_array))

    # Main signal
    ax1.plot(t_array, w_isw_array, color="#005b96", linewidth=2.0, label="w_isw")
    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)

    # Negative velocity area in blue
    ax1.fill_between(
        t_array,
        0.0,
        w_isw_array,
        where=(w_isw_array < 0),
        color="#6b9ac4",
        alpha=0.45,
        label="Negative velocity",
    )

    # Kept integration area
    ax1.fill_between(
        t_array,
        0.0,
        w_isw_array,
        where=(idx_array >= keep_left) & (idx_array <= keep_right),
        color="#f4a1c1",
        alpha=0.55,
        label="Kept for integration",
    )

    # Cut-off area (inside the original positive lobe but outside kept interval)
    cut_mask = (
        (idx_array >= base_left)
        & (idx_array <= base_right)
        & ~(
            (idx_array >= keep_left)
            & (idx_array <= keep_right)
        )
        & (w_isw_array > 0)
    )
    ax1.fill_between(
        t_array,
        0.0,
        w_isw_array,
        where=cut_mask,
        color="green",
        alpha=0.35,
        label="Cut-off area",
    )

    # Threshold line
    ax1.axhline(
        result["w_threshold"],
        color="green",
        linestyle=":",
        linewidth=1.5,
        alpha=0.8,
        label=f"Cut threshold ({cut_pct:.1f}% of peak)",
    )

    # Mark and annotate key vertices/intersections
    peak_idx = result["peak_idx"]
    peak_t = t_array[peak_idx]
    peak_w = w_isw_array[peak_idx]
    ax1.scatter([peak_t], [peak_w], color="red", s=40, zorder=6)
    _annotate_point(
        ax1,
        peak_t,
        peak_w,
        f"Peak\n({peak_t:.1f}s, {peak_w:.3f}m/s)",
        18,
        18,
        color="red",
        fontsize=8,
    )

    p0_t, p0_w = t_array[base_left], w_isw_array[base_left]
    p1_t, p1_w = t_array[base_right], w_isw_array[base_right]
    ax1.scatter([p0_t, p1_t], [p0_w, p1_w], color="black", s=24, zorder=6)
    _annotate_point(ax1, p0_t, p0_w, f"Zero L\n({p0_t:.1f}s, {p0_w:.3f})", -95, 22)
    _annotate_point(ax1, p1_t, p1_w, f"Zero R\n({p1_t:.1f}s, {p1_w:.3f})", 24, 22)

    c0_t, c0_w = t_array[keep_left], w_isw_array[keep_left]
    c1_t, c1_w = t_array[keep_right], w_isw_array[keep_right]
    ax1.scatter([c0_t, c1_t], [c0_w, c1_w], color="green", s=28, zorder=6)
    _annotate_point(ax1, c0_t, c0_w, f"Cut L\n({c0_t:.1f}s, {c0_w:.3f})", -95, -42, color="green")
    _annotate_point(ax1, c1_t, c1_w, f"Cut R\n({c1_t:.1f}s, {c1_w:.3f})", 24, -42, color="green")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Water Vertical Velocity (m/s)", color="#005b96")
    ax1.tick_params(axis="y", labelcolor="#005b96")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Depth on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(t_array, depth_obs, color="darkorange", linewidth=1.5, linestyle="--", label="Glider depth")
    ax2.set_ylabel("Depth (m)", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax2.set_ylim(1000, 0)

    peak_depth = depth_obs[peak_idx]
    ax2.scatter([peak_t], [peak_depth], color="darkorange", s=28, zorder=6)
    _annotate_point(
        ax2,
        peak_t,
        peak_depth,
        f"Depth@Peak\n({peak_t:.1f}s, {peak_depth:.1f}m)",
        24,
        -38,
        color="darkorange",
    )

    title = (
        f"Single W-A Lagrangian Cut Plot | Group: {group_name} | Cut: {cut_pct:.1f}%\n"
        f"dh={result['dh']:.2f} m, true_h0={result['true_h0']:.2f} m, error={result['error_pct']:.2f}%"
    )
    ax1.set_title(title)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    base_dir = "D:\\PYTHON\\layout design\\V_Wave_Data"
    groups = list_groups(base_dir)

    if not groups:
        print("No available group folders found under V_Wave_Data.")
        sys.exit(1)

    print("Available groups:")
    for i, name in enumerate(groups, 1):
        print(f"{i}: {name}")

    while True:
        try:
            idx = int(input(f"Select group (1-{len(groups)}): ").strip()) - 1
            if 0 <= idx < len(groups):
                break
            print("Invalid group index, please try again.")
        except ValueError:
            print("Please input an integer.")

    while True:
        try:
            cut_pct = float(input("Input cut percentage (0-40): ").strip())
            if 0.0 <= cut_pct <= 40.0:
                break
            print("Cut percentage must be in [0, 40].")
        except ValueError:
            print("Please input a numeric value.")

    group_name = groups[idx]
    group_path = os.path.join(base_dir, group_name)

    print(f"Processing group: {group_name}, cut = {cut_pct:.1f}%")
    try:
        result = run_single_cut(group_path, cut_pct)
        plot_cut_result(result, cut_pct, group_name)
    except Exception as exc:
        print(f"Failed to process selected group: {exc}")
        import traceback

        traceback.print_exc()
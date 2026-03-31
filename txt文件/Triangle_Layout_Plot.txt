"""
三滑翔机 + 直线内孤立波的三维布局可视化。

功能:
1. 复用 Basic_Horizonal_Models.py 的部署与采样流程。
2. 对单个波组绘制 4 个关键时刻的 3D 布局图：
   - 最早发车时刻
   - 1 号滑翔机相遇时刻
   - 2 号滑翔机相遇时刻
   - 3 号滑翔机相遇时刻
3. 每个时刻同时显示三台滑翔机当前位置、完整预计轨迹，以及内孤立波热跃层变形面。

说明:
当前采样模型中的内孤立波数据不做旋转，始终按全局 X 方向传播的直线波前处理。
传播偏角 theta 体现在三台滑翔机组成的阵列部署几何上，而不是体现在波场网格旋转上。
因此本脚本固定重建 X 向传播的波面，同时显示被旋转后的三滑翔机阵列与各自 X 向采样轨迹。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Arc

from Basic_Horizonal_Models import ConfigManager, get_glider_config, run_tdoa_group


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "V_Wave_Data_Line"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Pic"

NODE_COLORS = {
    1: "#d1495b",
    2: "#2b59c3",
    3: "#2a9d8f",
}


def _extract_w_series(raw: dict) -> np.ndarray:
    """从 raw 字典中提取垂直流速序列，兼容不同采样脚本的字段名。"""
    for key in ("w_sampled", "w_sampled_array", "w_obs_array", "w_isw_array", "w_obs"):
        if key in raw:
            return np.asarray(raw[key], dtype=float)
    raise KeyError(
        "raw 数据中未找到垂直流速序列。请在采样结果里保存 w_sampled，"
        "或提供 w_isw_array / w_obs_array 等等效字段。"
    )


def _find_peak_cut_zero_points(t: np.ndarray, w: np.ndarray) -> dict:
    """提取波峰、30% cut、左右过零点等特征。"""
    peak_idx = int(np.argmax(w))
    peak_t = float(t[peak_idx])
    peak_w = float(w[peak_idx])
    cut_value = 0.3 * peak_w

    left_cut_candidates = np.where(w[: peak_idx + 1] <= cut_value)[0]
    right_cut_candidates = np.where(w[peak_idx:] <= cut_value)[0]
    left_cut_idx = int(left_cut_candidates[-1]) if left_cut_candidates.size else 0
    right_cut_idx = int(peak_idx + right_cut_candidates[0]) if right_cut_candidates.size else len(w) - 1

    left_zero_candidates = np.where(w[: peak_idx + 1] <= 0.0)[0]
    right_zero_candidates = np.where(w[peak_idx:] <= 0.0)[0]
    left_zero_idx = int(left_zero_candidates[-1]) if left_zero_candidates.size else 0
    right_zero_idx = int(peak_idx + right_zero_candidates[0]) if right_zero_candidates.size else len(w) - 1

    return {
        "peak_idx": peak_idx,
        "peak_t": peak_t,
        "peak_w": peak_w,
        "cut_value": cut_value,
        "left_cut_idx": left_cut_idx,
        "right_cut_idx": right_cut_idx,
        "left_zero_idx": left_zero_idx,
        "right_zero_idx": right_zero_idx,
    }


# ==========================================
# 可视化 1: 采样过程三子图 (带特征点标注)
# ==========================================
def plot_sampling_process(result_dict):
    """
    绘制三架滑翔机的采样过程：包含运动路径、垂直流速、最大值点、30% Cut 点及过零点。
    兼容 raw 中的 w_sampled / w_isw_array / w_obs_array 等字段名。
    """
    node_results = result_dict["node_results"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    fig.suptitle("滑翔机拉格朗日采样过程及 30% Cut 特征点", fontsize=16, fontweight="bold")

    for idx, node_id in enumerate([1, 2, 3]):
        ax_z = axes[idx]

        if node_id not in node_results:
            ax_z.set_visible(False)
            continue

        ax_w = ax_z.twinx()
        raw = node_results[node_id]["raw"]
        t = np.asarray(raw["t_global_array"], dtype=float)
        z = np.asarray(raw["z_track"], dtype=float)
        w = _extract_w_series(raw)
        feature = _find_peak_cut_zero_points(t, w)
        color = NODE_COLORS.get(node_id, "#1f77b4")

        ax_z.plot(t, z, color="black", linestyle="--", linewidth=1.4, label="滑翔机运动轨迹")
        ax_z.set_ylabel(f"{node_id}号深度 (m)", color="black", fontsize=11)
        ax_z.tick_params(axis="y", labelcolor="black")

        ax_w.plot(t, w, color=color, linewidth=2.0, label="观测垂直流速")
        ax_w.set_ylabel("垂直流速 (m/s)", color=color, fontsize=11)
        ax_w.tick_params(axis="y", labelcolor=color)

        peak_idx = feature["peak_idx"]
        left_cut_idx = feature["left_cut_idx"]
        right_cut_idx = feature["right_cut_idx"]
        left_zero_idx = feature["left_zero_idx"]
        right_zero_idx = feature["right_zero_idx"]

        ax_w.scatter(
            [t[peak_idx]],
            [w[peak_idx]],
            color="red",
            marker="o",
            s=48,
            zorder=5,
            label="波峰",
        )
        ax_w.scatter(
            [t[left_cut_idx], t[right_cut_idx]],
            [w[left_cut_idx], w[right_cut_idx]],
            color="magenta",
            marker="^",
            s=50,
            zorder=5,
            label="30% Cut 点",
        )
        ax_w.scatter(
            [t[left_zero_idx], t[right_zero_idx]],
            [w[left_zero_idx], w[right_zero_idx]],
            color="green",
            marker="X",
            s=52,
            zorder=5,
            label="过零点",
        )
        ax_w.axhline(feature["cut_value"], color="magenta", linestyle=":", alpha=0.6)
        ax_w.axhline(0.0, color="green", linestyle="-", alpha=0.25)

        ax_w.annotate(
            f"Peak\n({feature['peak_t']:.1f}s, {feature['peak_w']:.3f})",
            xy=(t[peak_idx], w[peak_idx]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.8),
        )

        ax_z.set_title(f"{node_id}号滑翔机采样过程", fontsize=12, loc="left")
        ax_z.grid(True, alpha=0.28)

        lines_z, labels_z = ax_z.get_legend_handles_labels()
        lines_w, labels_w = ax_w.get_legend_handles_labels()
        merged = dict(zip(labels_z + labels_w, lines_z + lines_w))
        ax_w.legend(merged.values(), merged.keys(), loc="upper right", fontsize=9)

    axes[-1].set_xlabel("绝对时间 T (s)", fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    return fig


# ==========================================
# 可视化 2: 俯视图 (海面位置、相遇位置及传播角度)
# ==========================================
def plot_layout_topview(result_dict, theta_real_deg):
    """
    绘制阵列俯视图，解除 X/Y 轴等比例限制以凸显 X 方向变化。
    相遇位置按垂直流速峰值点近似提取。
    """
    node_results = result_dict["node_results"]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_aspect("auto")

    all_x = []
    all_y = []
    encounter_points = []

    for node_id in [1, 2, 3]:
        if node_id not in node_results:
            continue

        node_result = node_results[node_id]
        raw = node_result["raw"]
        w = _extract_w_series(raw)
        max_idx = int(np.argmax(w))

        x0 = float(node_result["X0"])
        y0 = float(node_result["Y0"])
        x_enc = float(raw["x_track_global"][max_idx])
        y_enc = float(raw["y_track_global"][max_idx])
        encounter_points.append((x_enc, y_enc))

        all_x.extend([x0, x_enc])
        all_y.extend([y0, y_enc])

        ax.plot(x0, y0, "ks", markersize=7, label="水面发车位置" if node_id == 1 else "")
        ax.plot(x_enc, y_enc, "ro", markersize=7, label="水下相遇位置" if node_id == 1 else "")
        ax.annotate(
            "",
            xy=(x_enc, y_enc),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, ls="--"),
        )
        ax.text(x0, y0 + 100.0, f"Node {node_id}\nStart", fontsize=9, ha="center")

    if len(encounter_points) == 3:
        enc_pts = np.asarray(encounter_points, dtype=float)
        triangle = np.vstack((enc_pts, enc_pts[0]))
        ax.plot(triangle[:, 0], triangle[:, 1], "b-", lw=2, alpha=0.55, label="阵型连线")

        tip_x, tip_y = enc_pts[2]
        ax.plot([tip_x, tip_x], [tip_y - 1500, tip_y + 1500], "g-.", lw=2, label="理论平直波前")

        arc = Arc(
            (tip_x, tip_y),
            width=1200,
            height=1200,
            theta1=90 - theta_real_deg,
            theta2=90,
            color="#f77f00",
            lw=1.8,
        )
        ax.add_patch(arc)
        ax.text(tip_x + 180, tip_y + 420, r"$\theta$", color="#f77f00", fontsize=13)

    if all_x and all_y:
        x_margin = max(400.0, (max(all_x) - min(all_x)) * 0.5)
        y_margin = max(300.0, (max(all_y) - min(all_y)) * 0.2)
        ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
        ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)

    ax.set_title(
        f"组网观测俯视图 (传播偏角 $\\theta$ = {theta_real_deg:.1f}°)\n"
        "注意: 已解除等比例限制，X 方向变化被视觉放大",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("全局坐标 X (m)", fontsize=12)
    ax.set_ylabel("全局坐标 Y (m)", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.7)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=10)
    fig.tight_layout()
    return fig


def resolve_data_dir(base_data_dir: str | os.PathLike[str], group_name: str | None = None, group_index: int = 1) -> Path:
    """Resolve a run directory from explicit path, group name, or 1-based index."""
    base_path = Path(base_data_dir)
    if group_name:
        data_dir = base_path / group_name
        if not data_dir.exists():
            raise FileNotFoundError(f"找不到指定数据组: {data_dir}")
        return data_dir

    groups = ConfigManager.list_groups(str(base_path))
    if not groups:
        raise FileNotFoundError(f"数据目录下没有可用波组: {base_path}")

    group_index = max(1, int(group_index))
    if group_index > len(groups):
        raise IndexError(f"group_index={group_index} 超出范围，当前仅有 {len(groups)} 组")
    return base_path / groups[group_index - 1]


def load_wave_context(data_dir: Path) -> dict:
    """Load arrays and parameters needed for wave surface reconstruction."""
    z = np.load(data_dir / "z.npy")
    x_grid = np.load(data_dir / "x_grid.npy")
    y_grid = np.load(data_dir / "y_grid.npy")
    w_profile = np.load(data_dir / "W_profile.npy")
    params = ConfigManager.load_params(str(data_dir / "params.json"))

    if z[0] > z[-1]:
        z = np.flip(z)
        w_profile = np.flip(w_profile)

    return {
        "z": z,
        "x_grid": x_grid,
        "y_grid": y_grid,
        "w_profile": w_profile,
        "params": params,
    }


def interpolate_glider_state(node_result: dict, t_abs: float) -> dict:
    """Interpolate a node state at an arbitrary absolute time."""
    raw = node_result["raw"]
    t_series = np.asarray(raw["t_global_array"], dtype=float)
    x_series = np.asarray(raw["x_track_global"], dtype=float)
    y_series = np.asarray(raw["y_track_global"], dtype=float)
    z_series = np.asarray(raw["z_track"], dtype=float)

    t0 = float(node_result["T0"])
    x0 = float(node_result["X0"])
    y0 = float(node_result["Y0"])

    if t_abs <= t0:
        return {
            "x": x0,
            "y": y0,
            "z": 0.0,
            "status": "pending" if t_abs < t0 else "started",
        }

    if t_abs >= t_series[-1]:
        return {
            "x": float(x_series[-1]),
            "y": float(y_series[-1]),
            "z": float(z_series[-1]),
            "status": "completed",
        }

    return {
        "x": float(np.interp(t_abs, t_series, x_series)),
        "y": float(np.interp(t_abs, t_series, y_series)),
        "z": float(np.interp(t_abs, t_series, z_series)),
        "status": "active",
    }


def get_progress_mask(node_result: dict, t_abs: float) -> np.ndarray:
    """Return the sampled segment already traversed by the glider at t_abs."""
    t_series = np.asarray(node_result["raw"]["t_global_array"], dtype=float)
    return t_series <= t_abs


def compute_wave_center_x(
    node_results: dict,
    params: dict,
    glider_cfg: dict,
    t_abs: float,
    anchor_node: int | None = None,
) -> float:
    """Estimate the instantaneous wave-crest x position under the fixed global-x wave model."""
    cp = float(params["c0"])
    t_meet = float(params["thermocline_depth"]) * (6000.0 / float(glider_cfg["depth_max"]))

    if anchor_node is not None:
        node_result = node_results[anchor_node]
        x0 = float(node_result["X0"])
        t0 = float(node_result["T0"])
        return float(x0 + (float(glider_cfg["v_g"]) + cp) * t_meet - cp * (t_abs - t0))

    centers = []

    for node_result in node_results.values():
        x0 = float(node_result["X0"])
        t0 = float(node_result["T0"])
        centers.append(x0 + (float(glider_cfg["v_g"]) + cp) * t_meet - cp * (t_abs - t0))

    return float(np.mean(centers))


def build_wave_surface(wave_ctx: dict, wave_center_x: float, max_points_x: int = 60, max_points_y: int = 70):
    """Reconstruct the deformed thermocline surface from saved ISW parameters."""
    z = wave_ctx["z"]
    x_grid = wave_ctx["x_grid"]
    y_grid = wave_ctx["y_grid"]
    w_profile = wave_ctx["w_profile"]
    params = wave_ctx["params"]

    x_step = max(1, len(x_grid) // max_points_x)
    y_step = max(1, len(y_grid) // max_points_y)
    x_sample = x_grid[::x_step]
    y_sample = y_grid[::y_step]
    x_mesh, y_mesh = np.meshgrid(x_sample, y_sample, indexing="ij")

    thermocline_depth = float(params["thermocline_depth"])
    h0 = float(params["h0"])
    D = float(params["D"])
    mode_idx = int(np.argmin(np.abs(z - thermocline_depth)))
    mode_scale = abs(float(w_profile[mode_idx]))

    x_effective = x_mesh - wave_center_x
    sech2 = (1.0 / np.cosh(x_effective / D)) ** 2
    z_surface = thermocline_depth + h0 * mode_scale * sech2

    return x_mesh, y_mesh, z_surface


def build_snapshot_times(result: dict) -> list[tuple[float, str, int | None]]:
    """Return the four snapshot moments requested by the user."""
    node_results = result["node_results"]
    t_initial = min(float(node_result["T0"]) for node_result in node_results.values())
    return [
        (t_initial, "Initial deployment", None),
        (float(result["t1_obs"]), "Glider 1 encounter", 1),
        (float(result["t2_obs"]), "Glider 2 encounter", 2),
        (float(result["t3_obs"]), "Glider 3 encounter", 3),
    ]


def compute_axis_limits(snapshot_times: list[tuple[float, str, int | None]], result: dict, wave_ctx: dict, glider_cfg: dict):
    """Use routes and wave positions together to compute stable axes across subplots."""
    node_results = result["node_results"]
    xs = []
    ys = []
    zs = [0.0]

    for node_result in node_results.values():
        xs.extend(np.asarray(node_result["raw"]["x_track_global"], dtype=float).tolist())
        ys.extend(np.asarray(node_result["raw"]["y_track_global"], dtype=float).tolist())
        zs.extend(np.asarray(node_result["raw"]["z_track"], dtype=float).tolist())

    half_span_x = 0.5 * (float(wave_ctx["x_grid"][-1]) - float(wave_ctx["x_grid"][0]))
    for t_abs, _, _ in snapshot_times:
        center_x = compute_wave_center_x(node_results, wave_ctx["params"], glider_cfg, t_abs)
        xs.extend([center_x - half_span_x, center_x + half_span_x])

    x_margin = max(600.0, 0.08 * (max(xs) - min(xs)))
    y_margin = max(600.0, 0.10 * (max(ys) - min(ys)))
    z_max = max(zs) + 80.0

    return {
        "xlim": (min(xs) - x_margin, max(xs) + x_margin),
        "ylim": (min(ys) - y_margin, max(ys) + y_margin),
        "zlim": (z_max, 0.0),
    }


def draw_ocean_surface(ax, axis_limits: dict):
    """Draw the sea surface plane z=0."""
    x_vals = np.linspace(axis_limits["xlim"][0], axis_limits["xlim"][1], 2)
    y_vals = np.linspace(axis_limits["ylim"][0], axis_limits["ylim"][1], 2)
    x_mesh, y_mesh = np.meshgrid(x_vals, y_vals, indexing="ij")
    z_mesh = np.zeros_like(x_mesh)

    ax.plot_surface(
        x_mesh / 1000.0,
        y_mesh / 1000.0,
        z_mesh,
        color="#8ecae6",
        alpha=0.14,
        linewidth=0,
        shade=False,
    )
    ax.text(
        axis_limits["xlim"][1] / 1000.0,
        axis_limits["ylim"][1] / 1000.0,
        0.0,
        "Sea surface",
        color="#0b4f6c",
        fontsize=9,
        ha="right",
    )


def draw_glider_sampling_planes(ax, result: dict, axis_limits: dict):
    """Draw one vertical sampling plane for each glider at fixed y."""
    x_vals = np.linspace(axis_limits["xlim"][0], axis_limits["xlim"][1], 2)
    z_vals = np.linspace(0.0, axis_limits["zlim"][0], 2)
    x_mesh, z_mesh = np.meshgrid(x_vals, z_vals, indexing="ij")

    for node_id, node_result in result["node_results"].items():
        color = NODE_COLORS[node_id]
        y_const = float(node_result["Y0"])
        y_mesh = np.full_like(x_mesh, y_const)

        ax.plot_surface(
            x_mesh / 1000.0,
            y_mesh / 1000.0,
            z_mesh,
            color=color,
            alpha=0.05,
            linewidth=0,
            shade=False,
        )
        ax.text(
            axis_limits["xlim"][0] / 1000.0,
            y_const / 1000.0,
            axis_limits["zlim"][0] * 0.08,
            f"G{node_id} plane",
            color=color,
            fontsize=8,
        )


def plot_snapshot(ax, t_abs: float, title: str, result: dict, wave_ctx: dict, glider_cfg: dict, axis_limits: dict, highlight_node: int | None = None):
    """Draw one 3D snapshot with ISW surface, glider routes, and current positions."""
    node_results = result["node_results"]
    wave_center_x = compute_wave_center_x(
        node_results,
        wave_ctx["params"],
        glider_cfg,
        t_abs,
        anchor_node=highlight_node,
    )
    x_mesh, y_mesh, z_surface = build_wave_surface(wave_ctx, wave_center_x)

    draw_ocean_surface(ax, axis_limits)
    draw_glider_sampling_planes(ax, result, axis_limits)

    ax.plot_surface(
        x_mesh / 1000.0,
        y_mesh / 1000.0,
        z_surface,
        cmap="Blues",
        alpha=0.55,
        linewidth=0,
        antialiased=True,
        shade=True,
    )

    for node_id, node_result in node_results.items():
        raw = node_result["raw"]
        color = NODE_COLORS[node_id]
        x_track = np.asarray(raw["x_track_global"], dtype=float)
        y_track = np.asarray(raw["y_track_global"], dtype=float)
        z_track = np.asarray(raw["z_track"], dtype=float)

        ax.plot(
            x_track / 1000.0,
            y_track / 1000.0,
            z_track,
            color=color,
            alpha=0.18,
            linewidth=1.2,
        )

        progress_mask = get_progress_mask(node_result, t_abs)
        if np.any(progress_mask):
            ax.plot(
                x_track[progress_mask] / 1000.0,
                y_track[progress_mask] / 1000.0,
                z_track[progress_mask],
                color=color,
                alpha=0.9,
                linewidth=2.2,
            )

        state = interpolate_glider_state(node_result, t_abs)
        marker = "^" if state["status"] == "pending" else "o"
        size = 140 if node_id == highlight_node else 90
        edge_width = 1.8 if node_id == highlight_node else 0.8

        ax.scatter(
            [state["x"] / 1000.0],
            [state["y"] / 1000.0],
            [state["z"]],
            color=color,
            marker=marker,
            s=size,
            edgecolors="black",
            linewidths=edge_width,
            zorder=10,
        )
        ax.text(
            state["x"] / 1000.0,
            state["y"] / 1000.0,
            state["z"] + 20.0,
            f"G{node_id}",
            color=color,
            fontsize=10,
            weight="bold",
        )

        ax.scatter(
            [float(node_result["X0"]) / 1000.0],
            [float(node_result["Y0"]) / 1000.0],
            [0.0],
            color=color,
            marker="x",
            s=40,
            alpha=0.8,
        )

    x_mid = 0.5 * (axis_limits["xlim"][0] + axis_limits["xlim"][1])
    y_mid = 0.5 * (axis_limits["ylim"][0] + axis_limits["ylim"][1])
    z_annot = float(wave_ctx["params"]["thermocline_depth"]) + 35.0
    ax.quiver(
        x_mid / 1000.0,
        y_mid / 1000.0,
        z_annot,
        -0.8,
        0.0,
        0.0,
        color="#1d3557",
        linewidth=2.0,
        arrow_length_ratio=0.25,
    )
    ax.text(x_mid / 1000.0 - 0.9, y_mid / 1000.0, z_annot - 20.0, "ISW propagation (-X)", color="#1d3557", fontsize=9)

    ax.set_title(f"{title}\n t = {t_abs:.1f} s", fontsize=12, pad=10)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Depth (m)")
    ax.set_xlim(axis_limits["xlim"][0] / 1000.0, axis_limits["xlim"][1] / 1000.0)
    ax.set_ylim(axis_limits["ylim"][0] / 1000.0, axis_limits["ylim"][1] / 1000.0)
    ax.set_zlim(*axis_limits["zlim"])
    ax.view_init(elev=26, azim=-58)
    ax.grid(True, alpha=0.25)


def plot_top_view(
    fig,
    result: dict,
    wave_ctx: dict,
    glider_cfg: dict,
    snapshot_times: list[tuple[float, str, int | None]],
    axis_limits: dict,
    L_spacing: float,
    H_spacing: float,
    theta_real_deg: float,
):
    """Draw one 2D top-view figure with four subplots matching the 3D snapshots."""
    D = float(wave_ctx["params"]["D"])
    wave_y_min = float(wave_ctx["y_grid"][0]) / 1000.0
    wave_y_max = float(wave_ctx["y_grid"][-1]) / 1000.0
    node_results = result["node_results"]
    encounter_points = {}
    for node_id, node_result in node_results.items():
        ref_time = float(result[f"t{node_id}_ref"])
        encounter_points[node_id] = interpolate_glider_state(node_result, ref_time)

    base_mid_x = 0.5 * (encounter_points[1]["x"] + encounter_points[2]["x"]) / 1000.0
    base_mid_y = 0.5 * (encounter_points[1]["y"] + encounter_points[2]["y"]) / 1000.0
    apex_x = encounter_points[3]["x"] / 1000.0
    apex_y = encounter_points[3]["y"] / 1000.0

    for idx, (t_abs, title, highlight_node) in enumerate(snapshot_times, start=1):
        ax = fig.add_subplot(2, 2, idx)
        wave_center_x = compute_wave_center_x(
            node_results,
            wave_ctx["params"],
            glider_cfg,
            t_abs,
            anchor_node=highlight_node,
        )
        wave_left = (wave_center_x - D) / 1000.0
        wave_right = (wave_center_x + D) / 1000.0

        x_wave = np.linspace(axis_limits["xlim"][0] / 1000.0, axis_limits["xlim"][1] / 1000.0, 240)
        y_wave = np.linspace(wave_y_min, wave_y_max, 120)
        x_mesh, y_mesh = np.meshgrid(x_wave, y_wave)
        x_relative_m = (x_mesh - wave_center_x / 1000.0) * 1000.0
        wave_strength = (1.0 / np.cosh(x_relative_m / D)) ** 2
        ax.contourf(
            x_mesh,
            y_mesh,
            wave_strength,
            levels=np.linspace(0.08, 1.0, 9),
            cmap="Blues",
            alpha=0.42,
            zorder=0,
        )
        ax.plot(
            [wave_center_x / 1000.0, wave_center_x / 1000.0],
            [wave_y_min, wave_y_max],
            color="#1565c0",
            linestyle="--",
            linewidth=1.6,
            label="ISW crest",
        )
        ax.plot(
            [axis_limits["xlim"][0] / 1000.0, axis_limits["xlim"][1] / 1000.0],
            [wave_y_min, wave_y_min],
            color="#4ea8de",
            linestyle=":",
            linewidth=0.9,
            alpha=0.8,
        )
        ax.plot(
            [axis_limits["xlim"][0] / 1000.0, axis_limits["xlim"][1] / 1000.0],
            [wave_y_max, wave_y_max],
            color="#4ea8de",
            linestyle=":",
            linewidth=0.9,
            alpha=0.8,
        )
        ax.annotate(
            "ISW propagation",
            xy=((wave_center_x / 1000.0) - 0.8, axis_limits["ylim"][1] / 1000.0 - 0.2),
            xytext=((wave_center_x / 1000.0) + 1.0, axis_limits["ylim"][1] / 1000.0 - 0.2),
            arrowprops=dict(arrowstyle="->", color="#1565c0", lw=1.5),
            color="#1565c0",
            fontsize=9,
            ha="left",
            va="center",
        )
        ax.annotate(
            "ISW span in y",
            xy=(wave_center_x / 1000.0 + 0.2, wave_y_min),
            xytext=(wave_center_x / 1000.0 + 0.2, wave_y_max),
            arrowprops=dict(arrowstyle="<->", color="#4ea8de", lw=1.2),
            color="#4ea8de",
            fontsize=8,
            ha="left",
            va="center",
            rotation=90,
        )

        encounter_points_x = []
        encounter_points_y = []
        for node_id, encounter_state in encounter_points.items():
            encounter_points_x.append(encounter_state["x"] / 1000.0)
            encounter_points_y.append(encounter_state["y"] / 1000.0)

        ax.plot(
            encounter_points_x + [encounter_points_x[0]],
            encounter_points_y + [encounter_points_y[0]],
            color="#6c757d",
            linestyle=":",
            linewidth=1.1,
            alpha=0.9,
            label="Encounter array",
        )
        ax.plot(
            [base_mid_x, apex_x],
            [base_mid_y, apex_y],
            color="#495057",
            linestyle="-.",
            linewidth=1.3,
            alpha=0.9,
        )

        for node_id, node_result in node_results.items():
            color = NODE_COLORS[node_id]
            x_track = np.asarray(node_result["raw"]["x_track_global"], dtype=float)
            y_track = np.asarray(node_result["raw"]["y_track_global"], dtype=float)
            progress_mask = get_progress_mask(node_result, t_abs)
            state = interpolate_glider_state(node_result, t_abs)

            ax.plot(x_track / 1000.0, y_track / 1000.0, color=color, alpha=0.18, linewidth=1.0)
            if np.any(progress_mask):
                ax.plot(
                    x_track[progress_mask] / 1000.0,
                    y_track[progress_mask] / 1000.0,
                    color=color,
                    alpha=0.95,
                    linewidth=2.0,
                )

            size = 80 if node_id == highlight_node else 50
            ax.scatter(
                [state["x"] / 1000.0],
                [state["y"] / 1000.0],
                color=color,
                marker="o",
                s=size,
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )
            ax.text(state["x"] / 1000.0, state["y"] / 1000.0, f" G{node_id}", color=color, fontsize=9)
            ax.scatter(
                [float(node_result["X0"]) / 1000.0],
                [float(node_result["Y0"]) / 1000.0],
                color=color,
                marker="x",
                s=24,
                alpha=0.8,
            )

            ref_state = encounter_points[node_id]
            ax.scatter(
                [ref_state["x"] / 1000.0],
                [ref_state["y"] / 1000.0],
                color=color,
                marker="s",
                s=28,
                alpha=0.9,
            )

            if node_id == highlight_node:
                ax.scatter(
                    [state["x"] / 1000.0],
                    [state["y"] / 1000.0],
                    color="#ffd166",
                    marker="o",
                    s=170,
                    alpha=0.22,
                    linewidths=0,
                    zorder=4,
                )

        ax.set_title(f"{title}\n t = {t_abs:.1f} s", fontsize=11)
        ax.set_xlim(axis_limits["xlim"][0] / 1000.0, axis_limits["xlim"][1] / 1000.0)
        ax.set_ylim(axis_limits["ylim"][0] / 1000.0, axis_limits["ylim"][1] / 1000.0)
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")



def build_top_view_legend_handles():
    """Build a compact Chinese legend for gliders and marker meanings."""
    return [
        Line2D([0], [0], color=NODE_COLORS[1], linewidth=2.2, label="1号滑翔机"),
        Line2D([0], [0], color=NODE_COLORS[2], linewidth=2.2, label="2号滑翔机"),
        Line2D([0], [0], color=NODE_COLORS[3], linewidth=2.2, label="3号滑翔机"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="black", linestyle="None", label="当前位置"),
        Line2D([0], [0], marker="s", color="black", markerfacecolor="black", linestyle="None", label="相遇位置"),
        Line2D([0], [0], marker="x", color="black", linestyle="None", label="水面位置"),
    ]


# def create_3d_snapshot_figures(
#     result: dict,
#     wave_ctx: dict,
#     glider_cfg: dict,
#     snapshot_times: list[tuple[float, str, int | None]],
#     axis_limits: dict,
#     L_spacing: float,
#     H_spacing: float,
#     theta_real_deg: float,
# ):
#     """Temporarily disabled while focusing on 2D top-view optimization."""
#     figures = []
#     group_name = wave_ctx.get("group_name", "unknown")
#
#     for t_abs, title, highlight_node in snapshot_times:
#         fig = plt.figure(figsize=(10, 8))
#         ax = fig.add_subplot(1, 1, 1, projection="3d")
#         plot_snapshot(
#             ax=ax,
#             t_abs=t_abs,
#             title=title,
#             result=result,
#             wave_ctx=wave_ctx,
#             glider_cfg=glider_cfg,
#             axis_limits=axis_limits,
#             highlight_node=highlight_node,
#         )
#         fig.suptitle(
#             f"{group_name} | L={L_spacing:.0f} m, H={H_spacing:.0f} m, theta={theta_real_deg:.1f} deg",
#             fontsize=12,
#             y=0.98,
#         )
#         fig.tight_layout()
#         figures.append(fig)
#
#     return figures


def save_interactive_figures(figures: list, top_view_fig, output_dir: Path, group_name: str):
    """Save the top view and, when enabled again, optional 3D snapshots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    top_view_path = output_dir / f"triangle_layout_topview_{group_name}.png"
    top_view_fig.savefig(top_view_path, dpi=220, bbox_inches="tight")

    snapshot_labels = ["initial", "g1", "g2", "g3"]
    for fig, suffix in zip(figures, snapshot_labels):
        fig.savefig(output_dir / f"triangle_layout_{suffix}_{group_name}.png", dpi=220, bbox_inches="tight")

    print(f"已保存交互布局图到目录: {output_dir}")


def save_analysis_figures(sampling_fig, layout_fig, output_dir: Path, group_name: str):
    """Save the sampling-process figure and the single top-view figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sampling_path = output_dir / f"sampling_process_{group_name}.png"
    layout_path = output_dir / f"layout_topview_{group_name}.png"
    sampling_fig.savefig(sampling_path, dpi=220, bbox_inches="tight")
    layout_fig.savefig(layout_path, dpi=220, bbox_inches="tight")
    print(f"已保存采样过程图: {sampling_path}")
    print(f"已保存俯视图: {layout_path}")


def generate_layout_views(
    data_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] | None = None,
    L_spacing: float = 4000.0,
    H_spacing: float = 2000.0,
    theta_real_deg: float = 15.0,
    t0_ref: float = 10000.0,
    cut_percentage: float = 30.0,
    glider_cfg: dict | None = None,
    show: bool = True,
    save: bool = False,
):
    """Generate one 2D top-view window and four independent interactive 3D windows."""
    data_dir = Path(data_dir)
    glider_cfg = dict(get_glider_config() if glider_cfg is None else glider_cfg)
    wave_ctx = load_wave_context(data_dir)
    wave_ctx["group_name"] = data_dir.name

    result = run_tdoa_group(
        data_dir=str(data_dir),
        L_spacing=L_spacing,
        H_spacing=H_spacing,
        glider_cfg=glider_cfg,
        theta_real_deg=theta_real_deg,
        t0_ref=t0_ref,
        cut_percentage=cut_percentage,
    )

    snapshot_times = build_snapshot_times(result)
    axis_limits = compute_axis_limits(snapshot_times, result, wave_ctx, glider_cfg)

    top_view_fig = plt.figure(figsize=(15, 11))
    plot_top_view(
        top_view_fig,
        result,
        wave_ctx,
        glider_cfg,
        snapshot_times,
        axis_limits,
        L_spacing,
        H_spacing,
        theta_real_deg,
    )
    top_view_fig.suptitle(
        f"Top view of three gliders and the ISW | {data_dir.name}\n"
        f"L={L_spacing:.0f} m, H={H_spacing:.0f} m, array rotation theta={theta_real_deg:.1f} deg",
        fontsize=14,
        y=0.98,
    )
    top_view_fig.legend(
        handles=build_top_view_legend_handles(),
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=9,
        frameon=True,
        borderaxespad=0.8,
    )
    top_view_fig.tight_layout(rect=(0.0, 0.02, 0.84, 0.95))

    # 3D views are temporarily disabled so that we can focus on improving the 2D top view.
    snapshot_figures = []

    if save:
        save_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
        save_interactive_figures(snapshot_figures, top_view_fig, save_dir, data_dir.name)

    if show:
        plt.show()
    else:
        plt.close(top_view_fig)
        for fig in snapshot_figures:
            plt.close(fig)

    return {
        "top_view_figure": top_view_fig,
        "snapshot_figures": snapshot_figures,
        "result": result,
    }


def generate_cli_plots(
    data_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] | None = None,
    L_spacing: float = 4000.0,
    H_spacing: float = 2000.0,
    theta_real_deg: float = 15.0,
    t0_ref: float = 10000.0,
    cut_percentage: float = 30.0,
    glider_cfg: dict | None = None,
    show: bool = True,
    save: bool = False,
):
    """命令行主入口：直接生成采样过程图和单张俯视图。"""
    data_dir = Path(data_dir)
    glider_cfg = dict(get_glider_config() if glider_cfg is None else glider_cfg)

    result = run_tdoa_group(
        data_dir=str(data_dir),
        L_spacing=L_spacing,
        H_spacing=H_spacing,
        glider_cfg=glider_cfg,
        theta_real_deg=theta_real_deg,
        t0_ref=t0_ref,
        cut_percentage=cut_percentage,
    )

    sampling_fig = plot_sampling_process(result)
    layout_fig = plot_layout_topview(result, theta_real_deg)

    if save:
        save_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
        save_analysis_figures(sampling_fig, layout_fig, save_dir, data_dir.name)

    if show:
        plt.show()
    else:
        plt.close(sampling_fig)
        plt.close(layout_fig)

    return {
        "sampling_figure": sampling_fig,
        "layout_figure": layout_fig,
        "result": result,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="命令行一键弹出采样过程图和阵列俯视图")
    parser.add_argument("--base-data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="波组根目录")
    parser.add_argument("--group-name", type=str, default=None, help="指定波组目录名，例如 run_20260325_142734")
    parser.add_argument("--group-index", type=int, default=1, help="若未指定 group-name，则使用 1-based 组号")
    parser.add_argument("--output-dir", type=str, default=None, help="若启用保存，则输出目录")
    parser.add_argument("--L-spacing", type=float, default=4000.0, help="三角形底边长度 (m)")
    parser.add_argument("--H-spacing", type=float, default=2000.0, help="三角形高度 (m)")
    parser.add_argument("--theta", type=float, default=15.0, help="传播偏角 (deg)")
    parser.add_argument("--t0-ref", type=float, default=10000.0, help="理论参考时刻 (s)")
    parser.add_argument("--cut", type=float, default=30.0, help="30cut 阈值百分比")
    parser.add_argument("--save", action="store_true", help="是否额外保存采样过程图和俯视图")
    parser.add_argument("--no-show", action="store_true", help="只生成对象或保存，不弹出图窗")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    data_dir = resolve_data_dir(
        base_data_dir=args.base_data_dir,
        group_name=args.group_name,
        group_index=args.group_index,
    )
    generate_cli_plots(
        data_dir=data_dir,
        output_dir=args.output_dir,
        L_spacing=args.L_spacing,
        H_spacing=args.H_spacing,
        theta_real_deg=args.theta,
        t0_ref=args.t0_ref,
        cut_percentage=args.cut,
        show=not args.no_show,
        save=args.save,
    )


if __name__ == "__main__":
    main()
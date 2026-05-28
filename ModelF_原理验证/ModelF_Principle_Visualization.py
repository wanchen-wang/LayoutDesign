"""Create the ModelF principle sketch.

Builds a static figure and GIF explaining the ideal sampling principle:
the glider stays horizontally fixed near the strongest displaced layer, the
wave passes over it, and the positive vertical-velocity lobe is integrated.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


OUT_DIR = Path(__file__).resolve().parent / "Results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sech2(x):
    return 1.0 / np.cosh(x) ** 2


def build_ideal_fields():
    depth_max = 1000.0
    h0 = 120.0
    d_scale = 950.0
    c = 2.2

    z = np.linspace(0.0, depth_max, 500)
    x = np.linspace(-4500.0, 4500.0, 700)
    w_profile = np.sin(np.pi * z / depth_max)
    z_star = float(z[np.argmax(np.abs(w_profile))])
    w_star = float(np.interp(z_star, z, w_profile))

    t = np.linspace(-2800.0, 2800.0, 900)
    s_t = -c * t / d_scale
    displacement_t = h0 * sech2(s_t) * w_star
    w_t = -(2.0 * h0 * c / d_scale) * sech2(s_t) * np.tanh(s_t) * w_star

    return {
        "depth_max": depth_max,
        "h0": h0,
        "d_scale": d_scale,
        "c": c,
        "z": z,
        "x": x,
        "w_profile": w_profile,
        "z_star": z_star,
        "w_star": w_star,
        "t": t,
        "displacement_t": displacement_t,
        "w_t": w_t,
    }


def make_static_figure(data):
    z = data["z"]
    x = data["x"]
    w_profile = data["w_profile"]
    z_star = data["z_star"]
    h0 = data["h0"]
    d_scale = data["d_scale"]
    t = data["t"]
    w_t = data["w_t"]
    displacement_t = data["displacement_t"]

    eta_center = h0 * sech2(x / d_scale)
    tw0_idx = int(np.argmin(np.abs(t)))
    pos_end_candidates = np.where((t > 0) & (w_t > 0.02 * np.max(w_t)))[0]
    t_end_idx = int(pos_end_candidates[-1]) if len(pos_end_candidates) else len(t) - 1
    dh_raw = np.trapezoid(w_t[tw0_idx:t_end_idx], x=t[tw0_idx:t_end_idx])

    fig = plt.figure(figsize=(13.5, 8.2), dpi=160)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], wspace=0.26, hspace=0.34)

    ax_mode = fig.add_subplot(gs[0, 0])
    ax_mode.plot(w_profile, z, color="#234f9f", lw=2.5)
    ax_mode.axhline(z_star, color="#d14b36", lw=1.8, ls="--")
    ax_mode.scatter([np.interp(z_star, z, w_profile)], [z_star], s=60, color="#d14b36", zorder=5)
    ax_mode.set_ylim(data["depth_max"], 0)
    ax_mode.set_xlabel("Layer response W(z)")
    ax_mode.set_ylabel("Depth (m)")
    ax_mode.set_title("Choose the strongest displaced layer")
    ax_mode.grid(alpha=0.22)

    ax_wave = fig.add_subplot(gs[0, 1])
    ax_wave.plot(x, z_star + eta_center, color="#246b5a", lw=2.5)
    ax_wave.axhline(z_star, color="#777777", lw=1.2, ls=":")
    ax_wave.axvline(0, color="#d14b36", lw=1.8, ls="--")
    ax_wave.scatter([0], [z_star + h0], s=70, color="#d14b36", zorder=5)
    ax_wave.set_ylim(z_star + 175, z_star - 35)
    ax_wave.set_xlim(x.min(), x.max())
    ax_wave.set_xlabel("Distance from wave center (m)")
    ax_wave.set_ylabel("Depth of selected layer (m)")
    ax_wave.set_title("Wave trough passes the fixed station")
    ax_wave.grid(alpha=0.22)

    ax_w = fig.add_subplot(gs[1, 0])
    ax_w.plot(t, w_t, color="#193b7a", lw=2.2)
    ax_w.axhline(0, color="#444444", lw=1)
    ax_w.axvline(0, color="#d14b36", lw=1.8, ls="--")
    ax_w.fill_between(
        t[tw0_idx:t_end_idx],
        0,
        w_t[tw0_idx:t_end_idx],
        color="#f09abb",
        alpha=0.58,
        label=f"integral after zero-crossing = {dh_raw:.1f} m",
    )
    ax_w.set_xlabel("Time relative to trough center (s)")
    ax_w.set_ylabel("Vertical water velocity w (m/s)")
    ax_w.set_title("Best estimate uses the clean positive lobe")
    ax_w.legend(loc="upper right", fontsize=8, frameon=False)
    ax_w.grid(alpha=0.22)

    ax_depth = fig.add_subplot(gs[1, 1])
    depth_t = z_star + displacement_t
    ax_depth.plot(t, depth_t, color="#6b4c9a", lw=2.2)
    ax_depth.axvline(0, color="#d14b36", lw=1.8, ls="--")
    ax_depth.scatter([0], [z_star + h0], s=70, color="#d14b36", zorder=5)
    ax_depth.set_ylim(z_star + h0 + 45, z_star - 20)
    ax_depth.set_xlabel("Time relative to trough center (s)")
    ax_depth.set_ylabel("Layer / ideal glider depth (m)")
    ax_depth.set_title("Horizontal fixed, vertical follows the wave")
    ax_depth.grid(alpha=0.22)

    fig.suptitle("ModelF Principle Check", fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(OUT_DIR / "ModelF_Principle_Visualization.png", bbox_inches="tight")
    plt.close(fig)


def make_animation(data):
    x = data["x"]
    z_star = data["z_star"]
    h0 = data["h0"]
    d_scale = data["d_scale"]
    c = data["c"]

    frames_t = np.linspace(-1900, 1900, 90)
    fig, (ax_scene, ax_series) = plt.subplots(
        2,
        1,
        figsize=(9.2, 7.2),
        dpi=130,
        gridspec_kw={"height_ratios": [1.25, 1.0]},
    )

    ax_scene.set_xlim(-3800, 3800)
    ax_scene.set_ylim(z_star + 175, z_star - 55)
    ax_scene.set_xlabel("Horizontal distance from fixed station (m)")
    ax_scene.set_ylabel("Depth (m)")
    ax_scene.set_title("The wave moves; the ideal glider keeps x fixed")
    ax_scene.grid(alpha=0.22)
    ax_scene.axvline(0, color="#d14b36", lw=1.6, ls="--")
    ax_scene.axhline(z_star, color="#777777", lw=1.0, ls=":")
    wave_line, = ax_scene.plot([], [], color="#246b5a", lw=2.6)
    glider_dot = ax_scene.scatter([], [], s=95, color="#d14b36", zorder=6)
    scene_text = ax_scene.text(-3650, z_star + 150, "", fontsize=9, color="#333333")

    t_full = data["t"]
    w_full = data["w_t"]
    ax_series.set_xlim(frames_t.min(), frames_t.max())
    ax_series.set_ylim(w_full.min() * 1.16, w_full.max() * 1.16)
    ax_series.set_xlabel("Time relative to trough center (s)")
    ax_series.set_ylabel("w at fixed station (m/s)")
    ax_series.grid(alpha=0.22)
    ax_series.axhline(0, color="#444444", lw=1)
    ax_series.axvline(0, color="#d14b36", lw=1.4, ls="--")
    ax_series.plot(t_full, w_full, color="#b7c6e8", lw=1.5)
    current_marker = ax_series.scatter([], [], s=55, color="#193b7a", zorder=6)

    def update(frame_t):
        center = c * frame_t
        eta = h0 * sech2((x - center) / d_scale)
        wave_line.set_data(x, z_star + eta)
        eta_station = h0 * sech2((0.0 - center) / d_scale)
        glider_dot.set_offsets([[0.0, z_star + eta_station]])
        s = -c * frame_t / d_scale
        w_now = -(2.0 * h0 * c / d_scale) * sech2(s) * np.tanh(s)
        current_marker.set_offsets([[frame_t, w_now]])
        scene_text.set_text(f"t = {frame_t:5.0f} s\nw = {w_now: .3f} m/s")
        return wave_line, glider_dot, current_marker, scene_text

    anim = FuncAnimation(fig, update, frames=frames_t, interval=80, blit=True)
    anim.save(OUT_DIR / "ModelF_Principle_Visualization.gif", writer=PillowWriter(fps=12))
    plt.close(fig)


if __name__ == "__main__":
    fields = build_ideal_fields()
    make_static_figure(fields)
    make_animation(fields)
    print(OUT_DIR / "ModelF_Principle_Visualization.png")
    print(OUT_DIR / "ModelF_Principle_Visualization.gif")

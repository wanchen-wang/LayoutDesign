"""Create ModelF sampling trajectory GIFs.

Uses generated ModelF ideal wave data and one of the A/B1/B2/B3 strategies to
render the glider trajectory together with the sampled vertical-water-speed
time series and integration lobe.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from ModelF_Sampling_Strategies import (
    DEFAULT_DATA_DIR,
    RESULTS_DIR,
    STRATEGIES,
    load_wave_case,
    simulate_strategy,
)


def sech2(value):
    return 1.0 / np.cosh(value) ** 2


def wave_line(case, x_line, center):
    h0 = case["h0"]
    d_scale = case["d_scale"]
    z_star = case["z_star"]
    w_star = abs(case["w_star"])
    eta = h0 * w_star * sech2((x_line - center) / d_scale)
    return z_star + eta


def make_animation(strategy_code: str, wave_index: int = 0):
    run_dirs = sorted([p for p in DEFAULT_DATA_DIR.iterdir() if p.is_dir()])
    case = load_wave_case(run_dirs[wave_index])
    strategy = STRATEGIES[strategy_code]
    res = simulate_strategy(case, strategy, return_track=True)
    track = res["track"]
    start, end = res["integral_slice"]

    frame_indices = np.linspace(0, len(track) - 1, 120).astype(int)
    x_line = np.linspace(-4200, 4200, 500)
    z_star = case["z_star"]
    h0 = case["h0"]
    z_pad = max(80.0, 0.65 * h0)

    fig, (ax_scene, ax_w) = plt.subplots(
        2,
        1,
        figsize=(9.5, 7.8),
        dpi=130,
        gridspec_kw={"height_ratios": [1.25, 1.0]},
    )

    ax_scene.set_xlim(x_line.min(), x_line.max())
    ax_scene.set_ylim(z_star + h0 + z_pad, z_star - z_pad)
    ax_scene.set_xlabel("Distance around glider station (m)")
    ax_scene.set_ylabel("Depth (m)")
    ax_scene.set_title(f"{strategy.label} | wave {case['wave_id']}")
    ax_scene.grid(alpha=0.22)
    ax_scene.axvline(0, color="#d14b36", lw=1.4, ls="--")
    ax_scene.axhline(z_star, color="#777777", lw=1.0, ls=":")
    wave_artist, = ax_scene.plot([], [], color="#246b5a", lw=2.5)
    glider_artist = ax_scene.scatter([], [], s=90, color="#d14b36", zorder=6)
    path_artist, = ax_scene.plot([], [], color="#d14b36", lw=1.2, alpha=0.65)
    info_text = ax_scene.text(x_line.min() + 120, z_star + h0 + z_pad * 0.62, "", fontsize=9)

    ax_w.plot(track["Time"], track["W"], color="#b7c6e8", lw=1.5)
    if end > start:
        ax_w.fill_between(
            track["Time"].iloc[start:end],
            0,
            track["W"].iloc[start:end],
            color="#f09abb",
            alpha=0.58,
            label="integrated lobe",
        )
    ax_w.axhline(0, color="#444444", lw=1)
    ax_w.axvline(0, color="#d14b36", lw=1.2, ls="--")
    ax_w.set_xlabel("Time relative to t0 (s)")
    ax_w.set_ylabel("Sampled vertical water speed (m/s)")
    ax_w.set_title(
        f"calc={res['calc_h0']:.1f} m, true={res['true_h0']:.1f} m, error={res['rel_error']:.2f}%"
    )
    ax_w.grid(alpha=0.22)
    ax_w.legend(frameon=False, loc="upper right")
    marker = ax_w.scatter([], [], s=55, color="#193b7a", zorder=6)

    def update(idx):
        row = track.iloc[idx]
        time_now = float(row["Time"])
        center = -case["cp"] * time_now
        wave_artist.set_data(x_line, wave_line(case, x_line, center))
        glider_artist.set_offsets([[0.0, float(row["Z"])]])
        path_artist.set_data(np.zeros(idx + 1), track["Z"].iloc[: idx + 1])
        marker.set_offsets([[time_now, float(row["W"])]])
        info_text.set_text(
            f"t = {time_now:6.0f} s\n"
            f"z = {float(row['Z']):6.1f} m\n"
            f"w = {float(row['W']): .3f} m/s"
        )
        return wave_artist, glider_artist, path_artist, marker, info_text

    anim = FuncAnimation(fig, update, frames=frame_indices, interval=75, blit=True)
    out_path = RESULTS_DIR / f"ModelF_{strategy_code}_Sampling.gif"
    anim.save(out_path, writer=PillowWriter(fps=12))
    plt.close(fig)
    print(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default="A")
    parser.add_argument("--wave-index", type=int, default=0)
    args = parser.parse_args()
    make_animation(args.strategy, args.wave_index)


if __name__ == "__main__":
    main()

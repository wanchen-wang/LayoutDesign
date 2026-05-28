"""Visualize ModelF generated internal-wave fields.

This script mirrors the ModelA plotting workflow, but writes figures to the
ModelF Results directory instead of opening interactive windows.  It can
regenerate one ModelF case with a 3D y-grid, then plot:

- background stratification,
- vertical mode structure,
- temperature x-y and x-z slices,
- vertical and horizontal velocity x-z sections,
- a 3D vertical-velocity surface when more than one y level is available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm

from ModelF_Ideal_Wave_Generate import OUT_DIR, generate_batch


RESULTS_DIR = Path(__file__).resolve().parent / "Results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


REQUIRED_FILES = [
    "z.npy",
    "x_grid.npy",
    "y_grid.npy",
    "T_profile.npy",
    "rho_profile.npy",
    "N2_profile.npy",
    "W_profile.npy",
    "U_profile.npy",
    "T_3D.npy",
    "W_Vel_3D.npy",
    "U_Vel_3D.npy",
    "params.json",
]


def sorted_run_dirs(data_dir: Path = OUT_DIR) -> list[Path]:
    return sorted([p for p in data_dir.iterdir() if p.is_dir()])


def latest_complete_case(data_dir: Path = OUT_DIR) -> Path | None:
    for run_dir in reversed(sorted_run_dirs(data_dir)):
        if all((run_dir / name).exists() for name in REQUIRED_FILES):
            return run_dir
    return None


def load_case(run_dir: Path) -> dict:
    with open(run_dir / "params.json", "r", encoding="utf-8") as fp:
        params = json.load(fp)

    return {
        "run_dir": run_dir,
        "z": np.load(run_dir / "z.npy"),
        "x": np.load(run_dir / "x_grid.npy"),
        "y": np.load(run_dir / "y_grid.npy"),
        "temperature": np.load(run_dir / "T_profile.npy"),
        "density": np.load(run_dir / "rho_profile.npy"),
        "n2": np.load(run_dir / "N2_profile.npy"),
        "w_profile": np.load(run_dir / "W_profile.npy"),
        "u_profile": np.load(run_dir / "U_profile.npy"),
        "temperature_3d": np.load(run_dir / "T_3D.npy"),
        "w_vel_3d": np.load(run_dir / "W_Vel_3D.npy"),
        "u_vel_3d": np.load(run_dir / "U_Vel_3D.npy"),
        "params": params,
    }


def save_figure(fig: plt.Figure, name: str) -> Path:
    out_path = RESULTS_DIR / name
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    return out_path


def plot_background_stratification(case: dict) -> Path:
    z = case["z"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.8))

    axes[0].plot(case["temperature"], z, color="#c7382f", lw=2)
    axes[0].set_title("Temperature")
    axes[0].set_xlabel("deg C")

    axes[1].plot(case["density"], z, color="#2458a6", lw=2)
    axes[1].set_title("Density")
    axes[1].set_xlabel("kg/m^3")

    axes[2].plot(case["n2"], z, color="#237447", lw=2)
    axes[2].set_title("Buoyancy frequency N2")
    axes[2].set_xlabel("s^-2")

    for ax in axes:
        ax.set_ylim(z.max(), z.min())
        ax.set_ylabel("Depth (m)")
        ax.grid(alpha=0.25)

    fig.suptitle("ModelF Background Stratification", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "ModelF_Background_Stratification.png")


def plot_vertical_structure(case: dict) -> Path:
    z = case["z"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.8))

    axes[0].plot(case["n2"], z, color="#237447", lw=2)
    axes[0].set_title("N2")

    axes[1].plot(case["w_profile"], z, color="#2355a0", lw=2.3)
    axes[1].axvline(0, color="0.25", lw=1, ls=":")
    axes[1].set_title("Vertical mode W(z)")

    axes[2].plot(case["u_profile"], z, color="#b33a33", lw=2.3)
    axes[2].axvline(0, color="0.25", lw=1, ls=":")
    axes[2].set_title("Horizontal mode U(z)=dW/dz")

    z_star = float(case["params"].get("thermocline_depth", z[np.argmax(np.abs(case["w_profile"]))]))
    for ax in axes:
        ax.axhline(z_star, color="black", lw=1.2, ls="--", alpha=0.7)
        ax.set_ylim(z.max(), z.min())
        ax.set_ylabel("Depth (m)")
        ax.grid(alpha=0.25)

    fig.suptitle("ModelF Vertical Structure", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "ModelF_Vertical_Structure.png")


def plot_temperature_slices(case: dict) -> Path:
    x = case["x"]
    y = case["y"]
    z = case["z"]
    temp = case["temperature_3d"]

    z_star = float(case["params"].get("thermocline_depth", z[np.argmax(np.abs(case["w_profile"]))]))
    z_idx = int(np.argmin(np.abs(z - z_star)))
    y_center_idx = len(y) // 2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    if len(y) > 1:
        xy = temp[:, :, z_idx]
        c0 = axes[0].contourf(y / 1000.0, x / 1000.0, xy, levels=28, cmap="RdYlBu_r")
        axes[0].contour(y / 1000.0, x / 1000.0, xy, levels=8, colors="k", linewidths=0.35, alpha=0.35)
        axes[0].set_xlabel("Along-crest y (km)")
        axes[0].set_ylabel("Propagation x (km)")
        axes[0].set_title(f"X-Y temperature at z ~= {z[z_idx]:.0f} m")
    else:
        c0 = axes[0].plot(x / 1000.0, temp[:, 0, z_idx], color="#c7382f", lw=2)[0]
        axes[0].set_xlabel("Propagation x (km)")
        axes[0].set_ylabel("Temperature (deg C)")
        axes[0].set_title(f"Temperature line at z ~= {z[z_idx]:.0f} m")
        axes[0].grid(alpha=0.25)

    xz = temp[:, y_center_idx, :]
    x_mesh, z_mesh = np.meshgrid(x / 1000.0, z, indexing="ij")
    c1 = axes[1].contourf(x_mesh, z_mesh, xz, levels=28, cmap="RdYlBu_r")
    axes[1].contour(x_mesh, z_mesh, xz, levels=8, colors="k", linewidths=0.35, alpha=0.35)
    axes[1].set_ylim(z.max(), z.min())
    axes[1].set_xlabel("Propagation x (km)")
    axes[1].set_ylabel("Depth (m)")
    axes[1].set_title("X-Z temperature at y center")

    if len(y) > 1:
        fig.colorbar(c0, ax=axes[0], label="Temperature (deg C)", pad=0.02)
    fig.colorbar(c1, ax=axes[1], label="Temperature (deg C)", pad=0.02)
    fig.suptitle("ModelF Temperature Field", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "ModelF_Temperature_Slices.png")


def wave_profile_at_y_center(case: dict) -> tuple[np.ndarray, np.ndarray]:
    x = case["x"]
    y = case["y"]
    z = case["z"]
    params = case["params"]
    y_center = float(y[len(y) // 2])
    x_effective = x - float(params.get("a_coef", 0.0)) * y_center**2
    d_scale = float(params.get("D", 1000.0))
    h0 = float(params.get("h0", 0.0))
    z_star = float(params.get("thermocline_depth", z[np.argmax(np.abs(case["w_profile"]))]))
    displacement = h0 * (1.0 / np.cosh(x_effective / d_scale)) ** 2
    return x, z_star + displacement


def plot_velocity_xz(case: dict, key: str, title: str, output_name: str, colorbar_label: str) -> Path:
    x = case["x"]
    z = case["z"]
    y_center_idx = len(case["y"]) // 2
    field = case[key][:, y_center_idx, :]

    vmax = float(np.nanmax(np.abs(field)))
    norm = CenteredNorm(0.0, halfrange=max(vmax, 1e-12))
    x_mesh, z_mesh = np.meshgrid(x / 1000.0, z, indexing="ij")
    wave_x, wave_z = wave_profile_at_y_center(case)

    max_idx = np.unravel_index(np.nanargmax(np.abs(field)), field.shape)
    max_depth = z[max_idx[1]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for ax in axes:
        contour = ax.contourf(x_mesh, z_mesh, field, levels=41, cmap="bwr", norm=norm)
        ax.contour(x_mesh, z_mesh, field, levels=13, colors="k", linewidths=0.3, alpha=0.25, norm=norm)
        ax.set_ylim(z.max(), z.min())
        ax.set_xlabel("Propagation x (km)")
        ax.set_ylabel("Depth (m)")
        ax.grid(alpha=0.18)

    axes[0].set_title(f"{title} x-z section")
    axes[1].plot(wave_x / 1000.0, wave_z, color="black", lw=2.2, label="displaced strong layer")
    axes[1].fill_between(wave_x / 1000.0, case["params"].get("thermocline_depth", 150.0), wave_z, color="#f3d46b", alpha=0.28)
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].set_title(f"{title} with wave displacement\nmax |velocity| depth ~= {max_depth:.0f} m")

    fig.colorbar(contour, ax=axes, label=colorbar_label, pad=0.02)
    fig.suptitle(f"ModelF {title}", fontsize=13, fontweight="bold")
    return save_figure(fig, output_name)


def plot_vertical_velocity_3d(case: dict) -> Path | None:
    y = case["y"]
    if len(y) <= 1:
        return None

    x = case["x"]
    z = case["z"]
    field = case["w_vel_3d"]
    max_idx = np.unravel_index(np.nanargmax(np.abs(field)), field.shape)
    depth_idx = max_idx[2]
    w_xy = field[:, :, depth_idx]

    x_mesh, y_mesh = np.meshgrid(x / 1000.0, y / 1000.0, indexing="ij")
    vmax = float(np.nanmax(np.abs(w_xy)))
    norm = CenteredNorm(0.0, halfrange=max(vmax, 1e-12))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        x_mesh,
        y_mesh,
        w_xy,
        cmap="bwr",
        norm=norm,
        edgecolor="none",
        antialiased=True,
        alpha=0.92,
    )
    ax.view_init(elev=34, azim=-58)
    ax.set_title(f"ModelF 3D vertical velocity surface at z ~= {z[depth_idx]:.0f} m")
    ax.set_xlabel("Propagation x (km)")
    ax.set_ylabel("Along-crest y (km)")
    ax.set_zlabel("Vertical velocity (m/s)")
    fig.colorbar(surface, ax=ax, shrink=0.58, pad=0.08, label="Vertical velocity (m/s)")
    return save_figure(fig, "ModelF_Vertical_Velocity_3D_Surface.png")


def make_visualizations(case: dict) -> list[Path]:
    outputs = [
        plot_background_stratification(case),
        plot_vertical_structure(case),
        plot_temperature_slices(case),
        plot_velocity_xz(
            case,
            "w_vel_3d",
            "Vertical Velocity Field",
            "ModelF_Vertical_Velocity_Field.png",
            "Vertical velocity (m/s)",
        ),
        plot_velocity_xz(
            case,
            "u_vel_3d",
            "Horizontal Velocity Field",
            "ModelF_Horizontal_Velocity_Field.png",
            "Horizontal velocity (m/s)",
        ),
    ]
    surface_path = plot_vertical_velocity_3d(case)
    if surface_path is not None:
        outputs.append(surface_path)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize generated ModelF flow fields.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Specific ModelF case directory to plot.")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate Ideal_000 before plotting.")
    parser.add_argument("--nx", type=int, default=120, help="x grid size when regenerating.")
    parser.add_argument("--ny", type=int, default=80, help="y grid size when regenerating; use 1 for x-z only.")
    parser.add_argument("--nz", type=int, default=500, help="z grid size when regenerating.")
    parser.add_argument("--seed", type=int, default=20260528, help="Random seed when regenerating.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.regenerate:
        generate_batch(count=1, seed=args.seed, nx=args.nx, ny=args.ny, nz=args.nz)

    run_dir = args.run_dir or latest_complete_case()
    if run_dir is None:
        generate_batch(count=1, seed=args.seed, nx=args.nx, ny=args.ny, nz=args.nz)
        run_dir = latest_complete_case()

    if run_dir is None:
        raise FileNotFoundError("No complete ModelF case is available for visualization.")

    case = load_case(run_dir)
    outputs = make_visualizations(case)
    print(f"case={run_dir}")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()

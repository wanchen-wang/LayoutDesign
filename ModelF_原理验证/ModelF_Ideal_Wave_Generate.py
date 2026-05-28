"""ModelF self-consistent internal solitary wave generator.

ModelF follows the same generation stages as ModelA:

1. Generate a background stratification profile.
2. Solve the first-mode vertical structure.
3. Generate an internal solitary wave temperature and velocity field.
4. Save the arrays with the same names expected by the downstream samplers.

The only deliberate physical difference is the material-layer lookup used for
velocity.  Temperature displacement first tells us which original water layer
has moved to the current depth.  The vertical and horizontal velocity fields
then use that original layer label, not the current geometric depth.  This
keeps displacement and velocity tied to the same water parcel.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eig


OUT_DIR = Path(__file__).resolve().parent / "Generated_Ideal_Wave_Data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sech2(value: np.ndarray | float) -> np.ndarray | float:
    """Return sech(value)^2."""
    return 1.0 / np.cosh(value) ** 2


def generate_background_stratification(
    rng: np.random.Generator,
    depth_max: float = 1000.0,
    num_points: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a ModelA-style random temperature, density and N2 profile."""
    z = np.linspace(0.0, depth_max, num_points)

    thermocline_depth = rng.uniform(80.0, 200.0)
    thermocline_thickness = rng.uniform(30.0, 80.0)
    surface_temp = rng.uniform(27.0, 29.0)
    bottom_temp = rng.uniform(3.0, 5.0)

    temperature = bottom_temp + (surface_temp - bottom_temp) * 0.5 * (
        1.0 + np.tanh((thermocline_depth - z) / thermocline_thickness)
    )

    rho_0 = 1024.0
    thermal_expansion = 0.2
    density = 1028.0 - thermal_expansion * temperature

    drho_dz = np.gradient(density, z)
    n2 = np.maximum((9.81 / rho_0) * drho_dz, 1e-7)
    return z, temperature, density, n2


def calculate_vertical_structure(
    z: np.ndarray,
    n2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve the first vertical mode using the same eigenvalue logic as ModelA."""
    dz = z[1] - z[0]
    n_points = len(z)

    main_diag = -2.0 * np.ones(n_points - 2)
    off_diag = np.ones(n_points - 3)
    d2 = sp.diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1]) / dz**2

    n2_clean = np.asarray(n2, dtype=float).copy()
    n2_clean[~np.isfinite(n2_clean)] = 1e-7
    n2_clean = np.maximum(n2_clean, 1e-7)
    n2_interior = sp.diags(n2_clean[1:-1], 0)

    evals, evecs = eig(d2.toarray(), n2_interior.toarray())
    idx = int(np.argmin(np.abs(evals)))
    eigenvalue = float(np.real(evals[idx]))
    c0 = float(1.1 * np.sqrt(-1.0 / eigenvalue))

    w_interior = np.real(evecs[:, idx])
    w_interior = w_interior / np.max(np.abs(w_interior))
    if w_interior[np.argmax(np.abs(w_interior))] < 0.0:
        w_interior = -w_interior

    w_profile = np.zeros(n_points)
    w_profile[1:-1] = w_interior
    u_profile = np.gradient(w_profile, z)
    return w_profile, u_profile, c0


def invert_layer_labels(
    z_grid: np.ndarray,
    displaced_depth: np.ndarray,
) -> np.ndarray:
    """Map current depths back to their original material-layer labels."""
    order = np.argsort(displaced_depth)
    return np.interp(
        z_grid,
        displaced_depth[order],
        z_grid[order],
        left=float(z_grid.min()),
        right=float(z_grid.max()),
    )


def calculate_nonlinear_phase_speed(
    z: np.ndarray,
    u_profile: np.ndarray,
    c0: float,
    h0: float,
) -> float:
    """Apply ModelA's KdV-style nonlinear phase-speed correction."""
    integral_u3 = np.trapezoid(u_profile**3, z)
    integral_u2 = np.trapezoid(u_profile**2, z)
    if abs(integral_u2) < 1e-12:
        return c0
    alpha = (3.0 * c0 / 2.0) * (integral_u3 / integral_u2)
    return float(c0 + (alpha * h0) / 3.0)


def generate_self_consistent_isw_block(
    rng: np.random.Generator,
    z: np.ndarray,
    temperature: np.ndarray,
    w_profile: np.ndarray,
    u_profile: np.ndarray,
    c0: float,
    nx: int = 100,
    ny: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    """Generate a ModelA-style ISW block, with ModelF material-label velocities."""
    x_grid = np.linspace(-5000.0, 5000.0, nx)
    ly = float(rng.uniform(50000.0, 100000.0))
    y_grid = np.array([0.0]) if ny == 1 else np.linspace(-ly / 2.0, ly / 2.0, ny)

    h0 = float(rng.uniform(80.0, 150.0))
    d_scale = float(rng.uniform(800.0, 1500.0))
    phase_speed = calculate_nonlinear_phase_speed(z, u_profile, c0, h0)

    max_offset = float(rng.uniform(1000.0, 3000.0))
    a_coef = max_offset / (ly / 2.0) ** 2

    temp_3d = np.empty((len(x_grid), len(y_grid), len(z)), dtype=float)
    w_vel_3d = np.empty_like(temp_3d)
    u_vel_3d = np.empty_like(temp_3d)

    for ix, x in enumerate(x_grid):
        for iy, y in enumerate(y_grid):
            x_effective = x - a_coef * y**2
            shape = float(sech2(x_effective / d_scale))
            slope_shape = shape * np.tanh(x_effective / d_scale)

            displaced_depth = z + h0 * shape * np.abs(w_profile)
            label_z = invert_layer_labels(z, displaced_depth)

            temp_3d[ix, iy, :] = np.interp(label_z, z, temperature)
            w_label = np.interp(label_z, z, w_profile)
            u_label = np.interp(label_z, z, u_profile)

            w_vel_3d[ix, iy, :] = (2.0 * h0 * phase_speed / d_scale) * slope_shape * w_label
            u_vel_3d[ix, iy, :] = phase_speed * h0 * shape * u_label

    return x_grid, y_grid, temp_3d, w_vel_3d, u_vel_3d, h0, ly, a_coef, d_scale


def save_run_data(
    run_dir: Path,
    z: np.ndarray,
    temperature: np.ndarray,
    density: np.ndarray,
    n2: np.ndarray,
    w_profile: np.ndarray,
    u_profile: np.ndarray,
    c0: float,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    temp_3d: np.ndarray,
    w_vel_3d: np.ndarray,
    u_vel_3d: np.ndarray,
    h0: float,
    ly: float,
    a_coef: float,
    d_scale: float,
) -> Path:
    """Save ModelF data using ModelA-compatible file names."""
    run_dir.mkdir(parents=True, exist_ok=True)

    np.save(run_dir / "z.npy", z)
    np.save(run_dir / "T_profile.npy", temperature)
    np.save(run_dir / "rho_profile.npy", density)
    np.save(run_dir / "N2_profile.npy", n2)
    np.save(run_dir / "W_profile.npy", w_profile)
    np.save(run_dir / "U_profile.npy", u_profile)
    np.save(run_dir / "x_grid.npy", x_grid)
    np.save(run_dir / "y_grid.npy", y_grid)
    np.save(run_dir / "T_3D.npy", temp_3d)
    np.save(run_dir / "W_Vel_3D.npy", w_vel_3d)
    np.save(run_dir / "U_Vel_3D.npy", u_vel_3d)

    params = {
        "c0": float(c0),
        "h0": float(h0),
        "Ly": float(ly),
        "a_coef": float(a_coef),
        "D": float(d_scale),
        "thermocline_depth": float(z[np.argmax(np.abs(w_profile))]),
        "generator": "ModelF_ModelA_Process_With_Material_Layer_Velocity",
        "velocity_coordinate": "original_material_layer_label",
    }
    with open(run_dir / "params.json", "w", encoding="utf-8") as fp:
        json.dump(params, fp, indent=2, ensure_ascii=False)

    return run_dir


def generate_one(
    index: int,
    rng: np.random.Generator,
    nx: int = 100,
    ny: int = 1,
    nz: int = 500,
) -> Path:
    """Generate one ModelF wave case."""
    run_dir = OUT_DIR / f"Ideal_{index:03d}"

    z, temperature, density, n2 = generate_background_stratification(rng, num_points=nz)
    w_profile, u_profile, c0 = calculate_vertical_structure(z, n2)
    x_grid, y_grid, temp_3d, w_vel_3d, u_vel_3d, h0, ly, a_coef, d_scale = (
        generate_self_consistent_isw_block(
            rng,
            z,
            temperature,
            w_profile,
            u_profile,
            c0,
            nx=nx,
            ny=ny,
        )
    )

    return save_run_data(
        run_dir,
        z,
        temperature,
        density,
        n2,
        w_profile,
        u_profile,
        c0,
        x_grid,
        y_grid,
        temp_3d,
        w_vel_3d,
        u_vel_3d,
        h0,
        ly,
        a_coef,
        d_scale,
    )


def generate_batch(
    count: int = 20,
    seed: int = 20260527,
    nx: int = 100,
    ny: int = 1,
    nz: int = 500,
) -> list[Path]:
    """Generate a deterministic batch of ModelF wave cases."""
    rng = np.random.default_rng(seed)
    return [generate_one(i, rng, nx=nx, ny=ny, nz=nz) for i in range(count)]


if __name__ == "__main__":
    paths = generate_batch()
    print(OUT_DIR)
    print(f"generated={len(paths)}")

import numpy as np

from Model1_Petrel_Glider_Inverse_Dynamics import PetrelGliderInverseDynamics

petrel = PetrelGliderInverseDynamics()


def fetch_surface_data(zeta_array, v_array, solver=None):
    """遍历 (滑翔角, 速度) 网格，获取电池位移和净浮力。"""
    solver = solver or petrel
    zeta_grid, velocity_grid = np.meshgrid(zeta_array, v_array)

    rp1_grid = np.zeros_like(zeta_grid)
    delta_b_grid = np.zeros_like(zeta_grid)

    for i in range(zeta_grid.shape[0]):
        for j in range(zeta_grid.shape[1]):
            try:
                result = solver.inverse_kinematics_solver(
                    V_target=velocity_grid[i, j],
                    zeta_target_deg=zeta_grid[i, j],
                )
                rp1_grid[i, j] = result["所需电池包位移_rp1 (m)"] * 100
                delta_b_grid[i, j] = result["所需净浮力_Delta_B (N)"]
            except ValueError:
                rp1_grid[i, j] = np.nan
                delta_b_grid[i, j] = np.nan

    return zeta_grid, velocity_grid, rp1_grid, delta_b_grid


def _safe_theta(solver, zeta_deg):
    try:
        return np.degrees(solver.solve_angles_analytical(zeta_deg)[0])
    except ValueError:
        return np.nan


def compute_theta_bounds(solver=None):
    """把 theta=±15° 和 ±45° 映射回对应的滑翔角。"""
    solver = solver or petrel

    zeta_scan_climb = np.linspace(3, 55, 2000)
    theta_scan_climb = np.array([_safe_theta(solver, zeta) for zeta in zeta_scan_climb])
    climb_mask = ~np.isnan(theta_scan_climb)
    zeta_theta15_climb = float(np.interp(15.0, theta_scan_climb[climb_mask], zeta_scan_climb[climb_mask]))
    zeta_theta45_climb = float(np.interp(45.0, theta_scan_climb[climb_mask], zeta_scan_climb[climb_mask]))

    zeta_scan_dive = np.linspace(-55, -3, 2000)
    theta_scan_dive = np.array([_safe_theta(solver, zeta) for zeta in zeta_scan_dive])
    dive_mask = ~np.isnan(theta_scan_dive)
    zeta_theta15_dive = float(np.interp(-15.0, theta_scan_dive[dive_mask], zeta_scan_dive[dive_mask]))
    zeta_theta45_dive = float(np.interp(-45.0, theta_scan_dive[dive_mask], zeta_scan_dive[dive_mask]))

    return {
        "zeta_theta15_climb": zeta_theta15_climb,
        "zeta_theta45_climb": zeta_theta45_climb,
        "zeta_theta15_dive": zeta_theta15_dive,
        "zeta_theta45_dive": zeta_theta45_dive,
    }


def build_model2_data(solver=None):
    """生成 Model2 所需的全部网格数据与约束边界。"""
    solver = solver or petrel

    v_range = np.linspace(0.1, 0.6, 50)
    zeta_dive = np.linspace(-55, -3, 50)
    zeta_climb = np.linspace(3, 55, 50)

    zeta_d_grid, v_d_grid, rp1_d_grid, delta_b_d_grid = fetch_surface_data(zeta_dive, v_range, solver=solver)
    zeta_c_grid, v_c_grid, rp1_c_grid, delta_b_c_grid = fetch_surface_data(zeta_climb, v_range, solver=solver)
    theta_bounds = compute_theta_bounds(solver=solver)

    return {
        "v_range": v_range,
        "zeta_dive": zeta_dive,
        "zeta_climb": zeta_climb,
        "ZETA_D": zeta_d_grid,
        "V_D": v_d_grid,
        "RP1_D": rp1_d_grid,
        "DELTA_B_D": delta_b_d_grid,
        "ZETA_C": zeta_c_grid,
        "V_C": v_c_grid,
        "RP1_C": rp1_c_grid,
        "DELTA_B_C": delta_b_c_grid,
        **theta_bounds,
    }


def main():
    data = build_model2_data()
    print(
        f"下潜阶段：θ=-45° → ζ={data['zeta_theta45_dive']:.2f}°，"
        f"θ=-15° → ζ={data['zeta_theta15_dive']:.2f}°"
    )
    print(
        f"上浮阶段：θ=+15° → ζ={data['zeta_theta15_climb']:.2f}°，"
        f"θ=+45° → ζ={data['zeta_theta45_climb']:.2f}°"
    )
    return data


if __name__ == "__main__":
    main()
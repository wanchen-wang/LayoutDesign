"""
内孤立波三维可视化程序
z轴设置：海面为0，向下为正方向（0到1000米）
"""

import os
import json
import datetime

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eig


# 数据保存工具

def save_run_data(base_dir, z, T, rho, N2, W, U, c0,
                  x_grid, y_grid, T_3D, W_Vel_3D,
                  h0, Ly, a_coef, D,
                  extra_slices=None):
    """Save all relevant arrays and parameters for one simulation run.

    Parameters
    ----------
    base_dir : str
        Base output directory where run folders will be created.
    extra_slices : dict, optional
        Dictionary of additional 2D slices to save, where keys are names
        (e.g. "xz_temp") and values are the 2D numpy arrays.
    """
    # create hierarchical directory: base/YYYYMMDD_HHMMSS/
    now = datetime.datetime.now()
    run_name = now.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # save 1D profiles
    np.save(os.path.join(run_dir, 'z.npy'), z)
    np.save(os.path.join(run_dir, 'T_profile.npy'), T)
    np.save(os.path.join(run_dir, 'rho_profile.npy'), rho)
    np.save(os.path.join(run_dir, 'N2_profile.npy'), N2)
    np.save(os.path.join(run_dir, 'W_profile.npy'), W)
    np.save(os.path.join(run_dir, 'U_profile.npy'), U)

    # save 3D fields
    np.save(os.path.join(run_dir, 'T_3D.npy'), T_3D)
    np.save(os.path.join(run_dir, 'W_Vel_3D.npy'), W_Vel_3D)

    # save horizontal grids
    np.save(os.path.join(run_dir, 'x_grid.npy'), x_grid)
    np.save(os.path.join(run_dir, 'y_grid.npy'), y_grid)

    # save parameters as json
    params = {
        'c0': float(c0),
        'h0': float(h0),
        'Ly': float(Ly),
        'a_coef': float(a_coef),
        'D': float(D),
        'thermocline_depth': float(z[np.argmax(np.abs(W))]),  # 温跃层核心深度
    }
    with open(os.path.join(run_dir, 'params.json'), 'w') as fp:
        json.dump(params, fp, indent=2)

    # save extra 2D slices (e.g., x-z cross sections)
    if extra_slices is not None:
        for name, arr in extra_slices.items():
            path = os.path.join(run_dir, f"{name}.csv")
            # assume arr is 2D and we can write with header rows
            # We'll stack x (row) with columns for each z
            np.savetxt(path, arr, delimiter=',')

    return run_dir


def generate_background_stratification(depth_max=1000, num_points=500):
    """
    第一步：生成具有一定随机性的海洋背景层化剖面
    z轴：海面为0，向下为正方向（0到depth_max米）
    """
    # 1. 设定垂直深度网格（0 到 depth_max 米，向下为正）
    z = np.linspace(0, depth_max, num_points)
    
    # 2. 引入随机性：随机生成温跃层的中心深度和厚度
    # 南海的主温跃层通常在 50-250m 之间
    thermocline_depth = np.random.uniform(80, 200)  # 温跃层中心深度（随机范围: 80m 到 200m）
    thermocline_thickness = np.random.uniform(30, 80)  # 温跃层厚度/缓和度（随机范围）
    
    # 3. 模拟温度剖面（使用双曲正切函数模拟真实的三层海洋结构）
    T_surface = np.random.uniform(27, 29)  # 表层温度约 27-29 度
    T_bottom = np.random.uniform(3, 5)     # 1000m深处温度约 3-5 度
    
    # 计算每层的温度（z向下为正，所以z越大温度越低）
    Temperature = T_bottom + (T_surface - T_bottom) * 0.5 * (1 + np.tanh((thermocline_depth - z) / thermocline_thickness))
    
    # 4. 根据温度计算海水密度（这里使用一个简化的线性状态方程）
    # 真实海洋密度在 1021 - 1027 kg/m^3 之间浮动
    rho_0 = 1024.0
    alpha = 0.2  # 热膨胀系数的近似倍数
    Density = 1028 - alpha * Temperature 
    
    # 5. 计算浮力频率的平方 N^2 = (g/rho_0) * (d_rho / d_z)
    # z向下为正时，密度随深度增加，drho_dz为正，N^2为正
    g = 9.81
    drho_dz = np.gradient(Density, z) 
    N2 = (g / rho_0) * drho_dz
    # 确保 N2 不为负数（维持物理上的稳定层化）
    N2 = np.maximum(N2, 1e-7)
    
    return z, Temperature, Density, N2


def calculate_vertical_structure(z, N2):
    """
    第二步：计算第一模态内波垂直结构
    """
    # grid spacing
    dz = z[1] - z[0]  # use first interval (should be uniform)
    N_points = len(z)

    # 2. 构建有限差分二阶导数矩阵
    main_diag = -2.0 * np.ones(N_points - 2)
    off_diag = np.ones(N_points - 3)
    D2 = sp.diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1]) / (dz**2)

    # 3. 清理 N2 并构建对角矩阵（内部点）
    N2_clean = np.copy(N2)
    mask = ~np.isfinite(N2_clean)
    if np.any(mask):
        N2_clean[mask] = 1e-7
    N2_clean = np.maximum(N2_clean, 1e-7)
    N2_interior = sp.diags(N2_clean[1:-1], 0)

    # 4. 求解广义特征值问题
    A = D2.toarray()
    B = N2_interior.toarray()
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(B)):
        raise ValueError("Matrices A or B contain NaN/Inf; check N2 profile")
    evals_all, evecs_all = eig(A, B)
    
    # 选择绝对值最小的特征值（第一斜压模态）
    idx = np.argmin(np.abs(evals_all))
    eigenvalue = np.real(evals_all[idx])
    c0 = 1.1 * np.sqrt(-1.0 / eigenvalue)
    
    # 5. 提取特征向量，构建垂直结构 W
    W_interior = np.real(evecs_all[:, idx])
    # W(z) 标准化为最大值为 1 （这是合理的，它决定了波幅的基准参考量级）
    W_interior = W_interior / np.max(np.abs(W_interior))
    if W_interior[np.argmax(np.abs(W_interior))] < 0:
        W_interior = -W_interior
        
    # 6. 拼合边界条件（海面海底 W=0）
    W = np.zeros(N_points)
    W[1:-1] = W_interior
    
    # 7. 计算水平结构 U(z)
    # 【核心修复】：必须严格遵从 U(z) = dW/dz 的物理推导，绝对不能单独对U进行标准化！
    U = np.gradient(W, z)
    
    return W, U, c0


def generate_3d_curved_isw_block(z, W, U, c0, T):
    """
    第三步：生成带有"弯曲波前"的三维大尺度方块
    生成三维的内孤立波温度和流速场，包含 y 轴的超大跨度和弯曲的波前
    """
    # 1. 设定水平空间网格
    nx, ny = 100, 200  
    x = np.linspace(-5000, 5000, nx) 
    Ly = np.random.uniform(50000, 100000) 
    y = np.linspace(-Ly/2, Ly/2, ny)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # 2. 设定孤立波的物理参数
    amplitude_h0 = np.random.uniform(80, 150) # 随机最大波幅（向下为正）
    D = np.random.uniform(800, 1500)          # 特征半宽度

    # ==========================================
    # 核心修正：计算环境非线性参数 alpha 与 非线性相速 C
    # ==========================================
    # 利用水平模态结构 U(z) 积分计算二次非线性系数 alpha (KdV理论)
    integral_U3 = np.trapezoid(U**3, z)
    integral_U2 = np.trapezoid(U**2, z)
    alpha = (3.0 * c0 / 2.0) * (integral_U3 / integral_U2)
    
    # 计算受到振幅非线性加速影响的真实波速 C_nonlinear
    C_nonlinear = c0 + (alpha * amplitude_h0) / 3.0
    
    # 3. 引入波前几何曲率（抛物线弯曲模型）
    max_offset = np.random.uniform(1000, 3000)
    a_coef = max_offset / (Ly / 2)**2
    X_crest = a_coef * Y**2
    X_effective = X - X_crest
    
    # 4. 计算三维空间剖面
    sech2_x = (1.0 / np.cosh(X_effective / D))**2
    sech2_tanh_x = sech2_x * np.tanh(X_effective / D)
    
    W_3d = W.reshape(1, 1, len(z))
    Displacement_3D = amplitude_h0 * sech2_x * np.abs(W_3d) 
    
    Effective_Z = Z - Displacement_3D
    Temperature_3D = np.interp(Effective_Z, z, T)
    
    # 6. 生成三维垂直流速场
    # ==========================================
    # 核心修正：补充求导法则遗漏的常数项 2.0，并替换为非线性波速 C_nonlinear
    # ==========================================
    empirical_factor = 1
    Vertical_Velocity_3D = empirical_factor * ((2.0 * amplitude_h0 * C_nonlinear / D) * sech2_tanh_x * W_3d)
    
    return x, y, Temperature_3D, Vertical_Velocity_3D, amplitude_h0, Ly, a_coef, D






# ==========================================
# 主程序执行函数
# ==========================================

def run_simulation(save=True):
    """Execute one full calculation/visualization run.

    Parameters
    ----------
    save : bool
        If True, data files (including xz csv) are written to `output/`
        in a timestamped subdirectory.
    """
    # 第一步：生成背景层化剖面
    print("正在生成海洋背景层化剖面...")
    z, T, rho, N2 = generate_background_stratification()

    # 第二步：计算垂直结构
    print("正在计算第一模态内波垂直结构...")
    W, U, c0 = calculate_vertical_structure(z, N2)
    print(f"计算成功！该次随机生成的环境第一模态波速 c0 约为: {c0:.2f} m/s")

    # 第三步：生成三维数据
    print("正在生成三维内孤立波数据...")
    x_grid, y_grid, T_3D, W_Vel_3D, h0, Ly, a_coef, D = generate_3d_curved_isw_block(z, W, U, c0, T)
    print(f"成功生成带有曲率的大尺度三维内孤立波数据！")
    print(f"   随机生成的 Y 轴跨度为: {Ly/1000:.1f} km")
    print(f"   数据方块尺寸 (X, Y, Z): 10km × {Ly/1000:.1f}km × 1000m")
    print(f"   三维温度矩阵形状: {T_3D.shape}")

    # x-z 剖面准备
    y_center_idx = len(y_grid) // 2
    xz_temp = T_3D[:, y_center_idx, :]
    xz_vel = W_Vel_3D[:, y_center_idx, :]

    def save_xz_csv(array2d, x_coord, z_coord, file_path):
        """Write a 2D array with x/z coordinates as a CSV file."""
        with open(file_path, 'w') as f:
            header = ',' + ','.join(f"{zi:.3f}" for zi in z_coord)
            f.write(header + '\n')
            for xi, row in zip(x_coord, array2d):
                line = f"{xi:.3f}," + ','.join(f"{val:.6e}" for val in row)
                f.write(line + '\n')

    run_directory = None
    if save:
        base_output = os.path.join(os.getcwd(), 'v_wave_data')
        run_directory = save_run_data(base_output, z, T, rho, N2, W, U, c0,
                                      x_grid, y_grid, T_3D, W_Vel_3D,
                                      h0, Ly, a_coef, D)
        save_xz_csv(xz_temp, x_grid, z, os.path.join(run_directory, 'xz_temp.csv'))
        save_xz_csv(xz_vel, x_grid, z, os.path.join(run_directory, 'xz_vel.csv'))
        print(f"数据已保存到：{run_directory}")
        print(f"xz 剖面 CSV 文件已生成：{os.path.join(run_directory, 'xz_temp.csv')} 和 xz_vel.csv")

    return run_directory


"""
内孤立波三维可视化程序
z轴设置：海面为0，向下为正方向（0到1000米）
"""

import os
import json
import datetime

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.linalg import eig
from mpl_toolkits.mplot3d import Axes3D


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


def plot_multiple_3d_isotherm_surfaces(x, y, z, W, a_coef, h0, D, Ly, T_3D):
    """
    绘制多个等温面的3D曲面图，展示不同深度等温面的振幅变化
    不使用深度分颜色，而是用不同颜色区分不同等温面
    """
    # 1. 找到垂直结构 W(z) 起伏最大的深度索引（即温跃层核心）
    max_w_idx = np.argmax(np.abs(W))
    base_depth = z[max_w_idx]  # 该等温面在没有波浪时的平静深度
    
    # 2. 选择三个不同深度的等温面
    # 第一个：最大振幅所在深度
    depth1 = base_depth
    idx1 = max_w_idx
    
    # 第二个：比最大振幅深度浅一些（向上，振幅较小）
    depth2 = base_depth - 90  # 向上30米
    idx2 = np.argmin(np.abs(z - depth2))
    
    # 第三个：比最大振幅深度深一些（向下，振幅较小）
    depth3 = base_depth + 90  # 向下30米
    idx3 = np.argmin(np.abs(z - depth3))
    
    # 3. 构建 2D 水平网格 (x-y 平面)
    X_2D, Y_2D = np.meshgrid(x, y, indexing='ij')
    
    # 4. 计算二维平面上的弯曲波前
    X_crest_2d = a_coef * Y_2D**2
    X_effective_2d = X_2D - X_crest_2d
    sech2_2d = (1.0 / np.cosh(X_effective_2d / D))**2
    
    # 5. 计算三个等温面的实际深度
    # 对于下凹型内孤立波，在波峰处等温面向下移动（深度增加）
    # W在第一斜压模态中通常为正值，但内孤立波是下凹的，所以位移应该使等温面向下移动
    # 使用绝对值确保下凹（深度增加）
    Surface_Z1 = depth1 + (h0 * sech2_2d * np.abs(W[idx1]))  # 最大振幅等温面
    Surface_Z2 = depth2 + (h0 * sech2_2d * np.abs(W[idx2]))  # 较浅等温面（振幅较小）
    Surface_Z3 = depth3 + (h0 * sech2_2d * np.abs(W[idx3]))  # 较深等温面（振幅较小）
    #print(f"浅层衰减系数: {np.abs(W[idx2]):.3f}")
    #print(f"核心衰减系数: {np.abs(W[idx1]):.3f}")
    #print(f"深层衰减系数: {np.abs(W[idx3]):.3f}")
    
    # 6. 开始 3D 渲染绘制
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制三个等温面，使用不同颜色和透明度
    surf1 = ax.plot_surface(X_2D/1000, Y_2D/1000, Surface_Z1, 
                           color='red', alpha=0.7, edgecolor='none', antialiased=True, label=f'Depth {depth1:.0f}m')
    surf2 = ax.plot_surface(X_2D/1000, Y_2D/1000, Surface_Z2, 
                           color='blue', alpha=0.6, edgecolor='none', antialiased=True, label=f'Depth {depth2:.0f}m')
    surf3 = ax.plot_surface(X_2D/1000, Y_2D/1000, Surface_Z3, 
                           color='green', alpha=0.6, edgecolor='none', antialiased=True, label=f'Depth {depth3:.0f}m')
    
    # 7. 调整视角与坐标轴标签
    ax.view_init(elev=35, azim=-60)
    
    ax.set_title(f"3D Multiple Isotherm Surfaces\nRed: {depth1:.0f}m (max amplitude), Blue: {depth2:.0f}m, Green: {depth3:.0f}m", 
                 fontsize=14, pad=20)
    ax.set_xlabel("Propagation Distance x (km)", labelpad=10)
    ax.set_ylabel("Along-Crest Distance y (km)", labelpad=10)
    ax.set_zlabel("Depth z (m)", labelpad=10)
    
    # 限制 Z 轴深度显示范围
    z_min = min(np.min(Surface_Z1), np.min(Surface_Z2), np.min(Surface_Z3))
    z_max = max(np.max(Surface_Z1), np.max(Surface_Z2), np.max(Surface_Z3))
    ax.set_zlim(z_min - 10, z_max + 10)
    
    plt.tight_layout()
    plt.show()


def plot_vertical_velocity_2d(x_grid, z, W_Vel_3D, y_center_idx, a_coef, D, h0):
    """
    绘制垂直流速的二维图，并叠加波形轮廓
    """
    # 提取Y轴中心的垂直流速切片
    W_Vel_xz = W_Vel_3D[:, y_center_idx, :]
    
    # 计算波形轮廓（用于叠加显示）
    x_center = len(x_grid) // 2
    x_relative = x_grid - x_grid[x_center]  # 相对于波峰中心的位置
    sech2_wave = (1.0 / np.cosh(x_relative / D))**2
    wave_profile = h0 * sech2_wave  # 波形轮廓
    
    # 找到最大垂直流速所在的深度
    max_vel_idx_2d = np.unravel_index(np.argmax(np.abs(W_Vel_xz)), W_Vel_xz.shape)
    max_vel_depth = z[max_vel_idx_2d[1]]  # 第二个维度是深度
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：垂直流速的等值线图
    c1 = ax1.contourf(x_grid/1000, z, W_Vel_xz.T, levels=30, cmap='RdBu_r')
    ax1.contour(x_grid/1000, z, W_Vel_xz.T, levels=15, colors='k', linewidths=0.5, alpha=0.3)
    fig.colorbar(c1, ax=ax1, label='Vertical Velocity (m/s)')
    ax1.set_title("Vertical Velocity Field (x-z plane)")
    ax1.set_xlabel("Propagation Distance x (km)")
    ax1.set_ylabel("Depth z (m)")
    ax1.set_ylim(1000, 0)  # 1000在最下面，0在最上面
    ax1.grid(True, alpha=0.3)
    
    # 右图：垂直流速 + 叠加波形轮廓
    c2 = ax2.contourf(x_grid/1000, z, W_Vel_xz.T, levels=30, cmap='RdBu_r')
    ax2.contour(x_grid/1000, z, W_Vel_xz.T, levels=15, colors='k', linewidths=0.5, alpha=0.3)
    
    # 叠加波形轮廓（在最大流速深度处）
    # 将波形轮廓叠加在最大流速深度附近
    wave_depth = max_vel_depth
    wave_y = wave_depth + wave_profile  # 波形轮廓在深度方向的位置
    
    ax2.plot(x_grid/1000, wave_y, 'k-', linewidth=3, label='Wave Profile', alpha=0.8)
    ax2.fill_between(x_grid/1000, wave_depth, wave_y, alpha=0.3, color='yellow', label='Wave Displacement')
    
    fig.colorbar(c2, ax=ax2, label='Vertical Velocity (m/s)')
    ax2.set_title(f"Vertical Velocity + Wave Profile\nMax velocity at depth {max_vel_depth:.0f}m")
    ax2.set_xlabel("Propagation Distance x (km)")
    ax2.set_ylabel("Depth z (m)")
    ax2.set_ylim(1000, 0)  # 1000在最下面，0在最上面
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_vertical_velocity_3d(x_grid, y_grid, z, W_Vel_3D, W, a_coef, h0, D):
    """
    绘制垂直流速的三维图
    """
    # 找到最大垂直流速所在的深度索引
    max_vel_idx = np.unravel_index(np.argmax(np.abs(W_Vel_3D)), W_Vel_3D.shape)
    max_vel_depth_idx = max_vel_idx[2]
    max_vel_depth = z[max_vel_depth_idx]
    
    # 提取该深度的垂直流速切片
    W_Vel_xy = W_Vel_3D[:, :, max_vel_depth_idx]
    
    # 构建 2D 水平网格
    X_2D, Y_2D = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制垂直流速的3D表面
    surf = ax.plot_surface(X_2D/1000, Y_2D/1000, W_Vel_xy, 
                           cmap='RdBu_r', edgecolor='none', alpha=0.9, antialiased=True)
    
    ax.view_init(elev=35, azim=-60)
    
    ax.set_title(f"3D Vertical Velocity Field\nAt depth {max_vel_depth:.0f}m (max velocity depth)", 
                 fontsize=14, pad=20)
    ax.set_xlabel("Propagation Distance x (km)", labelpad=10)
    ax.set_ylabel("Along-Crest Distance y (km)", labelpad=10)
    ax.set_zlabel("Vertical Velocity (m/s)", labelpad=10)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Vertical Velocity (m/s)', pad=0.1)
    
    plt.tight_layout()
    plt.show()



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

    # 绘图展示第一步的结果
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

    ax1.plot(T, z, 'r')
    ax1.set_title("Temperature (°C)")
    ax1.set_xlabel("Temperature (°C)")
    ax1.set_ylabel("Depth (m)")
    ax1.set_ylim(1000, 0)
    ax1.grid(True)

    ax2.plot(rho, z, 'b')
    ax2.set_title("Density (kg/m^3)")
    ax2.set_xlabel("Density (kg/m³)")
    ax2.set_ylabel("Depth (m)")
    ax2.set_ylim(1000, 0)
    ax2.grid(True)

    ax3.plot(N2, z, 'g')
    ax3.set_title("Buoyancy Frequency N^2 (s^-2)")
    ax3.set_xlabel("N² (s⁻²)")
    ax3.set_ylabel("Depth (m)")
    ax3.set_ylim(1000, 0)
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    # 第二步：计算垂直结构
    print("正在计算第一模态内波垂直结构...")
    W, U, c0 = calculate_vertical_structure(z, N2)
    print(f"计算成功！该次随机生成的环境第一模态波速 c0 约为: {c0:.2f} m/s")

    # 展示结构图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
    ax1.plot(N2, z, 'g'); ax1.set_title("Buoyancy Frequency N^2"); ax1.set_ylim(1000, 0); ax1.grid(True)
    ax2.plot(W, z, 'b', linewidth=2); ax2.axvline(0, color='k', linestyle='--'); ax2.set_title("Vertical Structure W(z)"); ax2.set_ylim(1000, 0); ax2.grid(True)
    ax3.plot(U, z, 'r', linewidth=2); ax3.axvline(0, color='k', linestyle='--'); ax3.set_title("Horizontal Structure U(z) = dW/dz"); ax3.set_ylim(1000, 0); ax3.grid(True)
    plt.tight_layout(); plt.show()

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

    # 第四步：二维切片可视化（包括俯视xy和xz剖面）
    # xy剖面在温跃层核心附近（约150m）
    z_idx = np.argmin(np.abs(z - 150))
    T_xy_slice = T_3D[:, :, z_idx]

    # xz剖面通过y轴中心线
    y_center_idx = len(y_grid) // 2
    T_xz_slice = T_3D[:, y_center_idx, :]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    # 左图：xy俯视
    c1 = ax1.contourf(y_grid/1000, x_grid/1000, T_xy_slice, levels=20, cmap='RdYlBu_r')
    ax1.contour(y_grid/1000, x_grid/1000, T_xy_slice, levels=5, colors='k', linewidths=0.5, alpha=0.5)
    fig.colorbar(c1, ax=ax1, label='Temperature (°C)')
    ax1.set_title("Top-Down View (x-y plane) at Depth ~150m\nCurved Crest Line")
    ax1.set_xlabel("Along-Crest Distance y (km)")
    ax1.set_ylabel("Propagation Distance x (km)")

    # 右图：xz切面，深度从上到下
    X, Z = np.meshgrid(x_grid/1000, z)
    c2 = ax2.contourf(X, Z, T_xz_slice.T, levels=20, cmap='RdYlBu_r')
    ax2.contour(X, Z, T_xz_slice.T, levels=5, colors='k', linewidths=0.5, alpha=0.5)
    fig.colorbar(c2, ax=ax2, label='Temperature (°C)')
    ax2.set_title("XZ Cross Section at y center (km)")
    ax2.set_xlabel("Propagation Distance x (km)")
    ax2.set_ylabel("Depth z (m)")
    ax2.set_ylim(1000, 0)

    plt.tight_layout(); plt.show()

    print("正在生成多个等温面3D可视化...")
    plot_multiple_3d_isotherm_surfaces(x_grid, y_grid, z, W, a_coef, h0, D, Ly, T_3D)
    print("正在生成垂直流速二维图（叠加波形）...")
    plot_vertical_velocity_2d(x_grid, z, W_Vel_3D, y_center_idx, a_coef, D, h0)
    print("正在生成垂直流速三维图...")
    plot_vertical_velocity_3d(x_grid, y_grid, z, W_Vel_3D, W, a_coef, h0, D)

    return run_directory


if __name__ == "__main__":
    run_simulation(save=True)

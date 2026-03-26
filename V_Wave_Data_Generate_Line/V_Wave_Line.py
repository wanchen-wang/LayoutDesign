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
    # 写入单层目录：base_dir 由调用方决定（例如 run_YYYYMMDD_HHMMSS）
    run_dir = base_dir
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


def generate_3d_straight_isw_block(z, T, W, U, c0):
    """
    第三步：生成标准直线波前的三维大尺度方块
    （移除了原先的抛物线几何拟合，回归经典的平面波假设）
    """
    # 1. 设定水平空间网格
    nx, ny = 100, 200  
    x = np.linspace(-5000, 5000, nx) 
    Ly = np.random.uniform(50000, 100000) 
    y = np.linspace(-Ly/2, Ly/2, ny)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # 2. 设定孤立波的物理参数
    amplitude_h0 = np.random.uniform(80, 150) # 最大波幅
    D = np.random.uniform(800, 1500)          # 特征半宽度

    # 3. 计算环境非线性参数 alpha 与 非线性相速 C
    integral_U3 = np.trapezoid(U**3, z)
    integral_U2 = np.trapezoid(U**2, z)
    alpha = (3.0 * c0 / 2.0) * (integral_U3 / integral_U2)
    C_nonlinear = c0 + (alpha * amplitude_h0) / 3.0
    
    # ==========================================
    # 核心修改：去除抛物线曲率，生成标准直线波前
    # ==========================================
    X_effective = X  # 直接使用原始的 X 坐标，没有任何 Y 轴方向的偏移
    a_coef = 0.0     # 将曲率系数设为 0，以保持与旧绘图和保存接口的兼容性
    
    # 4. 计算三维空间剖面
    sech2_x = (1.0 / np.cosh(X_effective / D))**2
    sech2_tanh_x = sech2_x * np.tanh(X_effective / D)
    
    W_3d = W.reshape(1, 1, len(z))
    Displacement_3D = amplitude_h0 * sech2_x * np.abs(W_3d) 
    
    # 5. 生成三维温度场 (使用插值，避免硬编码)
    Effective_Z = Z - Displacement_3D
    Temperature_3D = np.interp(Effective_Z, z, T)
    
    # 6. 生成三维垂直流速场
    Vertical_Velocity_3D = (2.0 * amplitude_h0 * C_nonlinear / D) * sech2_tanh_x * W_3d
    
    return x, y, Temperature_3D, Vertical_Velocity_3D, amplitude_h0, Ly, a_coef, D

# ==========================================
# 主程序执行函数
# ==========================================

def run_simulation(save=True, base_folder="V_Wave_Data_Line"):
    """
    执行单次模拟的主函数，并指定新的基础保存文件夹
    """
    # 1 & 2. 生成背景与垂直结构
    z, T, rho, N2 = generate_background_stratification()
    W, U, c0 = calculate_vertical_structure(z, N2)
    
    # 3. 生成直线波前的三维场
    x_grid, y_grid, T_3D, W_Vel_3D, h0, Ly, a_coef, D = generate_3d_straight_isw_block(z, T, W, U, c0)
    
    # 4. 保存数据
    if save:
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        # 以时间戳创建子文件夹存放单次运行数据
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base_folder, f"run_{timestamp}")
        os.makedirs(run_dir)
        
        # 调用之前写好的保存函数（保持参数不变）
        save_run_data(run_dir, z, T, rho, N2, W, U, c0, x_grid, y_grid, T_3D, W_Vel_3D, h0, Ly, a_coef, D)
        print(f"数据已成功保存至目录: {run_dir}")
        
    return x_grid, y_grid, z, T_3D, W_Vel_3D, W


"""
v_wave.py 绘图模块
提取所有可视化函数和绘图代码
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import json
import sys


def plot_background_stratification(T, rho, N2, z):
    """绘制背景层化剖面"""
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


def plot_vertical_structure(N2, W, U, z):
    """绘制垂直结构"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
    ax1.plot(N2, z, 'g'); ax1.set_title("Buoyancy Frequency N^2"); ax1.set_ylim(1000, 0); ax1.grid(True)
    ax2.plot(W, z, 'b', linewidth=2); ax2.axvline(0, color='k', linestyle='--'); ax2.set_title("Vertical Structure W(z)"); ax2.set_ylim(1000, 0); ax2.grid(True)
    ax3.plot(U, z, 'r', linewidth=2); ax3.axvline(0, color='k', linestyle='--'); ax3.set_title("Horizontal Structure U(z) = dW/dz"); ax3.set_ylim(1000, 0); ax3.grid(True)
    plt.tight_layout(); 
    plt.show()


def plot_2d_slices(x_grid, z, T_3D, y_grid):
    """绘制二维切片（xy 俯视和 xz 剖面）"""
    # xy 剖面在温跃层核心附近（约150m）
    z_idx = np.argmin(np.abs(z - 150))
    T_xy_slice = T_3D[:, :, z_idx]

    # xz 剖面通过 y 轴中心线
    y_center_idx = len(y_grid) // 2
    T_xz_slice = T_3D[:, y_center_idx, :]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    # 左图：xy 俯视
    c1 = ax1.contourf(y_grid/1000, x_grid/1000, T_xy_slice, levels=20, cmap='RdYlBu_r')
    ax1.contour(y_grid/1000, x_grid/1000, T_xy_slice, levels=5, colors='k', linewidths=0.5, alpha=0.5)
    fig.colorbar(c1, ax=ax1, label='Temperature (°C)')
    ax1.set_title("Top-Down View (x-y plane) at Depth ~150m\nCurved Crest Line")
    ax1.set_xlabel("Along-Crest Distance y (km)")
    ax1.set_ylabel("Propagation Distance x (km)")

    # 右图：xz 切面，深度从上到下
    X, Z = np.meshgrid(x_grid/1000, z)
    c2 = ax2.contourf(X, Z, T_xz_slice.T, levels=20, cmap='RdYlBu_r')
    ax2.contour(X, Z, T_xz_slice.T, levels=5, colors='k', linewidths=0.5, alpha=0.5)
    fig.colorbar(c2, ax=ax2, label='Temperature (°C)')
    ax2.set_title("XZ Cross Section at y center (km)")
    ax2.set_xlabel("Propagation Distance x (km)")
    ax2.set_ylabel("Depth z (m)")
    ax2.set_ylim(1000, 0)

    plt.tight_layout()
    plt.show()


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


def list_groups(base_dir="v_wave_data"):
    if not os.path.isdir(base_dir):
        return []
    items = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    items.sort()
    return items


def load_data(data_dir):
    """加载指定数据组的所有数据"""
    print(f"正在加载数据: {data_dir} ...")
    
    z = np.load(os.path.join(data_dir, 'z.npy'))
    x_grid = np.load(os.path.join(data_dir, 'x_grid.npy'))
    y_grid = np.load(os.path.join(data_dir, 'y_grid.npy'))
    W_Vel_3D = np.load(os.path.join(data_dir, 'W_Vel_3D.npy'))
    T_3D = np.load(os.path.join(data_dir, 'T_3D.npy'))
    T_profile = np.load(os.path.join(data_dir, 'T_profile.npy'))
    rho_profile = np.load(os.path.join(data_dir, 'rho_profile.npy'))
    N2_profile = np.load(os.path.join(data_dir, 'N2_profile.npy'))
    U_profile = np.load(os.path.join(data_dir, 'U_profile.npy'))
    W_profile = np.load(os.path.join(data_dir, 'W_profile.npy'))

    with open(os.path.join(data_dir, 'params.json'), 'r') as f:
        params = json.load(f)

    return {
        'z': z,
        'x_grid': x_grid,
        'y_grid': y_grid,
        'W_Vel_3D': W_Vel_3D,
        'T_3D': T_3D,
        'T': T_profile,
        'rho': rho_profile,
        'N2': N2_profile,
        'U': U_profile,
        'W': W_profile,
        'params': params
    }


if __name__ == "__main__":
    groups = list_groups()
    total = len(groups)

    if total == 0:
        print("没有可用的数据组，请先运行 v_wave 生成数据")
        sys.exit(1)

    print("可用的数据组：")
    for i, group in enumerate(groups, 1):
        print(f"{i}: {group}")

    # 选择数据组
    while True:
        try:
            choice = input(f"选择数据组 (1-{total}): ")
            idx = int(choice) - 1
            if 0 <= idx < total:
                break
            else:
                print("无效选择，请重新输入")
        except ValueError:
            print("请输入有效的数字")

    selected_group = groups[idx]
    path = os.path.join("v_wave_data", selected_group)
    print(f"正在加载数据组: {selected_group}")

    try:
        data = load_data(path)
        
        # 调用所有绘图函数
        print("生成背景层化剖面图...")
        plot_background_stratification(data['T'], data['rho'], data['N2'], data['z'])
        
        print("生成垂直结构图...")
        plot_vertical_structure(data['N2'], data['W'], data['U'], data['z'])
        
        print("生成二维切片图...")
        plot_2d_slices(data['x_grid'], data['z'], data['T_3D'], data['y_grid'])
        
        print("生成多等温面3D图...")
        params = data['params']
        plot_multiple_3d_isotherm_surfaces(
            data['x_grid'], data['y_grid'], data['z'], data['W'],
            params.get('a_coef', 0), params.get('h0', 0), 
            params.get('D', 0), params.get('Ly', 0), data['T_3D']
        )
        
        print("生成垂直流速二维图...")
        y_center_idx = len(data['y_grid']) // 2
        plot_vertical_velocity_2d(
            data['x_grid'], data['z'], data['W_Vel_3D'], y_center_idx,
            params.get('a_coef', 0), params.get('D', 0), params.get('h0', 0)
        )
        
        print("生成垂直流速3D图...")
        plot_vertical_velocity_3d(
            data['x_grid'], data['y_grid'], data['z'], data['W_Vel_3D'], data['W'],
            params.get('a_coef', 0), params.get('h0', 0), params.get('D', 0)
        )
        
        print("所有图表生成完成！")
        
    except Exception as e:
        print(f"处理失败: {e}")

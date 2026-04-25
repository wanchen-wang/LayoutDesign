import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# 确保当前目录在路径中，以便导入本地计算模块
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 导入你提供的两个现成计算程序
import Single_W_A_Lagrangian
import Single_W_A_Lagrangian_Hor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V_WAVE_DATA_DIR = PROJECT_ROOT / "ModelA_Virtual_Internal_Solitary_Wave_Data_Generation" / "V_Wave_Data_Hor"

def plot_spatial_comparison(path):
    # 1. 直接调用原计算程序，获取两种情况的结果
    print("[1/3] 正在运行未包含水平流的程序...")
    res_std = Single_W_A_Lagrangian.run_single(path)
    
    print("\n[2/3] 正在运行包含了水平流的程序...")
    res_hor = Single_W_A_Lagrangian_Hor.run_single(path, return_full=True)
    
    # 2. 提取时间序列和深度序列
    t_std, z_std, t_meet_std = res_std['t_array'], res_std['depth_obs'], res_std['t_meet']
    t_hor, z_hor, t_meet_hor = res_hor['t_array'], res_hor['depth_obs'], res_hor['t_meet']
    
    # 3. 重构水平位移 X (物理学积分：X = Σ v * dt)
    v_g = 0.22  # 滑翔机水平静水速度
    
    # --- 轨迹 A：无水平流 (Standard) ---
    dt_std = np.gradient(t_std)
    x_std_raw = np.cumsum(v_g * dt_std)  # 仅考虑滑翔机自身动力
    
    # --- 轨迹 B：有水平流 (Hor-Lagrangian) ---
    # 读取U_Vel_3D的y=0切片进行插值
    z_grid = np.load(os.path.join(path, 'z.npy'))
    x_grid = np.load(os.path.join(path, 'x_grid.npy'))
    y_grid = np.load(os.path.join(path, 'y_grid.npy'))
    U_Vel_3D = np.load(os.path.join(path, 'U_Vel_3D.npy'))
    U_Vel_3D = -U_Vel_3D
    y_center_idx = np.argmin(np.abs(y_grid - 0.0))
    U_Vel_xz = U_Vel_3D[:, y_center_idx, :]
    from scipy.interpolate import RegularGridInterpolator
    interp_u = RegularGridInterpolator((x_grid, z_grid), U_Vel_xz, bounds_error=False, fill_value=0.0)

    # 逐步积分：每步用当前(x, z)插值u，再推进x
    # 需要先读取物理参数获取波速 Cp
    import json
    with open(os.path.join(path, 'params.json'), 'r') as f:
        params = json.load(f)
    Cp = params.get('c0')

    # 逐步积分：每步用当前(x, z)插值u，再推进x
    dt_hor = np.gradient(t_hor)
    x_hor_raw = np.zeros(len(t_hor))
    
    for i in range(1, len(t_hor)):
        # 【核心修复 1】：计算滑翔机相对于波浪核心的“有效坐标 x_eff”
        # 它们在 t_meet_hor 时刻相遇 (此时相对距离为0)，相对速度为 v_g + Cp
        x_eff = (v_g + Cp) * (t_hor[i-1] - t_meet_hor) 
        
        # 使用相对坐标去波浪矩阵中提取水平流速
        u_i = float(interp_u((x_eff, z_hor[i-1])))
        
        # 【核心修复 2】：去掉多余的负号。前面已经翻转过 U_Vel_3D，直接做物理矢量叠加
        x_hor_raw[i] = x_hor_raw[i-1] + (v_g + u_i) * dt_hor[i]

    # 4. 坐标系对齐：将相遇点 (Meet Point) 的水平位置设为原点 0，直观对比偏移量
    meet_idx_std = np.argmin(np.abs(t_std - t_meet_std))
    x_std = x_std_raw - x_std_raw[meet_idx_std]
    
    meet_idx_hor = np.argmin(np.abs(t_hor - t_meet_hor))
    x_hor = x_hor_raw - x_hor_raw[meet_idx_hor]

    print("\n[3/3] 计算完毕，正在绘制空间轨迹对比图...")

    # 5. 绘图
    plt.figure(figsize=(10, 7))
    
    # 绘制无水平流轨迹 (灰色虚线)
    plt.plot(x_std, z_std, color='gray', linestyle='--', linewidth=2.5, 
             label='Standard Trajectory (No Hor-Current)', alpha=0.8)
    
    # 绘制有水平流轨迹 (深红色实线)
    plt.plot(x_hor, z_hor, color='crimson', linestyle='-', linewidth=2.5, 
             label='Actual Trajectory (With Hor-Current)')
    
    # 标注基准点：与波峰相遇的位置
    plt.scatter([0], [z_std[meet_idx_std]], color='black', s=80, marker='X', zorder=5)
    plt.annotate("Peak Encounter Point\n(Aligned at X=0)", 
                 (0, z_std[meet_idx_std]), textcoords="offset points", xytext=(-10, 15), 
                 ha='right', color='black', fontweight='bold')
    
    # 标注最深转折点 (Turning Points)
    turn_idx_std = np.argmax(z_std)
    turn_idx_hor = np.argmax(z_hor)
    plt.scatter(x_std[turn_idx_std], z_std[turn_idx_std], color='gray', s=100, marker='*', zorder=6)
    plt.scatter(x_hor[turn_idx_hor], z_hor[turn_idx_hor], color='crimson', s=100, marker='*', zorder=6)
    
    # 添加带背景框的文本，防止线段遮挡
    bbox_props_std = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    plt.annotate(f"Std Turn\nZ: {z_std[turn_idx_std]:.1f}m\nX: {x_std[turn_idx_std]:.1f}m", 
                 (x_std[turn_idx_std], z_std[turn_idx_std]), 
                 textcoords="offset points", xytext=(-20, -35), ha='center', color='dimgray', bbox=bbox_props_std)
    
    bbox_props_hor = dict(boxstyle="round,pad=0.3", fc="white", ec="crimson", alpha=0.8)
    plt.annotate(f"Actual Turn\nZ: {z_hor[turn_idx_hor]:.1f}m\nX: {x_hor[turn_idx_hor]:.1f}m", 
                 (x_hor[turn_idx_hor], z_hor[turn_idx_hor]), 
                 textcoords="offset points", xytext=(20, -35), ha='center', color='darkred', bbox=bbox_props_hor)

    # 图表装饰
    plt.title('Glider Spatial Trajectory (X vs Z) Deformation by Internal Wave', fontsize=14, fontweight='bold')
    plt.xlabel('Relative Horizontal Displacement X (m)', fontsize=12)
    plt.ylabel('Depth Z (m)', fontsize=12)
    plt.gca().invert_yaxis()  # 海洋学惯例：深度向下
    
    # 添加水平比例尺辅助线
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
    plt.tight_layout()
    plt.show()

def list_groups(base_dir=DEFAULT_V_WAVE_DATA_DIR):
    if not os.path.isdir(base_dir):
        return []
    items = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    items.sort()
    return items

if __name__ == "__main__":
    groups = list_groups()
    if not groups:
        print("未找到数据文件夹。请确保路径与计算程序一致。")
        sys.exit(1)
        
    selected_group = groups[55] # 这里可以修改为你想要测试的组号，或者添加交互式选择
    data_path = os.path.join(DEFAULT_V_WAVE_DATA_DIR, selected_group)
    print(f"[{selected_group}] 启动轨迹对比...")
    
    try:
        plot_spatial_comparison(data_path)
    except Exception as e:
        print(f"绘图失败: {e}")
        import traceback
        traceback.print_exc()
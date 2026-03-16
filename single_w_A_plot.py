"""
single_w_A.py 绘图模块
提取所有可视化函数和绘图代码
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ensure current directory is on path so that we can import local modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from single_w_A import run_single


def plot_anatomy_fig3(t_array, w_obs, depth_obs, t_meet, thermocline_depth):
    """
    复刻 Anatomy Fig 3 绘图：垂直流速时序和滑翔机深度
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 右侧 Y 轴：绘制滑翔机深度 (红线)
    ax2 = ax1.twinx()
    ax2.plot(t_array, depth_obs, color='tomato', linewidth=1.5, label='Glider Depth')
    ax2.set_ylim(1000, 0)  # 翻转深度轴
    ax2.set_ylabel('Depth (m)', color='tomato', fontsize=12)

    # 左侧 Y 轴：绘制垂直水速 (深蓝线)
    # 取消了marker，使用纯净的实线展现完美的无噪声物理波形
    ax1.plot(t_array, w_obs, color='#005b96', linestyle='-', linewidth=2)

    # 阴影填充：复刻 Anatomy 中下沉涂蓝、上升涂粉的效果
    ax1.fill_between(t_array, 0, w_obs, where=(w_obs < 0), color='#6b9ac4', alpha=0.5) 
    ax1.fill_between(t_array, 0, w_obs, where=(w_obs > 0), color='#f4a1c1', alpha=0.5)

    # 标记 t_w0 零点和参考线
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax1.axvline(t_meet, color='black', linestyle=':', linewidth=1.5, alpha=0.6)
    ax1.text(t_meet + 20, max(w_obs)*0.8, f'Peak Encounter\nDepth: {thermocline_depth:.1f}m', color='black')

    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel(r'$w_{isw}$ (m s$^{-1}$)', color='#005b96', fontsize=12)
    ax1.set_title('Reconstruction of Anatomy Fig 3 (Pure KdV Signal at Max Amplitude Depth)', fontsize=14)

    # 动态调整左侧 Y 轴的上下限，使其对称美观
    w_max = max(abs(np.min(w_obs)), abs(np.max(w_obs))) * 1.2
    if w_max > 0:
        ax1.set_ylim(-w_max, w_max)

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()


def plot_lagrangian_sampling(t_array, w_isw_array, w_obs_array, depth_obs, t_integral, w_integral, dh_raw, error_pct):
    """
    可视化拉格朗日采样：水速曲线和滑翔机深度轨迹
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制水速曲线
    ax1 = plt.gca()
    ax1.plot(t_array, w_isw_array, color='#005b96', label='Water Velocity $w_{isw}$', linewidth=2)
    ax1.fill_between(t_integral, 0, w_integral, color='#f4a1c1', alpha=0.6, label=f'Integrated Area (dh_raw={dh_raw:.1f}m)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Water Vertical Velocity (m/s)', color='#005b96')
    ax1.tick_params(axis='y', labelcolor='#005b96')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # 双 Y 轴绘制滑翔机拉格朗日深度轨迹
    ax2 = ax1.twinx()
    ax2.plot(t_array, depth_obs, color='darkorange', label="Glider True Depth $z_g$", linewidth=2, linestyle='--')
    ax2.set_ylabel('Depth (m, Down is positive)', color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.invert_yaxis() # 海洋学习惯：越深 Y 值越大，方向向下
    ax2.legend(loc='upper right')

    plt.title(f'Lagrangian Sampling Diagnosis (Error: {error_pct:.2f}%)')
    plt.tight_layout()
    plt.show()
    if not os.path.isdir(base_dir):
        return []
    items = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    items.sort()
    return items


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
    print(f"正在分析数据组: {selected_group}")

    try:
        result = run_single(path)
        # 调用绘图函数
        plot_anatomy_fig3(
            result['t_array'],
            result['w_obs'],
            result['depth_obs'],
            result['t_meet'],
            result['thermocline_depth']
        )
    except Exception as e:
        print(f"处理失败: {e}")

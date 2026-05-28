"""
Shiyan4.py 绘图模块
可视化实验组4采样：前半段不受拉格朗日影响，后半段受垂直流影响
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from shiyan4 import run_single_shiyan4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V_WAVE_DATA_DIR = PROJECT_ROOT / "ModelA_Virtual_Internal_Solitary_Wave_Data_Generation" / "V_Wave_Data"


def _annotate_point(ax, x, y, text, dx, dy, color='black', fontsize=8):
    """Annotate a point with readable boxed text and arrow."""
    dx = max(-24, min(24, dx))
    dy = max(-24, min(24, dy))
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords='offset points',
        fontsize=fontsize,
        color=color,
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=color, alpha=0.85),
    )


def plot_shiyan4_sampling(t_array, w_isw_array, depth_obs, mode_array, t_meet, thermocline_depth, error_pct):
    """
    可视化实验组4采样：
    - 垂直流速时序（显示波浪的正负瓣，即滑翔机观测到的流速）
    - 滑翔机深度轨迹（显示拉格朗日影响区域）
    - 标注模式切换点（垂直水流由负变正的时刻）
    """
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # 右侧 Y 轴：绘制滑翔机深度
    ax2 = ax1.twinx()
    ax2.plot(t_array, depth_obs, color='tomato', linewidth=2, label='Glider Depth')
    ax2.set_ylim(1000, 0)
    ax2.set_ylabel('Depth (m)', color='tomato', fontsize=12)

    # 左侧 Y 轴：绘制垂直水速（滑翔机观测到的流速）
    ax1.plot(t_array, w_isw_array, color='#005b96', linestyle='-', linewidth=2, label='Observed Velocity $w_{isw}$')

    # 阴影填充：下沉涂蓝、上升涂粉
    ax1.fill_between(t_array, 0, w_isw_array, where=(w_isw_array < 0), color='#6b9ac4', alpha=0.5)
    ax1.fill_between(t_array, 0, w_isw_array, where=(w_isw_array > 0), color='#f4a1c1', alpha=0.5)

    # 标注拉格朗日影响区域
    if len(mode_array) > 0:
        mode_changes = np.diff(mode_array)
        lagrangian_start_idx = np.where(mode_changes == 1)[0]
        
        for start_idx in lagrangian_start_idx:
            start_time = t_array[start_idx + 1]
            ax1.axvline(start_time, color='green', linestyle='--', linewidth=1.5, alpha=0.6)
            ax1.text(start_time + 10, max(w_isw_array)*0.6, 'Enter Lagrangian', color='green', fontsize=10, rotation=90)
        
        # 高亮拉格朗日影响区域背景（后半段）
        if len(lagrangian_start_idx) > 0:
            start = t_array[lagrangian_start_idx[0] + 1]
            ax1.axvspan(start, t_array[-1], color='green', alpha=0.1)

    # 标记参考线
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax1.axvline(t_meet, color='black', linestyle=':', linewidth=1.5, alpha=0.6)
    ax1.text(t_meet + 20, max(w_isw_array)*0.8, f'Peak Encounter\nDepth: {thermocline_depth:.1f}m', color='black')

    # 标注顶点与零线交点
    peak_idx = int(np.argmax(w_isw_array))
    trough_idx = int(np.argmin(w_isw_array))

    ax1.scatter([t_array[peak_idx]], [w_isw_array[peak_idx]], color='red', s=35, zorder=6)
    _annotate_point(
        ax1,
        t_array[peak_idx],
        w_isw_array[peak_idx],
        f"Peak\n({t_array[peak_idx]:.1f}s, {w_isw_array[peak_idx]:.3f}m/s)",
        18,
        18,
        color='red',
    )

    ax1.scatter([t_array[trough_idx]], [w_isw_array[trough_idx]], color='#004b7a', s=35, zorder=6)
    _annotate_point(
        ax1,
        t_array[trough_idx],
        w_isw_array[trough_idx],
        f"Trough\n({t_array[trough_idx]:.1f}s, {w_isw_array[trough_idx]:.3f}m/s)",
        18,
        -40,
        color='#004b7a',
    )

    meet_idx = int(np.argmin(np.abs(t_array - t_meet)))
    ax1.scatter([t_array[meet_idx]], [w_isw_array[meet_idx]], color='purple', s=30, zorder=6)
    _annotate_point(
        ax1,
        t_array[meet_idx],
        w_isw_array[meet_idx],
        f"Meet\n({t_array[meet_idx]:.1f}s, {w_isw_array[meet_idx]:.3f}m/s)",
        18,
        -40,
        color='purple',
    )

    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel(r'$w$ (m s$^{-1}$)', color='#005b96', fontsize=12)
    ax1.set_title(f'Shiyan4: Post-Lagrangian Sampling (Vertical Only) (Error: {error_pct:.2f}%)', fontsize=14)

    w_max = max(abs(np.min(w_isw_array)), abs(np.max(w_isw_array))) * 1.2
    if w_max > 0:
        ax1.set_ylim(-w_max, w_max)

    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def list_groups(base_dir=DEFAULT_V_WAVE_DATA_DIR):
    if not os.path.isdir(base_dir):
        return []
    items = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    items.sort()
    return items


if __name__ == "__main__":
    base_data_dir = DEFAULT_V_WAVE_DATA_DIR
    if not os.path.isdir(base_data_dir):
        print("没有可用的数据组，请先运行 v_wave 生成数据")
        sys.exit(1)

    groups = list_groups(base_data_dir)
    total = len(groups)

    if total == 0:
        print("没有可用的数据组，请先运行 v_wave 生成数据")
        sys.exit(1)

    print("可用的数据组：")
    for i, group in enumerate(groups, 1):
        print(f"{i}: {group}")

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
    path = os.path.join(base_data_dir, selected_group)
    print(f"正在分析数据组: {selected_group}")

    try:
        result = run_single_shiyan4(path, return_full=True)

        print("\n绘制实验组4采样图...")
        plot_shiyan4_sampling(
            result['t_array'],
            result['w_isw_array'],
            result['depth_obs'],
            result['mode_array'],
            result['t_meet'],
            result['thermocline_depth'],
            result['error_pct']
        )

        print(f"\n✅ 处理完成! 波高估计误差: {result['error_pct']:.2f}%")

    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()

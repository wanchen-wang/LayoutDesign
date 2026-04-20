"""
Model5_Upwelling_Static_Sampling 绘图模块
基于 Single_W_A_Plot.py 框架，展示上涌检测后保持深度采样的特性
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# ensure current directory is on path so that we can import local modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from Model6_Upwelling_Static_Sampling import run_single

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V_WAVE_DATA_DIR = PROJECT_ROOT / "ModelA_Virtual_Internal_Solitary_Wave_Data_Generation" / "V_Wave_Data"


def _annotate_point(ax, x, y, text, dx, dy, color='black', fontsize=8):
    """Annotate a point with readable boxed text and arrow."""
    # Keep labels close to the point for better readability.
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


def plot_upwelling_static_sampling(t_array, w_obs, depth_obs, t_meet, thermocline_depth, 
                                   upwelling_detected, upwelling_depth, error_pct, group_name):
    """
    绘制 Model5 上涌静态采样：垂直流速和滑翔机深度轨迹
    核心特点：显示上涌检测点和深度保持阶段
    """
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # 左侧 Y 轴：绘制垂直水速（蓝线）
    ax1.plot(t_array, w_obs, color='#005b96', linestyle='-', linewidth=2.0, label='Water Velocity $w_{isw}$')

    # 阴影填充：下沉（蓝）、上升（粉）
    ax1.fill_between(t_array, 0, w_obs, where=(w_obs < 0), color='#6b9ac4', alpha=0.45, label='Downwelling') 
    ax1.fill_between(t_array, 0, w_obs, where=(w_obs > 0), color='#f4a1c1', alpha=0.45, label='Upwelling')

    # 标记零线和峰值线
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    
    # 标注相遇时刻
    ax1.axvline(t_meet, color='black', linestyle=':', linewidth=1.5, alpha=0.6)
    ax1.text(t_meet + 20, max(w_obs)*0.7, f'Peak Encounter\nDepth: {thermocline_depth:.1f}m', 
             color='black', fontsize=9)

    # 标注峰值
    peak_idx = int(np.argmax(w_obs))
    ax1.scatter([t_array[peak_idx]], [w_obs[peak_idx]], color='red', s=50, zorder=6, marker='*')
    _annotate_point(
        ax1,
        t_array[peak_idx],
        w_obs[peak_idx],
        f"Peak\n({t_array[peak_idx]:.1f}s, {w_obs[peak_idx]:.4f}m/s)",
        18,
        18,
        color='red',
        fontsize=8
    )

    # 标注谷值
    trough_idx = int(np.argmin(w_obs))
    ax1.scatter([t_array[trough_idx]], [w_obs[trough_idx]], color='#004b7a', s=50, zorder=6, marker='v')
    _annotate_point(
        ax1,
        t_array[trough_idx],
        w_obs[trough_idx],
        f"Trough\n({t_array[trough_idx]:.1f}s, {w_obs[trough_idx]:.4f}m/s)",
        18,
        -45,
        color='#004b7a',
        fontsize=8
    )

    # 标注零线交点
    left_zero = peak_idx
    while left_zero > 0 and w_obs[left_zero] > 0:
        left_zero -= 1
    right_zero = peak_idx
    while right_zero < len(w_obs) - 1 and w_obs[right_zero] > 0:
        right_zero += 1

    ax1.scatter(
        [t_array[left_zero], t_array[right_zero]],
        [w_obs[left_zero], w_obs[right_zero]],
        color='black',
        s=28,
        zorder=6,
    )
    _annotate_point(ax1, t_array[left_zero], w_obs[left_zero], 
                    f"Zero L\n({t_array[left_zero]:.1f}s)", -95, 20, fontsize=7)
    _annotate_point(ax1, t_array[right_zero], w_obs[right_zero], 
                    f"Zero R\n({t_array[right_zero]:.1f}s)", 24, 20, fontsize=7)

    ax1.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel(r'$w_{isw}$ (m s$^{-1}$)', color='#005b96', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#005b96')
    ax1.grid(True, linestyle=':', alpha=0.5)

    # 右侧 Y 轴：绘制滑翔机深度（橙线）
    ax2 = ax1.twinx()
    ax2.plot(t_array, depth_obs, color='darkorange', linewidth=2.0, linestyle='--', 
             label='Glider Depth $z_g$')
    ax2.set_ylabel('Depth (m, Down is positive)', color='darkorange', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.invert_yaxis()  # 海洋学习惯：越深 Y 值越大，方向向下

    # 【关键改动】标注上涌检测点
    if upwelling_detected and upwelling_depth is not None:
        # 找到上涌检测时刻对应的时间
        upwelling_idx = None
        prev_w = None
        for i, w in enumerate(w_obs):
            if prev_w is not None and prev_w <= 0 and w > 0:
                upwelling_idx = i
                break
            prev_w = w
        
        if upwelling_idx is not None:
            ax2.scatter([t_array[upwelling_idx]], [depth_obs[upwelling_idx]], 
                       color='green', s=100, zorder=7, marker='D', label='Upwelling Detection')
            _annotate_point(ax2, t_array[upwelling_idx], depth_obs[upwelling_idx],
                           f"Upwelling Trigger\nTime: {t_array[upwelling_idx]:.1f}s\nDepth: {upwelling_depth:.1f}m",
                           -120, -50, color='green', fontsize=8)
            
            # 绘制上涌后深度保持的水平线
            ax2.axhline(upwelling_depth, color='green', linestyle='-.', linewidth=1.5, 
                        alpha=0.6, label='Static Sampling Depth')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    # 设置标题
    title = (
        f"Model5 Upwelling-Triggered Static Sampling\n"
        f"Group: {group_name} | "
        f"Computed dh={0:.2f}m | Error={error_pct:.2f}%\n"
        f"Upwelling Detected: {'Yes' if upwelling_detected else 'No'}"
    )
    ax1.set_title(title, fontsize=13, fontweight='bold')

    # 动态调整左侧 Y 轴的对称性
    w_max = max(abs(np.min(w_obs)), abs(np.max(w_obs))) * 1.2
    if w_max > 0:
        ax1.set_ylim(-w_max, w_max)

    plt.tight_layout()
    plt.show()


def list_groups(base_dir=DEFAULT_V_WAVE_DATA_DIR):
    """列出所有可用的数据组（目录）"""
    if not os.path.isdir(base_dir):
        return []
    items = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    items.sort()
    return items


if __name__ == "__main__":
    base_dir = DEFAULT_V_WAVE_DATA_DIR
    groups = list_groups(base_dir)
    total = len(groups)

    if total == 0:
        print("没有可用的数据组，请先运行 v_wave 生成数据")
        sys.exit(1)

    print("="*70)
    print("Model5 Upwelling Static Sampling - 绘图工具")
    print("="*70)
    print(f"发现 {total} 组可用数据\n")
    print("可用的数据组：")
    for i, group in enumerate(groups[:20], 1):  # 只显示前20个
        print(f"  {i:3d}: {group}")
    if total > 20:
        print(f"  ... 还有 {total-20} 组数据")

    # 选择数据组
    while True:
        try:
            choice = input(f"\n请选择数据组 (1-{total}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < total:
                break
            else:
                print(f"无效选择，请输入 1-{total} 之间的数字")
        except ValueError:
            print("请输入有效的整数")

    selected_group = groups[idx]
    path = os.path.join(base_dir, selected_group)
    print(f"\n正在分析数据组: {selected_group}")

    try:
        result = run_single(path)
        
        # 调用绘图函数
        plot_upwelling_static_sampling(
            result['t_array'],
            result['w_obs'],
            result['depth_obs'],
            result['t_meet'],
            result['thermocline_depth'],
            result['upwelling_detected'],
            result['upwelling_depth'],
            result['error_pct'],
            selected_group
        )
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()

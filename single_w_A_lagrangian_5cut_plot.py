"""
Plotting module for single_w_A_lagrangian_5cut analysis results.

Reads CSV results and generates visualizations for the 5% threshold Lagrangian method.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# ensure current directory is on path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from single_w_A_lagrangian_5cut import run_single_w_A_lagrange_threshold


def plot_single_result(data_dir, save_path=None):
    """
    Generate and display plot for a single analysis result.
    
    Parameters
    ----------
    data_dir : str
        Path to the data directory to analyze
    save_path : str, optional
        Path to save the figure. If None, displays interactively.
    """
    # Run analysis to get full data including arrays
    result = run_single_w_A_lagrange_threshold(data_dir)
    
    # Extract data
    t_array = result['t_array']
    w_isw_array = result['w_isw_array']
    depth_obs = result['depth_obs']
    error_pct = result['error_pct']
    t_w0 = result['t_w0']
    t_U = result['t_U']
    dh_raw = result['dh_raw']
    w_threshold = 0.05 * np.max(w_isw_array)  # 5% threshold
    
    # Find integral region
    mask = (t_array > t_w0) & (t_array < t_U)
    t_integral = t_array[mask]
    w_integral = w_isw_array[mask]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    ax1 = plt.gca()
    ax1.plot(t_array, w_isw_array, color='#005b96', label='Water Velocity $w_{isw}$', linewidth=2)
    # Plot 5% threshold reference line
    ax1.axhline(w_threshold, color='red', linestyle=':', label='5% Threshold Cutoff')
    ax1.fill_between(t_integral, w_threshold, w_integral, color='#f4a1c1', alpha=0.6, 
                     label=f'Integrated Area (dh_raw={dh_raw:.1f}m)')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Water Vertical Velocity (m/s)', color='#005b96')
    ax1.tick_params(axis='y', labelcolor='#005b96')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(t_array, depth_obs, color='darkorange', label="Glider True Depth $z_g$", 
             linewidth=2, linestyle='--')
    ax2.set_ylabel('Depth (m, Down is positive)', color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.invert_yaxis() 
    ax2.legend(loc='upper right')

    plt.title(f'Lagrangian Sampling with 5% Threshold (Error: {error_pct:.2f}%)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_from_csv(csv_file, base_dir="v_wave_data", output_dir="plots"):
    """
    Generate plots for all groups in the CSV results file.
    
    Parameters
    ----------
    csv_file : str
        Path to the CSV results file
    base_dir : str
        Base directory containing data groups
    output_dir : str
        Directory to save plots
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read CSV
    df = pd.read_csv(csv_file)
    
    print(f"从 {csv_file} 读取 {len(df)} 条结果")
    print(f"生成对应的图表...\n")
    
    for idx, row in df.iterrows():
        group = row['group']
        data_path = os.path.join(base_dir, group)
        
        if not os.path.isdir(data_path):
            print(f"警告: 数据目录不存在 {data_path}，跳过")
            continue
        
        print(f"[{idx+1}/{len(df)}] 生成图表: {group}")
        
        try:
            plot_file = os.path.join(output_dir, f"{group}_5cut.png")
            plot_single_result(data_path, save_path=plot_file)
        except Exception as e:
            print(f"  错误: {e}")
    
    print(f"\n所有图表已生成，保存在: {output_dir}")


if __name__ == "__main__":
    import numpy as np
    
    if len(sys.argv) > 1:
        # 如果提供了CSV文件路径，则批量生成图表
        csv_file = sys.argv[1]
        base_dir = sys.argv[2] if len(sys.argv) > 2 else "v_wave_data"
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "plots_5cut"
        plot_from_csv(csv_file, base_dir, output_dir)
    else:
        # 否则绘制第一个数据组
        base = "v_wave_data"
        if os.path.isdir(base):
            dirs = sorted(os.listdir(base))
            if dirs:
                data_path = os.path.join(base, dirs[0])
                print(f"绘制示例: {dirs[0]}")
                plot_single_result(data_path)
            else:
                print("未找到数据组")
        else:
            print(f"数据目录 {base} 不存在")

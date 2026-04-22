import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V_WAVE_DATA_DIR = PROJECT_ROOT / "ModelA_Virtual_Internal_Solitary_Wave_Data_Generation" / "V_Wave_Data"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "Analysis_Results_SwA_Lagrangian_Hor_Cut_Data"


def run_single(data_dir):
    """Execute a single virtual glider sampling using the data folder provided,
    considering horizontal current (U_profile) in Lagrangian stepping.
    """
    z = np.load(os.path.join(data_dir, 'z.npy'))
    x_grid = np.load(os.path.join(data_dir, 'x_grid.npy'))
    y_grid = np.load(os.path.join(data_dir, 'y_grid.npy'))
    W_Vel_3D = np.load(os.path.join(data_dir, 'W_Vel_3D.npy'))
    W_profile = np.load(os.path.join(data_dir, 'W_profile.npy'))
    U_profile = np.load(os.path.join(data_dir, 'U_profile.npy'))

    with open(os.path.join(data_dir, 'params.json'), 'r') as f:
        params = json.load(f)

    Cp = params.get('c0')
    thermocline_depth = params.get('thermocline_depth')
    true_h0 = params.get('h0')
    D = params.get('D', 1000.0)

    # prepare interpolator
    if z[0] > z[-1]:
        z = np.flip(z)
        W_Vel_3D = np.flip(W_Vel_3D, axis=2)
        W_profile = np.flip(W_profile)
        U_profile = np.flip(U_profile)
        
    interp_w = RegularGridInterpolator((x_grid, y_grid, z), W_Vel_3D,
                                       bounds_error=False, fill_value=0.0)

    # 提取滑翔机水平静水速度
    v_g = 0.22

    # 动态计算自适应时间窗口 (迎头相遇)
    u_bg_meet = float(np.interp(thermocline_depth, z, U_profile))
    V_rel = Cp + v_g + u_bg_meet  

    # 迎头相遇倒推初始位置
    t_meet = thermocline_depth * (6000.0 / 1000.0)  
    dt = 5.0  # 采样步长
    
    # 预积分：加入对水平流考虑，从 t=0 积分到 t_meet，获取相遇时的实际水平位移
    x_g_meet_true = 0.0
    z_g_temp = 0.0
    t_temp_array = np.arange(0, t_meet, dt)
    for t_tmp in t_temp_array:
        t_mod = t_tmp % 12000
        w_stdy = -1000.0 / 6000.0 if t_mod < 6000 else 1000.0 / 6000.0
        u_bg = float(np.interp(z_g_temp, z, U_profile))
        z_g_temp = z_g_temp - w_stdy * dt
        z_g_temp = np.clip(z_g_temp, 0.0, 1000.0)
        x_g_meet_true += (v_g + u_bg) * dt

    x_init = x_g_meet_true + Cp * t_meet  
    y_center = 0.0 

    # KdV波形在距离核心 5D 处能量基本衰减殆尽
    half_window_time = max(4000.0, (5.0 * D) / V_rel)
    start_time = max(0, t_meet - half_window_time)
    end_time = t_meet + half_window_time

    # 为保证轨迹与坐标连续，直接从 t=0 积分到 end_time，获取准确的位置状态
    t_full_array = np.arange(0, end_time, dt)
    w_isw_full = np.zeros_like(t_full_array, dtype=float)
    w_obs_full = np.zeros_like(t_full_array, dtype=float)
    depth_full = np.zeros_like(t_full_array, dtype=float)
    x_g_full = np.zeros_like(t_full_array, dtype=float)

    z_g = 0.0
    x_g = 0.0

    for i, t in enumerate(t_full_array):
        # 引擎静水理论速度 (w_stdy)
        t_mod = t % 12000
        w_stdy = -1000.0 / 6000.0 if t_mod < 6000 else 1000.0 / 6000.0

        # 当前深度下的背景水平流速
        u_bg = float(np.interp(z_g, z, U_profile))

        # 计算波浪核心当前位置
        X_wave_current = x_init - Cp * t 
        x_eff = x_g - X_wave_current 
        
        # 结合当前被水流推移过的真实深度 z_g，提取环境垂直流速
        w_isw = interp_w((x_eff, y_center, z_g))
        
        # 物理速度叠加
        w_obs_real = w_stdy + w_isw
        u_obs_real = v_g + u_bg
        
        # 记录当前帧数据
        w_isw_full[i] = w_isw
        w_obs_full[i] = w_obs_real
        depth_full[i] = z_g
        x_g_full[i] = x_g
        
        # 状态步进 
        z_g = z_g - w_obs_real * dt
        z_g = np.clip(z_g, 0.0, 1000.0)
        
        # 水平步进
        x_g = x_g + u_obs_real * dt

    # 截取 start_time 到 end_time 的窗口
    mask = (t_full_array >= start_time) & (t_full_array <= end_time)
    t_array = t_full_array[mask]
    w_isw_array = w_isw_full[mask]
    w_obs_array = w_obs_full[mask]
    depth_obs = depth_full[mask]

    # find integration window
    mask_win = (t_array > start_time) & (t_array < end_time)
    t_win = t_array[mask_win]
    w_win = w_isw_array[mask_win]
    
    if len(w_win) == 0:
        raise ValueError(f"No valid integration window found for {data_dir}")

    max_w_idx = np.argmax(w_win)
    tw0_idx = max_w_idx
    while tw0_idx > 0 and w_win[tw0_idx] > 0:
        tw0_idx -= 1
    t_w0 = t_win[tw0_idx]
    
    tu_idx = max_w_idx
    while tu_idx < len(w_win) - 1 and w_win[tu_idx] > 0:
        tu_idx += 1
    t_U = t_win[tu_idx]
    
    t_integral = t_win[tw0_idx:tu_idx]
    w_integral = w_win[tw0_idx:tu_idx]
    dh_raw = np.trapezoid(w_integral, x=t_integral)

    # 多普勒物理修正与误差计算
    z_idx = np.argmin(np.abs(z - thermocline_depth))
    W_z_meet = W_profile[z_idx]

    doppler_factor = V_rel / Cp
    h0_corrected = dh_raw * doppler_factor / W_z_meet
    
    error_abs = abs(h0_corrected - true_h0)
    error_pct = error_abs / true_h0 * 100
    duration = t_U - t_w0
    error_density = error_pct / duration if duration > 0 else 0.0

    return {
        'wave_id': os.path.basename(data_dir),
        't_w0': t_w0,
        't_U': t_U,
        'duration': duration,
        'dh_raw': dh_raw,
        'dh': h0_corrected,
        'true_h0': true_h0,
        'abs_error': error_abs,
        'error_pct': error_pct,
        'error_density': error_density,
        't_array': t_array,
        'w_isw_array': w_isw_array,
        'w_obs_array': w_obs_array,
        'depth_obs': depth_obs,
        't_meet': t_meet,
        'thermocline_depth': thermocline_depth
    }


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"🚀 启动带水平流(U_profile)的拉格朗日仿真与误差结算...")
    print(f"{'='*60}")

    base_data_dir = DEFAULT_V_WAVE_DATA_DIR
    wave_folders = [f for f in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, f))]
    print(f"[*] 共发现 {len(wave_folders)} 组内孤立波测试样本。\n")

    results_list = []
    for i, folder_name in enumerate(wave_folders, 1):
        data_dir = os.path.join(base_data_dir, folder_name)
        res = run_single(data_dir)
        results_list.append(res)
        print(f"[{i}/{len(wave_folders)}] ✓ {folder_name} 处理成功 (Error: {res['error_pct']:.2f}%)")

    if results_list:
        df = pd.DataFrame(results_list)
        os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
        output_filename = DEFAULT_RESULTS_DIR / "analysis_results_swA_lagrangian_hor_cut.csv"
        df.to_csv(output_filename, index=False)
        print(f"\n✅ 成功保存: {output_filename} (包含 {len(df)} 组观测数据，平均误差: {df['error_pct'].mean():.2f}%)")
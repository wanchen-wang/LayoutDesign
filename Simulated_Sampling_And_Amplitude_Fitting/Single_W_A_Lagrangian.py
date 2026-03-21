import os
import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def run_single(data_dir):
    """Execute a single virtual glider sampling using the data folder provided.

    Parameters
    ----------
    data_dir : str
        Path to one of the timestamped subdirectories under ``V_Wave_Data``.

    Returns
    -------
    dict
        Results including inferred amplitude ``dh`` and error percentage.
    """
    print(f"正在从相对路径加载数据: {data_dir} ...")
    z = np.load(os.path.join(data_dir, 'z.npy'))
    x_grid = np.load(os.path.join(data_dir, 'x_grid.npy'))
    y_grid = np.load(os.path.join(data_dir, 'y_grid.npy'))
    W_Vel_3D = np.load(os.path.join(data_dir, 'W_Vel_3D.npy'))
    W_profile = np.load(os.path.join(data_dir, 'W_profile.npy'))

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
    interp_w = RegularGridInterpolator((x_grid, y_grid, z), W_Vel_3D,
                                       bounds_error=False, fill_value=0.0)

    # 提取滑翔机水平速度
    v_g = 0.22

    # 动态计算自适应时间窗口 (迎头相遇)
    V_rel = Cp + v_g  

    # 迎头相遇倒推初始位置
    t_meet = thermocline_depth * (6000.0 / 1000.0)  
    x_g_meet = v_g * t_meet
    x_init = x_g_meet + Cp * t_meet  
    y_center = 0.0 

    # KdV波形在距离核心 5D 处能量基本衰减殆尽
    half_window_time = max(4000.0, (5.0 * D) / V_rel)
    start_time = max(0, t_meet - half_window_time)
    end_time = t_meet + half_window_time
    dt = 5.0  # 采样步长

    print(f"自适应时间窗口已设定: 半窗长 {half_window_time:.1f} 秒 (依据 5D={5*D:.1f}m, V_rel={V_rel:.2f}m/s)")
    
    t_array = np.arange(start_time, end_time, dt) 
    w_isw_array = np.zeros_like(t_array, dtype=float)
    w_obs_array = np.zeros_like(t_array, dtype=float)
    depth_obs = np.zeros_like(t_array, dtype=float)

    # 初始化滑翔机在 start_time 的自然深度
    t_mod_start = start_time % 12000
    if t_mod_start < 6000:
        z_g = t_mod_start * 1000.0 / 6000.0
    else:
        z_g = 1000.0 - (t_mod_start - 6000.0) * 1000.0 / 6000.0

    print("开始执行拉格朗日随波逐流深度迭代...")
    for i, t in enumerate(t_array):
        # 引擎静水理论速度 (w_stdy)
        t_mod = t % 12000
        w_stdy = -1000.0 / 6000.0 if t_mod < 6000 else 1000.0 / 6000.0

        # 计算波浪与滑翔机的相对坐标
        x_g = v_g * t
        X_wave_current = x_init - Cp * t 
        x_eff = x_g - X_wave_current 
        
        # 结合当前被水流推移过的真实深度 z_g，提取环境流速
        w_isw = interp_w((x_eff, y_center, z_g))
        
        # 物理速度叠加
        w_obs_real = w_stdy + w_isw
        
        # 记录当前帧数据
        w_isw_array[i] = w_isw
        w_obs_array[i] = w_obs_real
        depth_obs[i] = z_g
        
        # 状态步进 
        z_g = z_g - w_obs_real * dt
        z_g = np.clip(z_g, 0.0, 1000.0)

    # find integration window
    mask = (t_array > start_time) & (t_array < end_time)
    t_win = t_array[mask]
    w_win = w_isw_array[mask]
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

    print(f"积分区间锁定: 从 {t_w0:.1f} s 到 {t_U:.1f} s")
    print(f"提取的有效向上水流数据点数: {len(w_integral)} 个")
    print("\n================ 最终观测评价报告 ==================")
    print(f"【真实基准值】 Ground Truth 最大振幅 h0 = {true_h0:.2f} m")
    print(f"【修正推导值】 Inferred     最大振幅 dh = {h0_corrected:.2f} m")
    print(f"【测量绝对误差】 Error = {np.abs(h0_corrected - true_h0):.2f} m ({error_pct:.2f}%)")
    print("====================================================")

    return {
        'dh': h0_corrected,
        'true_h0': true_h0,
        'error_pct': error_pct,
        'params': params,
        't_w0': t_w0,
        't_U': t_U,
        'dh_raw': dh_raw,
        'doppler_factor': doppler_factor,
        'W_z_meet': W_z_meet,
        't_array': t_array,
        'w_isw_array': w_isw_array,
        'w_obs_array': w_obs_array,
        'depth_obs': depth_obs,
        't_meet': t_meet,
        'thermocline_depth': thermocline_depth,
    }


if __name__ == "__main__":
    # simple demo using first directory
    base = "V_Wave_Data"
    if os.path.isdir(base):
        dirs = sorted(os.listdir(base))
        if dirs:
            path = os.path.join(base, dirs[0])
            run_single(path)

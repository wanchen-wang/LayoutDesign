import os
import json
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

_GLOBAL_DATA_CACHE = {}
MAX_CACHE_SIZE = 4

def process_18cut(w_c_threshold, V_target, zeta_target, f_s, run_data_dir, return_diagnostic_data=False):
    """
    流场读取插值 + 动态拉格朗日仿真采样 + 18%截断误差评估
    :param w_c_threshold: 触发高频机动采样模式的海水垂直流速阈值 (m/s)
    :param V_target: 内孤立波区内的目标静水滑翔速度 (m/s)
    :param zeta_target: 目标滑翔轨迹角 (度，下潜为负)
    :param f_s: 目标采样频率 (Hz)
    :param run_data_dir: V_Wave_Data_Hor 生成的具体时间戳文件夹路径 (如 .../20260422_150147)
    :param return_diagnostic_data: 是否返回完整诊断数据（用于绘图）。False=轻量级（优化用），True=完整（绘图用）
    :return:
        - False: 字典，仅含误差指标
        - True: 字典，含完整诊断数据（df, 索引, 参数等）
    """
    # =====================================================================
    # 1. 基于 V_Wave_Data_Hor 结构的真实流场数据读取与插值器构建
    # =====================================================================
    if run_data_dir not in _GLOBAL_DATA_CACHE:
        
        # 【防爆内存】如果缓存满了，丢弃最老的一个文件夹数据
        if len(_GLOBAL_DATA_CACHE) >= MAX_CACHE_SIZE:
            oldest_key = next(iter(_GLOBAL_DATA_CACHE))
            _GLOBAL_DATA_CACHE.pop(oldest_key)
            
        # 从硬盘读取该文件夹专属的参数与三维矩阵
        with open(os.path.join(run_data_dir, 'params.json'), 'r') as f:
            params = json.load(f)
        
        x_grid = np.load(os.path.join(run_data_dir, 'x_grid.npy'))
        y_grid = np.load(os.path.join(run_data_dir, 'y_grid.npy'))
        z_grid = np.load(os.path.join(run_data_dir, 'z.npy'))
        W_Vel_3D = np.load(os.path.join(run_data_dir, 'W_Vel_3D.npy'))
        U_Vel_3D = -np.load(os.path.join(run_data_dir, 'U_Vel_3D.npy')) # 反转流向
        W_profile = np.load(os.path.join(run_data_dir, 'W_profile.npy'))
        
        if z_grid[0] > z_grid[-1]:
            z_grid = np.flip(z_grid)
            W_Vel_3D = np.flip(W_Vel_3D, axis=2)
            U_Vel_3D = np.flip(U_Vel_3D, axis=2)
            W_profile = np.flip(W_profile)
            
        # 为这组特定数据建立专属插值器
        interpolator_w = RegularGridInterpolator((x_grid, y_grid, z_grid), W_Vel_3D, bounds_error=False, fill_value=0.0)
        y_center_idx = np.argmin(np.abs(y_grid - 0.0))
        interpolator_u = RegularGridInterpolator((x_grid, z_grid), U_Vel_3D[:, y_center_idx, :], bounds_error=False, fill_value=0.0)
        
        # 存入缓存
        _GLOBAL_DATA_CACHE[run_data_dir] = (params, z_grid, W_profile, interpolator_w, interpolator_u)

    # 2. 从内存秒读这组数据的专属插值器
    params, z_grid, W_profile, interpolator_w, interpolator_u = _GLOBAL_DATA_CACHE[run_data_dir]
    
    # 提取物理参数
    Delta_Z_true = params.get('h0', 105.0)
    Cp = params.get('c0')
    thermocline_depth = params.get('thermocline_depth')
    D = params.get('D', 1000.0)

    # 3. 极限优化的元组传参（消灭列表重建开销）
    def get_flow(X, Z):
        # 注意：这里务必是圆括号 () 而不是方括号 []
        w_c = interpolator_w((X, 0.0, Z)) 
        u_bg = interpolator_u((X, Z))
        return float(w_c), float(u_bg)

    # =====================================================================
    # 2. 拉格朗日前向仿真与动态采样初始化
    # =====================================================================
    dt = 0.1

    v_g = 0.22

    # 计算相对速度（迎头相遇，考虑水平流）
    u_bg_meet = float(interpolator_u((0.0, thermocline_depth)))
    V_rel = Cp + v_g + u_bg_meet

    t_meet = thermocline_depth * (6000.0 / 1000.0)

    # 预积分：加入对水平流考虑，从 t=0 积分到 t_meet，获取相遇时的实际水平位移
    x_g_meet_true = 0.0
    z_g_temp = 0.0
    t_temp_array = np.arange(0, t_meet, dt)
    for t_tmp in t_temp_array:
        t_mod = t_tmp % 12000
        w_stdy = -1000.0 / 6000.0 if t_mod < 6000 else 1000.0 / 6000.0

        x_eff_est = (v_g + Cp) * (t_tmp - t_meet)
        u_bg = float(interpolator_u((x_eff_est, z_g_temp)))

        z_g_temp = z_g_temp - w_stdy * dt
        z_g_temp = np.clip(z_g_temp, 0.0, 1000.0)
        x_g_meet_true += (v_g + u_bg) * dt

    x_init = x_g_meet_true + Cp * t_meet

    # 自适应时间窗口
    half_window_time = max(4000.0, (5.0 * D) / V_rel)
    start_time = max(0.0, t_meet - half_window_time)
    end_time = t_meet + half_window_time

    # 在 start_time 时刻，根据锯齿深度剖面反算初始深度
    t_mod_start = start_time % 12000.0
    if t_mod_start < 6000.0:
        Z = t_mod_start * 1000.0 / 6000.0
    else:
        Z = 1000.0 - (t_mod_start - 6000.0) * 1000.0 / 6000.0
    Z = float(np.clip(Z, z_grid[0], z_grid[-1]))
    t = start_time

    x_g = 0.0

    # 常规巡航参数
    w_stdy_norm = 1000.0 / 6000.0
    V_norm = float(np.hypot(v_g, w_stdy_norm))
    zeta_norm = float(np.degrees(np.arcsin(-w_stdy_norm / V_norm)))
    f_norm = 0.2

    sampled_data = []
    time_since_sample = 0.0

    # =====================================================================
    # 3. 核心时空演化循环：基于流速触发机制的动态采样 + 拉格朗日速度叠加
    # =====================================================================
    t = start_time
    x_g = v_g * start_time  # ✅ 正确初始化起始时刻的绝对坐标
    while t < end_time:
        X_wave_current = x_init - Cp * t
        x_eff = x_g - X_wave_current

        w_c, u_bg = get_flow(x_eff, Z)

        # 机制一：状态触发机制
        if w_c >= w_c_threshold:
            current_V = V_target
            current_zeta = zeta_target
            current_fs = f_s
        else:
            current_V = V_norm
            current_zeta = zeta_norm
            current_fs = f_norm

        # 机制二：拉格朗日速度叠加（水平和垂直）
        zeta_rad = np.radians(current_zeta)
        w_g = -current_V * np.sin(zeta_rad)

        # W_Vel_3D 中 w_c 为 z 轴向上正方向（正值=上涌）
        w_abs = w_g - w_c
        x_abs = v_g + u_bg

        # 机制三：真实时空演化与离散采样
        Z += w_abs * dt
        Z = float(np.clip(Z, 0.0, 1000.0))
        x_g += x_abs * dt
        t += dt

        interval = 1.0 / current_fs
        time_since_sample += dt
        if time_since_sample >= interval:
            sampled_data.append({'Time': t, 'Z': Z, 'w_c': w_c, 'f_s': current_fs, 'u_bg': u_bg})
            time_since_sample -= interval

    # ── 18% 截断评估 ──────────────────────────────────────────────────────────
    df = pd.DataFrame(sampled_data)
    if df.empty:
        raise RuntimeError("仿真轨迹为空，请检查初始参数或流场数据。")

    t_array = df['Time'].values
    w_array = df['w_c'].values

    idx_max = int(np.argmax(w_array))
    w_max = float(w_array[idx_max])

    if w_max <= 0:
        raise RuntimeError("未检测到上涌正向波瓣，请检查参数。")

    cutoff_val = 0.18 * w_max

    idx_start = idx_max
    while idx_start > 0 and w_array[idx_start] > cutoff_val:
        idx_start -= 1

    idx_end = idx_max
    while idx_end < len(w_array) - 1 and w_array[idx_end] > cutoff_val:
        idx_end += 1

    t_integral = t_array[idx_start:idx_end]
    w_integral = w_array[idx_start:idx_end]
    dh_raw = np.trapezoid(w_integral, x=t_integral)

    z_idx = np.argmin(np.abs(z_grid - thermocline_depth))
    W_z_meet = float(W_profile[z_idx])
    doppler_factor = V_rel / Cp
    Delta_Z_calc = abs(dh_raw * doppler_factor / W_z_meet)
    J = abs(Delta_Z_calc - Delta_Z_true)
    abs_error = J
    rel_error = abs_error / Delta_Z_true * 100 if Delta_Z_true != 0 else float('inf')

    # ── 轻量级返回（用于贝叶斯优化）
    if not return_diagnostic_data:
        return {
            'J': J,
            'abs_error': abs_error,
            'rel_error': rel_error,
        }

    # ── 完整诊断返回（用于绘图）
    return {
        'J': J,
        'abs_error': abs_error,
        'rel_error': rel_error,
        'Delta_Z_calc': Delta_Z_calc,
        'Delta_Z_true': Delta_Z_true,
        'w_max': w_max,
        'dh_raw': dh_raw,
        'doppler_factor': doppler_factor,
        'W_z_meet': W_z_meet,
        'df': df,
        'idx_max': idx_max,
        'idx_start': idx_start,
        'idx_end': idx_end,
        'cutoff_val': cutoff_val,
        't_integral': t_integral,
        'w_integral': w_integral,
        'thermocline_depth': thermocline_depth,
        'V_norm': V_norm,
        'zeta_norm': zeta_norm,
        'w_c_threshold': w_c_threshold,
        'V_target': V_target,
        'zeta_target': zeta_target,
        'f_s': f_s,
        'params': params,
    }

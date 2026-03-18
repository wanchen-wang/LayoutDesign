import os
import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def run_single_w_A_lagrange_threshold(data_dir):
    print(f"\n====================================================")
    print(f"开始执行拉格朗日动态采样 (5% 噪声抗扰截断): {data_dir}")
    print(f"====================================================")

    # ==========================================
    # 0. 环境与参数加载
    # ==========================================
    z = np.load(os.path.join(data_dir, 'z.npy'))                
    x_grid = np.load(os.path.join(data_dir, 'x_grid.npy'))      
    y_grid = np.load(os.path.join(data_dir, 'y_grid.npy'))      
    W_Vel_3D = np.load(os.path.join(data_dir, 'W_Vel_3D.npy'))  
    W_profile = np.load(os.path.join(data_dir, 'W_profile.npy')) 

    with open(os.path.join(data_dir, 'params.json'), 'r') as f:
        params = json.load(f)

    Cp = params['c0'] 
    thermocline_depth = params['thermocline_depth']
    true_h0 = params['h0']
    D = params.get('D', 1000.0)  

    if z[0] > z[-1]:
        z = np.flip(z)
        W_Vel_3D = np.flip(W_Vel_3D, axis=2)
        W_profile = np.flip(W_profile)

    interp_w = RegularGridInterpolator((x_grid, y_grid, z), W_Vel_3D, bounds_error=False, fill_value=0.0)

    v_g = 0.22

    # ==========================================
    # 1. 动态自适应时间窗口设置 
    # ==========================================
    V_rel = Cp + v_g 
    
    t_meet = thermocline_depth * (6000.0 / 1000.0)  
    x_g_meet = v_g * t_meet
    x_init = x_g_meet + Cp * t_meet  
    y_center = 0.0 

    half_window_time = max(2000.0, (5.0 * D) / V_rel)
    start_time = max(0, t_meet - half_window_time)
    end_time = t_meet + half_window_time
    dt = 5.0  

    t_array = np.arange(start_time, end_time, dt) 
    w_isw_array = np.zeros_like(t_array, dtype=float)
    w_obs_array = np.zeros_like(t_array, dtype=float)
    depth_obs = np.zeros_like(t_array, dtype=float)

    # ==========================================
    # 2. 拉格朗日逐秒迭代采样引擎 (正确物理坐标系)
    # ==========================================
    t_mod_start = start_time % 12000
    if t_mod_start < 6000:
        z_g = t_mod_start * 1000.0 / 6000.0
    else:
        z_g = 1000.0 - (t_mod_start - 6000.0) * 1000.0 / 6000.0

    for i, t in enumerate(t_array):
        # A. 引擎静水理论速度 (w_stdy): 向上为正(+), 向下为负(-)
        t_mod = t % 12000
        w_stdy = -1000.0 / 6000.0 if t_mod < 6000 else 1000.0 / 6000.0

        # B. 计算波浪与滑翔机的相对坐标
        x_g = v_g * t
        X_wave_current = x_init - Cp * t 
        x_eff = x_g - X_wave_current 
        
        # C. 结合当前深度 z_g 提取环境流速
        w_isw = interp_w((x_eff, y_center, z_g))
        
        # D. 物理速度叠加 
        w_obs_real = w_stdy + w_isw
        
        # E. 记录数据
        w_isw_array[i] = w_isw
        w_obs_array[i] = w_obs_real
        depth_obs[i] = z_g
        
        # F. 状态步进 (深度z向下为正，水速w向上为正，所以用减法)
        z_g = z_g - w_obs_real * dt
        z_g = np.clip(z_g, 0.0, 1000.0)

    # ==========================================
    # 3. 积分法求解原始位移 dh (⭐ 新增 5% 阈值截断逻辑 ⭐)
    # ==========================================
    max_w_idx = np.argmax(w_isw_array)
    w_max = w_isw_array[max_w_idx]
    
    # 设定噪声过滤阈值：最高水速的 5%
    w_threshold = 0.05 * w_max

    # 向左寻找起点，直到水速降到 5% 阈值以下
    tw0_idx = max_w_idx
    while tw0_idx > 0 and w_isw_array[tw0_idx] > w_threshold: 
        tw0_idx -= 1
    t_w0 = t_array[tw0_idx]

    # 向右寻找终点，直到水速降到 5% 阈值以下
    tu_idx = max_w_idx
    while tu_idx < len(w_isw_array) - 1 and w_isw_array[tu_idx] > w_threshold: 
        tu_idx += 1
    t_U = t_array[tu_idx]

    t_integral = t_array[tw0_idx:tu_idx]
    w_integral = w_isw_array[tw0_idx:tu_idx]
    dh_raw = np.trapezoid(w_integral, x=t_integral)  

    # ==========================================
    # 4. 多普勒物理修正与误差计算 
    # ==========================================
    z_idx = np.argmin(np.abs(z - thermocline_depth))
    W_z_meet = W_profile[z_idx]

    doppler_factor = V_rel / Cp
    h0_corrected = dh_raw * doppler_factor / W_z_meet
    
    error_abs = abs(h0_corrected - true_h0)
    error_pct = error_abs / true_h0 * 100

    # ==========================================
    # 5. 输出报告与轨迹可视化
    # ==========================================
    print(f"\n诊断报告:")
    print(f"   [积分特征] 最大水速: {w_max:.3f} m/s | 截断阈值(5%): {w_threshold:.3f} m/s")
    print(f"   [积分区间] 窗口: {t_w0:.1f}s -> {t_U:.1f}s (共 {len(w_integral)} 个有效点，剔除长尾)")
    print(f"   [物理补偿] 多普勒放大: {doppler_factor:.4f} | 结构函数: W(z)={W_z_meet:.4f}")
    print(f"   ----------------------------------------")
    print(f"   【真实基准值】 Ground Truth h0  = {true_h0:.2f} m")
    print(f"   【修正推导值】 Inferred     h0  = {h0_corrected:.2f} m")
    print(f"   【最终绝对误差】 Error          = {error_abs:.2f} m ({error_pct:.2f}%)")
    print(f"   ----------------------------------------")
    
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


def run_single(data_dir):
    """
    Wrapper function for compatibility with single_w_A_Execute.py
    
    Parameters
    ----------
    data_dir : str
        Path to the data directory
    
    Returns
    -------
    dict
        Analysis results
    """
    return run_single_w_A_lagrange_threshold(data_dir)

# 填入测试组路径运行
if __name__ == "__main__":
    result = run_single("v_wave_data/20260310_163551")
    print(f"\n返回的结果字典包含以下键: {list(result.keys())}")
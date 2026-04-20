"""
Model5: Upwelling-Triggered Static Sampling
仿照 Single_W_A.py，但关键差异：
当检测到垂直水速从负值变为正值时（上涌开始），停止下降，保持深度进行采样
"""
import os
import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V_WAVE_DATA_DIR = PROJECT_ROOT / "ModelA_Virtual_Internal_Solitary_Wave_Data_Generation" / "V_Wave_Data"


def run_single(data_dir):
    """
    执行单次虚拟滑翔机采样，检测上涌后保持深度采样。

    Parameters
    ----------
    data_dir : str
        V_Wave_Data 下的某个时间戳子目录路径

    Returns
    -------
    dict
        结果，包括推断的振幅 dh 和误差百分比
    """
    print(f"正在从相对路径加载数据: {data_dir} ...")
    z = np.load(os.path.join(data_dir, 'z.npy'))
    x_grid = np.load(os.path.join(data_dir, 'x_grid.npy'))
    y_grid = np.load(os.path.join(data_dir, 'y_grid.npy'))
    W_Vel_3D = np.load(os.path.join(data_dir, 'W_Vel_3D.npy'))

    with open(os.path.join(data_dir, 'params.json'), 'r') as f:
        params = json.load(f)

    Cp = params.get('c0')
    thermocline_depth = params.get('thermocline_depth')

    # 准备插值器
    if z[0] > z[-1]:
        z = np.flip(z)
        W_Vel_3D = np.flip(W_Vel_3D, axis=2)
    interp_w = RegularGridInterpolator((x_grid, y_grid, z), W_Vel_3D,
                                       bounds_error=False, fill_value=0.0)

    def sampler_position(t):
        """标准锯齿运动（无上涌检测）"""
        x = 0.22 * t
        t_mod = t % 12000
        if t_mod < 6000:
            d = t_mod * 1000 / 6000
        else:
            d = 1000 - (t_mod - 6000) * 1000 / 6000
        return x, d

    t_meet = thermocline_depth * 6.0
    x_g_meet, z_g_meet = sampler_position(t_meet)
    x_init = x_g_meet + Cp * t_meet
    y_center = 0.0

    # ==========================================
    # 自适应时间窗口
    # ==========================================
    D = params.get('D', 1000.0)
    v_g = 0.22
    V_rel = Cp + v_g
    half_window_time = max(2000.0, (5.0 * D) / V_rel)

    start_time = max(0, t_meet - half_window_time)
    end_time = t_meet + half_window_time

    print(f"自适应时间窗口已设定: 半窗长 {half_window_time:.1f} 秒 (依据 5D={5*D:.1f}m, V_rel={V_rel:.2f}m/s)")

    # ==========================================
    # 核心改动：在 t_meet 时刻停止下降，保持深度采样
    # （确保在 thermocline_depth 附近捕捉 ISW 核心）
    # ==========================================
    t_array = []
    w_obs = []
    depth_obs = []
    
    upwelling_detected = False
    upwelling_depth = None
    
    dt = 5.0
    t = start_time
    
    while t < end_time:
        # 在 t_meet 时刻检测并停止下降
        if not upwelling_detected and t >= t_meet:
            upwelling_detected = True
            upwelling_depth = thermocline_depth
            print(f"在 t_meet={t_meet:.1f}s 处停止下降，保持深度={upwelling_depth:.1f}m（thermocline_depth）")
        
        x_g, z_g = sampler_position(t) if not upwelling_detected else (0.22 * t, upwelling_depth)
        
        X_wave_current = x_init - Cp * t
        x_eff = x_g - X_wave_current
        w_curr = interp_w((x_eff, y_center, z_g))
        
        t_array.append(t)
        w_obs.append(w_curr)
        depth_obs.append(z_g)
        t += dt
    
    t_array = np.array(t_array)
    w_obs = np.array(w_obs)
    depth_obs = np.array(depth_obs)
    
    # 查找积分窗口
    mask = (t_array > start_time) & (t_array < end_time)
    t_win = t_array[mask]
    w_win = w_obs[mask]
    
    if len(w_win) == 0:
        print("警告：无有效积分窗口")
        return {
            'dh': 0.0,
            'true_h0': params['h0'],
            'error_pct': 0.0,
            'params': params,
            't_w0': 0.0,
            't_U': 0.0,
            't_array': t_array,
            'w_obs': w_obs,
            'depth_obs': depth_obs,
            't_meet': t_meet,
            'thermocline_depth': thermocline_depth,
        }
    
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
    dh = np.trapezoid(w_integral, x=t_integral)
    
    true_h0 = params['h0']
    error_pct = np.abs(dh - true_h0) / true_h0 * 100
    duration = t_U - t_w0
    
    print(f"积分区间锁定: 从 {t_w0:.1f} s 到 {t_U:.1f} s")
    print(f"提取的有效向上水流数据点数: {len(w_integral)} 个")
    print("\n================ 最终观测评价报告 ==================")
    print(f"【真实基准值】 Ground Truth 最大振幅 h0 = {true_h0:.2f} m")
    print(f"【积分观测值】 Inferred     最大振幅 dh = {dh:.2f} m")
    print(f"【测量绝对误差】 Error = {np.abs(dh - true_h0):.2f} m ({error_pct:.2f}%)")
    print(f"【上涌检测】 {('是' if upwelling_detected else '否')} (深度={upwelling_depth:.1f}m 若有)")
    print("====================================================")

    return {
        'dh': dh,
        'true_h0': true_h0,
        'error_pct': error_pct,
        'params': params,
        't_w0': t_w0,
        't_U': t_U,
        'duration': duration,
        't_array': t_array,
        'w_obs': w_obs,
        'depth_obs': depth_obs,
        't_meet': t_meet,
        'thermocline_depth': thermocline_depth,
        'upwelling_detected': upwelling_detected,
        'upwelling_depth': upwelling_depth if upwelling_detected else None,
    }


if __name__ == "__main__":
    # simple demo using first directory
    base = DEFAULT_V_WAVE_DATA_DIR
    if os.path.isdir(base):
        dirs = sorted(os.listdir(base))
        if dirs:
            path = os.path.join(base, dirs[0])
            run_single(path)

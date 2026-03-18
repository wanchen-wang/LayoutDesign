import os
import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def run_single(data_dir):
    """Execute a single virtual glider sampling using the data folder provided.

    Parameters
    ----------
    data_dir : str
        Path to one of the timestamped subdirectories under ``v_wave_data``.

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

    with open(os.path.join(data_dir, 'params.json'), 'r') as f:
        params = json.load(f)

    Cp = params.get('c0')
    thermocline_depth = params.get('thermocline_depth')

    # prepare interpolator
    if z[0] > z[-1]:
        z = np.flip(z)
        W_Vel_3D = np.flip(W_Vel_3D, axis=2)
    interp_w = RegularGridInterpolator((x_grid, y_grid, z), W_Vel_3D,
                                       bounds_error=False, fill_value=0.0)

    def sampler_position(t):
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
    # 4. 生成时间序列并执行纯净虚拟采样
    # ==========================================
    # 从 params.json 中提取波宽尺度 D 和 相速度 Cp
    D = params.get('D', 1000.0)  # 容错获取 D 

    # 提取滑翔机水平速度
    v_g = 0.22

    # 动态计算自适应时间窗口 (迎头相遇)
    V_rel = Cp + v_g  

    # 【核心修改】：KdV/DJL 波形在距离核心 5 倍 D 处能量几乎衰减殆尽 (sech^2(5) < 0.0001)
    # 我们需要确保时间窗口足以覆盖波浪到达前的 5D 和离开后的 5D
    half_window_time = (5.0 * D) / V_rel

    # 增加一个保底时间，确保即使是非常窄的波，也能有足够的环境背景 (比如至少 2000 秒)
    half_window_time = max(2000.0, half_window_time)

    # 自适应时间窗口：以相遇时间为中心，前后各取 half_window_time
    start_time = max(0, t_meet - half_window_time)
    end_time = t_meet + half_window_time

    # 打印日志以供追踪
    print(f"自适应时间窗口已设定: 半窗长 {half_window_time:.1f} 秒 (依据 5D={5*D:.1f}m, V_rel={V_rel:.2f}m/s)")

    t_array = np.arange(start_time, end_time, 5)

    w_obs = np.zeros_like(t_array, dtype=float)
    depth_obs = np.zeros_like(t_array, dtype=float)

    for i, t in enumerate(t_array):
        x_g, z_g = sampler_position(t)
        depth_obs[i] = z_g
        X_wave_current = x_init - Cp * t
        x_eff = x_g - X_wave_current
        w_obs[i] = interp_w((x_eff, y_center, z_g))

    # find integration window
    mask = (t_array > start_time) & (t_array < end_time)
    t_win = t_array[mask]
    w_win = w_obs[mask]
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

    print(f"积分区间锁定: 从 {t_w0:.1f} s 到 {t_U:.1f} s")
    print(f"提取的有效向上水流数据点数: {len(w_integral)} 个")
    print("\n================ 最终观测评价报告 ==================")
    print(f"【真实基准值】 Ground Truth 最大振幅 h0 = {true_h0:.2f} m")
    print(f"【积分观测值】 Inferred     最大振幅 dh = {dh:.2f} m")
    print(f"【测量绝对误差】 Error = {np.abs(dh - true_h0):.2f} m ({error_pct:.2f}%)")
    print("====================================================")

    return {
        'dh': dh,
        'true_h0': true_h0,
        'error_pct': error_pct,
        'params': params,
        't_w0': t_w0,
        't_U': t_U,
        't_array': t_array,
        'w_obs': w_obs,
        'depth_obs': depth_obs,
        't_meet': t_meet,
        'thermocline_depth': thermocline_depth,
    }


if __name__ == "__main__":
    # simple demo using first directory
    base = "v_wave_data"
    if os.path.isdir(base):
        dirs = sorted(os.listdir(base))
        if dirs:
            path = os.path.join(base, dirs[0])
            run_single(path)

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V_WAVE_DATA_DIR = PROJECT_ROOT / "ModelA_Virtual_Internal_Solitary_Wave_Data_Generation" / "V_Wave_Data"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "Analysis_Results_Shiyan4"


def run_single_shiyan4(data_dir, return_full=False):
    """
    实验组4采样：前半段不受拉格朗日影响，后半段受垂直流影响。
    
    逻辑：
    1. 默认状态：不受水流影响，按标准轨迹运动
    2. 当检测到垂直水流由负变正时：进入拉格朗日影响模式
       - 下降速度加快（w_obs_real = w_stdy + w_isw）
       - 水平位置不受影响（按标准轨迹）
    3. 进入影响模式后持续受影响直到采样结束
    
    Parameters
    ----------
    data_dir : str
        Path to one of the timestamped subdirectories under V_Wave_Data.
    return_full : bool
        If True, return full diagnostic data for plotting.
        
    Returns
    -------
    dict
        Results including inferred amplitude dh and error percentage.
    """
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

    # 适配坐标轴方向
    if z[0] > z[-1]:
        z = np.flip(z)
        W_Vel_3D = np.flip(W_Vel_3D, axis=2)
        W_profile = np.flip(W_profile)

    interp_w = RegularGridInterpolator((x_grid, y_grid, z), W_Vel_3D,
                                       bounds_error=False, fill_value=0.0)

    v_g = 0.22
    w_stdy_norm = 1000.0 / 6000.0

    # 计算相对速度（迎头相遇）
    V_rel = Cp + v_g

    t_meet = thermocline_depth * (6000.0 / 1000.0)
    dt = 5.0

    # 自适应时间窗口
    half_window_time = max(4000.0, (5.0 * D) / V_rel)
    start_time = max(0, t_meet - half_window_time)
    end_time = t_meet + half_window_time

    t_full_array = np.arange(0, end_time, dt)
    w_isw_full = np.zeros_like(t_full_array, dtype=float)
    w_obs_full = np.zeros_like(t_full_array, dtype=float)
    depth_full = np.zeros_like(t_full_array, dtype=float)
    mode_full = np.zeros_like(t_full_array, dtype=int)  # 0=不受影响, 1=受拉格朗日影响

    z_g = 0.0
    w_isw_prev = 0.0  # 上一时刻的垂直水流值，用于检测由负变正的转变
    in_lagrangian_mode = False  # 是否处于拉格朗日影响模式

    for i, t in enumerate(t_full_array):
        t_mod = t % 12000
        w_stdy = -w_stdy_norm if t_mod < 6000 else w_stdy_norm

        # 计算波浪核心当前位置和相对距离（滑翔机水平位置按标准轨迹）
        x_g = v_g * t
        X_wave_current = v_g * t_meet + Cp * t_meet - Cp * t  # x_init = x_g_meet + Cp * t_meet
        x_eff = x_g - X_wave_current

        # 获取当前位置的垂直流速（不考虑水平流速）
        w_isw = interp_w((x_eff, 0.0, z_g))

        # ==================== 实验组4逻辑 ====================
        # 检测由负变正的转变 - 这是进入受影响模式的触发点
        if w_isw_prev < 0 and w_isw >= 0:
            # 【进入影响区域】垂直水流由负变正，开始受拉格朗日影响
            in_lagrangian_mode = True

        if in_lagrangian_mode:
            # 【后半段】受拉格朗日影响区域（只受垂直水流影响）
            w_obs_real = w_stdy + w_isw
            mode_full[i] = 1
        else:
            # 【前半段】不受影响，保持标准采样（不考虑水流影响）
            w_obs_real = w_stdy
            mode_full[i] = 0

        # 更新上一时刻的水流值
        w_isw_prev = w_isw

        # 记录数据
        w_isw_full[i] = w_isw
        w_obs_full[i] = w_obs_real
        depth_full[i] = z_g

        # 状态步进
        z_g = z_g - w_obs_real * dt
        z_g = np.clip(z_g, 0.0, 1000.0)

    # 截取有效窗口
    mask = (t_full_array >= start_time) & (t_full_array <= end_time)
    t_array = t_full_array[mask]
    w_isw_array = w_isw_full[mask]
    w_obs_array = w_obs_full[mask]
    depth_obs = depth_full[mask]
    mode_array = mode_full[mask]

    # 积分窗口（寻找正向波瓣）
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

    result = {
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
    }

    if return_full:
        result.update({
            't_array': t_array,
            'w_isw_array': w_isw_array,
            'w_obs_array': w_obs_array,
            'depth_obs': depth_obs,
            'mode_array': mode_array,
            't_meet': t_meet,
            'thermocline_depth': thermocline_depth,
        })

    return result


def batch_process(base_data_dir=DEFAULT_V_WAVE_DATA_DIR):
    print(f"\n{'='*60}")
    print(f"🚀 启动实验组4采样仿真（前半段不受影响，后半段受垂直流影响）...")
    print(f"{'='*60}")

    if not os.path.exists(base_data_dir):
        print(f"⚠️ 找不到数据目录: {base_data_dir}")
        return

    wave_folders = [f for f in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, f))]
    if not wave_folders:
        print("⚠️ 数据目录下没有找到子组数据文件夹！")
        return

    print(f"[*] 共发现 {len(wave_folders)} 组内孤立波测试样本。\n")

    results_list = []
    for i, folder_name in enumerate(wave_folders, 1):
        data_dir = os.path.join(base_data_dir, folder_name)
        try:
            res = run_single_shiyan4(data_dir)
            results_list.append(res)
            print(f"[{i}/{len(wave_folders)}] ✓ {folder_name} 处理成功 (Error: {res['error_pct']:.2f}%)")
        except Exception as e:
            print(f"[{i}/{len(wave_folders)}] ✗ {folder_name} 处理失败: {e}")

    if results_list:
        df = pd.DataFrame(results_list)
        os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
        output_filename = DEFAULT_RESULTS_DIR / "analysis_results_shiyan4.csv"
        df.to_csv(output_filename, index=False)
        print(f"\n✅ 成功保存: {output_filename} (包含 {len(df)} 组观测数据，平均误差: {df['error_pct'].mean():.2f}%)")


if __name__ == "__main__":
    batch_process()

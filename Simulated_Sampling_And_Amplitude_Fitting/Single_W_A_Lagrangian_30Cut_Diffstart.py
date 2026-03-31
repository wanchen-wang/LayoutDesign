"""
单滑翔机 30% 截断振幅拟合模块
==============================
职责：每次调用只处理一个滑翔机的一条轨迹（单次采样）。

  run_single_group_30cut  — 核心函数，供 Basic_Models.py 循环调用（每个滑翔机一次）
  run_batch_30cut         — 单滑翔机模式批处理，用于独立振幅误差分析
  get_glider_config       — 返回标准滑翔机运动学参数

该程序提供变换坐标起点功能
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SINGLE_GLIDER_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Analysis_Results_SwA_Lagrangian_Cut_Data")
DEFAULT_SINGLE_GLIDER_OUTPUT_CSV = os.path.join(
    DEFAULT_SINGLE_GLIDER_OUTPUT_DIR,
    "analysis_results_swA_lagrangian_30cut.csv",
)


# ==========================================
# 滑翔机运动学配置
# ==========================================
DEFAULT_GLIDER_CONFIG = {
    "v_g": 0.22,
    "v_z": 1000.0 / 6000.0,
    "depth_max": 1000.0,
}


def get_glider_config():
    """Return the glider kinematics used by the 30% cut workflow."""
    return dict(DEFAULT_GLIDER_CONFIG)


# ==========================================
# 内部辅助函数
# ==========================================
def _resolve_selected_groups(base_data_dir, start_idx=None, end_idx=None, max_groups=None):
    """按 1-based 索引范围和数量过滤待处理的波组文件夹列表。"""
    wave_folders = sorted(
        [f for f in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, f))]
    )
    if not wave_folders:
        return []

    if start_idx is None:
        start_idx = 1
    if end_idx is None:
        end_idx = len(wave_folders)

    start_idx = max(1, int(start_idx))
    end_idx = min(len(wave_folders), int(end_idx))
    if start_idx > end_idx:
        return []

    selected = wave_folders[start_idx - 1 : end_idx]
    if max_groups is not None:
        selected = selected[: max(0, int(max_groups))]
    return selected


# ==========================================
# 核心采样函数（单滑翔机，单次调用）
# ==========================================
def run_single_group_30cut(
    data_dir,
    X0_global=0.0,
    Y0_global=0.0,
    T0_global=0.0,
    cut_percentage=30,
    glider_config=None,
    enable_amplitude_calc=True,
):
    """
    单次 30% 截断采样计算。

    使用 3D 波场轨迹仿真滑翔机在指定多普勒比的波坏中的运动。
    当 enable_amplitude_calc=True 时，会执行振幅积分与误差计算；
    当 enable_amplitude_calc=False 时，仅返回时刻与轨迹信息。

    参数:
      data_dir      — 包含 z/x_grid/y_grid/W_Vel_3D/W_profile/params.json 的目录
      X0_global     — 滑翔机全局 X 方向出发点 (m)
      Y0_global     — 滑翔机全局 Y 方向出发点 (m)
      T0_global     — 滑翔机出发时刻 (s)
    cut_percentage — 截断阈値百分比（默认 30）
    enable_amplitude_calc — 是否执行振幅积分与误差计算
    """
    cfg = get_glider_config()
    if glider_config:
        cfg.update(glider_config)
    v_g = float(cfg["v_g"])
    depth_max = float(cfg["depth_max"])

    pct = max(0.0, min(40.0, float(cut_percentage))) / 100.0

    z = np.load(os.path.join(data_dir, 'z.npy'))
    x_grid = np.load(os.path.join(data_dir, 'x_grid.npy'))
    y_grid = np.load(os.path.join(data_dir, 'y_grid.npy'))
    W_Vel_3D = np.load(os.path.join(data_dir, 'W_Vel_3D.npy'))
    W_profile = np.load(os.path.join(data_dir, 'W_profile.npy')) if enable_amplitude_calc else None
    with open(os.path.join(data_dir, 'params.json'), 'r') as f:
        params = json.load(f)

    Cp = params['c0']
    thermocline_depth = params['thermocline_depth']
    true_h0 = params.get('h0', np.nan)
    D = params.get('D', depth_max)

    if z[0] > z[-1]:
        z = np.flip(z)
        W_Vel_3D = np.flip(W_Vel_3D, axis=2)
        if W_profile is not None:
            W_profile = np.flip(W_profile)

    interp_w = RegularGridInterpolator((x_grid, y_grid, z), W_Vel_3D, bounds_error=False, fill_value=0.0)

    V_rel = Cp + v_g
    t_meet = thermocline_depth * (6000.0 / depth_max)
    x_init = v_g * t_meet + Cp * t_meet

    half_window_time = max(4000.0, (8.0 * D) / V_rel)
    start_time = max(0.0, t_meet - half_window_time)
    end_time = t_meet + half_window_time
    dt = 5.0

    t_local_array = np.arange(start_time, end_time, dt)
    t_global_array = T0_global + t_local_array
    w_isw_array = np.zeros_like(t_local_array, dtype=float)
    x_track_global = np.zeros_like(t_local_array, dtype=float)
    y_track_global = np.full_like(t_local_array, Y0_global, dtype=float)
    z_track = np.zeros_like(t_local_array, dtype=float)

    # 一个下潜-上升周期 = 12000 s：前 6000 s 下潜，后 6000 s 上升
    # 根据全局发车时刻算出仿真开始时滑翔机处于哪个阶段、具体深度
    t_mod_start = (T0_global + start_time) % 12000.0
    if t_mod_start < 6000.0:
        z_g = t_mod_start * depth_max / 6000.0           # 下潜阶段
    else:
        z_g = depth_max - (t_mod_start - 6000.0) * depth_max / 6000.0  # 上升阶段

    for i, t_local in enumerate(t_local_array):
        t_global = T0_global + t_local
        t_mod    = t_global % 12000.0
        # 各阶段的基本垂向速度：下潜时为负（深度增大），上升时为正
        w_stdy = -depth_max / 6000.0 if t_mod < 6000.0 else depth_max / 6000.0

        # 滑翔机在全局坐标系中仅沿 +X 方向飞行，Y 坐标恒为 Y0_global
        x_g_global = X0_global + v_g * t_local

        # 波的参考点在全局坐标系中的当前位置（初始位置 - Cp * 时间）
        X_wave_current_global = X0_global + x_init - Cp * t_local

        # 滑翔机相对于波坐标系的 x 坐标，用于查询 W_Vel_3D
        x_eff_global = x_g_global - X_wave_current_global

        sample_point = np.array([[x_eff_global, Y0_global, z_g]], dtype=float)
        w_isw = float(interp_w(sample_point)[0])

        # 实际垂向速度 = 基本制定阶段速 + ISW 捧动速度
        w_obs_real = w_stdy + w_isw

        w_isw_array[i] = w_isw
        x_track_global[i] = x_g_global
        z_track[i] = z_g

        z_g = z_g - w_obs_real * dt
        z_g = np.clip(z_g, 0.0, depth_max)

    max_w_idx = np.argmax(w_isw_array)
    t_peak = t_global_array[max_w_idx]
    w_max = w_isw_array[max_w_idx]
    w_threshold = pct * w_max

    tw0_idx = max_w_idx
    while tw0_idx > 0 and w_isw_array[tw0_idx] > w_threshold:
        tw0_idx -= 1
    t_w0 = t_global_array[tw0_idx]

    tu_idx = max_w_idx
    while tu_idx < len(w_isw_array) - 1 and w_isw_array[tu_idx] > w_threshold:
        tu_idx += 1
    t_U = t_global_array[tu_idx]

    duration = t_U - t_w0

    if enable_amplitude_calc:
        t_integral = t_global_array[tw0_idx:tu_idx]
        w_integral = w_isw_array[tw0_idx:tu_idx]
        dh_raw = np.trapezoid(w_integral, x=t_integral)

        z_idx = np.argmin(np.abs(z - thermocline_depth))
        W_z_meet = W_profile[z_idx]
        doppler_factor = V_rel / Cp
        h0_corrected = dh_raw * doppler_factor / W_z_meet

        error_abs = abs(h0_corrected - true_h0)
        error_pct = error_abs / true_h0 * 100
    else:
        dh_raw = np.nan
        h0_corrected = np.nan
        error_abs = np.nan
        error_pct = np.nan

    return {
        'X0': float(X0_global),
        'Y0': float(Y0_global),
        'T0': float(T0_global),
        't_w0': t_w0,
        't_peak': t_peak,
        't_U': t_U,
        'duration': duration,
        'dh_raw': dh_raw,
        'dh': h0_corrected,
        'true_h0': true_h0,
        'abs_error': error_abs,
        'error_pct': error_pct,
        'error_density': (error_pct / duration) if duration > 0 else np.nan,
        't_global_array': t_global_array,
        'w_sampled': w_isw_array,
        'x_track_global': x_track_global,
        'y_track_global': y_track_global,
        'z_track': z_track,
    }


# ==========================================
# 批处理入口（单滑翔机模式，独立使用）
# ==========================================
def run_batch_30cut(
    base_data_dir="D:\\PYTHON\\layout design\\V_Wave_Data_Line",
    output_csv=DEFAULT_SINGLE_GLIDER_OUTPUT_CSV,
    start_idx=None,
    end_idx=None,
    max_groups=None,
    cut_percentage=30,
    glider_config=None,
    deployment_cmds=None,
):
    print(f"\n{'='*60}")
    print(f"🚀 启动固定阈值处理: {int(cut_percentage)}% cut")
    print(f"📂 数据目录: {base_data_dir}")
    print(f"🧾 输出文件: {output_csv}")
    print(f"{'='*60}")

    if not os.path.exists(base_data_dir):
        print(f"⚠️ 找不到数据目录: {base_data_dir}，请检查路径！")
        return None

    selected_groups = _resolve_selected_groups(
        base_data_dir=base_data_dir,
        start_idx=start_idx,
        end_idx=end_idx,
        max_groups=max_groups,
    )
    if not selected_groups:
        print("⚠️ 数据目录下没有找到子组数据文件夹！")
        return None

    print(f"[*] 实际处理组数: {len(selected_groups)}")
    print(f"[*] 组别范围: {selected_groups[0]} -> {selected_groups[-1]}\n")

    results = []
    for folder_name in selected_groups:
        data_dir = os.path.join(base_data_dir, folder_name)
        cmd = (deployment_cmds or {}).get(folder_name, {})
        X0_global = float(cmd.get('X0', 0.0))
        Y0_global = float(cmd.get('Y0', 0.0))
        T0_global = float(cmd.get('T0', 0.0))
        try:
            row = run_single_group_30cut(
                data_dir=data_dir,
                X0_global=X0_global,
                Y0_global=Y0_global,
                T0_global=T0_global,
                cut_percentage=cut_percentage,
                glider_config=glider_config,
                enable_amplitude_calc=True,
            )
            row['wave_id'] = folder_name
            results.append(row)
        except Exception as e:
            print(f"[警告] 子组 {folder_name} 处理异常，已跳过: {e}")

    if not results:
        print("⚠️ 没有可写入的数据。")
        return None

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"✅ 保存完成: {output_csv}")
    print(f"   样本数: {len(df)}, 平均误差: {df['error_pct'].mean():.2f}%")
    return output_csv

if __name__ == "__main__":
    print("固定 30% 截断批处理")
    print("可选输入处理组范围和处理组数，默认输出到 Analysis_Results_SwA_Lagrangian_Cut_Data。")

    start_text = input("起始组编号 start_idx (默认 1): ").strip()
    end_text = input("结束组编号 end_idx (默认最后一组): ").strip()
    count_text = input("最多处理组数 max_groups (默认不限): ").strip()

    start_idx = int(start_text) if start_text else None
    end_idx = int(end_text) if end_text else None
    max_groups = int(count_text) if count_text else None

    run_batch_30cut(
        start_idx=start_idx,
        end_idx=end_idx,
        max_groups=max_groups,
        cut_percentage=30,
    )
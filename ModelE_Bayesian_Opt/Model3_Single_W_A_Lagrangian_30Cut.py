import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

#这个程序是在ModelB的结论上，把30%截断的流程单独抽离出来，形成一个独立的脚本，专门用来处理30%截断的数据生成.
#这个程序的模拟运动和采样一直是基础设置，没有四个特征值的输入
#并且还修改了积分步长为0.05s，与Model3_Lagrange_Dynamic_Sampling_and_Error_Calculation.py保持一致，来验证两者在不激发阈值情况下的结果是否接近。
#与Model3_Lagrange_Dynamic_Sampling_and_Error_Calculation.py在不激发阈值情况下跑出的结果只差0.02%
#我认为可以当一种结果了。
#结果存在Analysis_A_UGstandard_Data文件夹

def process_30cut(
    base_data_dir="D:\\PYTHON\\layout design\\V_Wave_Data",
):
    pct_int = 30
    pct = 0.30

    print(f"\n{'='*60}")
    print(f"🚀 启动截断阈值 30% 的计算...")
    print(f"{'='*60}")

    if not os.path.exists(base_data_dir):
        print(f"⚠️ 找不到数据目录: {base_data_dir}，请检查路径！")
        return

    # 获取所有子组波浪数据的文件夹列表
    wave_folders = [f for f in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, f))]

    if not wave_folders:
        print("⚠️ 数据目录下没有找到子组数据文件夹！")
        return

    print(f"[*] 共发现 {len(wave_folders)} 组内孤立波测试样本。\n")
    print(f"▶ 正在处理截断阈值: {pct_int}% (w_threshold = {pct} * w_max) ...")

    results_for_current_threshold = []

    # 遍历每一个滑翔机与波浪相遇的子组
    for folder_name in wave_folders:
        data_dir = os.path.join(base_data_dir, folder_name)

        try:
            # 1. 加载子组环境数据
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

            # 适配坐标轴方向
            if z[0] > z[-1]:
                z = np.flip(z)
                W_Vel_3D = np.flip(W_Vel_3D, axis=2)
                W_profile = np.flip(W_profile)

            # 额外加载密度/温度/水平流速剖面（1D，与 z 对应）
            rho_profile = np.load(os.path.join(data_dir, 'rho_profile.npy'))
            T_profile   = np.load(os.path.join(data_dir, 'T_profile.npy'))
            U_profile   = np.load(os.path.join(data_dir, 'U_profile.npy'))

            interp_w = RegularGridInterpolator((x_grid, y_grid, z), W_Vel_3D, bounds_error=False, fill_value=0.0)

            # 海表参考温度（z=0 处）
            surface_T0 = float(np.interp(0.0, z, T_profile))

            v_g = 0.22

            # 2. 生成拉格朗日时间跑道
            V_rel = Cp + v_g
            t_meet = thermocline_depth * (6000.0 / 1000.0)
            x_init = v_g * t_meet + Cp * t_meet

            half_window_time = max(4000.0, (8.0 * D) / V_rel)
            start_time = max(0, t_meet - half_window_time)
            end_time = t_meet + half_window_time
            dt = 0.05

            t_array = np.arange(start_time, end_time, dt)
            w_isw_array = np.zeros_like(t_array, dtype=float)

            # 初始化滑翔机的起始下潜深度 (从循环外初始化一次)
            t_mod_start = start_time % 12000
            if t_mod_start < 6000:
                z_g = t_mod_start * 1000.0 / 6000.0
            else:
                z_g = 1000.0 - (t_mod_start - 6000.0) * 1000.0 / 6000.0

            # 3. 逐秒采样垂直水速 (Lagrangian深度跟踪)
            for i, t in enumerate(t_array):
                # 引擎静水理论下潜速度
                t_mod = t % 12000
                w_stdy = -1000.0 / 6000.0 if t_mod < 6000 else 1000.0 / 6000.0
                
                x_g = v_g * t
                X_wave_current = x_init - Cp * t 
                x_eff = x_g - X_wave_current 
                
                w_isw = interp_w((x_eff, 0.0, z_g))
                w_obs_real = w_stdy + w_isw
                
                w_isw_array[i] = w_isw
                
                # Lagrangian状态步进：基于实际观测速度更新深度
                z_g = z_g - w_obs_real * dt
                z_g = np.clip(z_g, 0.0, 1000.0)

            # 4. 根据 30% 动态寻点截断
            max_w_idx = np.argmax(w_isw_array)
            w_max = w_isw_array[max_w_idx]
            w_threshold = pct * w_max

            # 向左寻找起点
            tw0_idx = max_w_idx
            while tw0_idx > 0 and w_isw_array[tw0_idx] > w_threshold:
                tw0_idx -= 1
            t_w0 = t_array[tw0_idx]

            # 向右寻找终点
            tu_idx = max_w_idx
            while tu_idx < len(w_isw_array) - 1 and w_isw_array[tu_idx] > w_threshold:
                tu_idx += 1
            t_U = t_array[tu_idx]

            # 5. 梯形数值积分
            t_integral = t_array[tw0_idx:tu_idx]
            w_integral = w_isw_array[tw0_idx:tu_idx]
            dh_raw = np.trapezoid(w_integral, x=t_integral)

            # 6. 补偿与误差结算
            z_idx = np.argmin(np.abs(z - thermocline_depth))
            W_z_meet = W_profile[z_idx]
            doppler_factor = V_rel / Cp
            h0_corrected = dh_raw * doppler_factor / W_z_meet

            error_abs = abs(h0_corrected - true_h0)
            error_pct = error_abs / true_h0 * 100
            duration = t_U - t_w0

            results_for_current_threshold.append({
                'wave_id': folder_name,
                't_w0': t_w0,
                't_U': t_U,
                'duration': duration,
                'dh_raw': dh_raw,
                'dh': h0_corrected,
                'true_h0': true_h0,
                'error_pct': error_pct
            })

        except Exception as e:
            print(f"      [警告] 子组 {folder_name} 处理异常跳过: {e}")

    # 保存结果为 CSV 文件
    if results_for_current_threshold:
        df = pd.DataFrame(results_for_current_threshold)
        # 添加局部误差密度指标用于后续深度分析
        df['error_density'] = df['error_pct'] / df['duration']

        output_dir = "D:\\PYTHON\\layout design\\Analysis_A_UGstandard_Data"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"analysis_results_swA_lagrangian_{pct_int}cut_dt0.05.csv")
        df.to_csv(output_filename, index=False)
        print(f"   ✅ 成功保存: {output_filename} (包含 {len(df)} 组观测数据，平均误差: {df['error_pct'].mean():.2f}%)")

    print(f"\n{'='*60}")
    print(f"🎉 截断阈值 30% 的数据生成完毕！")
    print(f"{'='*60}")

if __name__ == "__main__":
    process_30cut()
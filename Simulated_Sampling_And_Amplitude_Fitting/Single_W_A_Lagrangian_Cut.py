import os
import json
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

def batch_process_multiple_thresholds(base_data_dir="D:\\PYTHON\\layout design\\V_Wave_Data"):
    print(f"\n{'='*60}")
    print(f"🚀 启动大规模敏感性扫描: 1% 到 40% 动态截断阈值测试")
    print(f"   理论支撑: 寻找全局截断误差与物理动能捕获的 Bias-Variance 最优点")
    print(f"{'='*60}")

    if not os.path.exists(base_data_dir):
        print(f"⚠️ 找不到数据目录: {base_data_dir}，请检查路径！")
        return

    # 获取所有子组波浪数据的文件夹列表
    wave_folders = [f for f in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, f))]
    
    if not wave_folders:
        print("⚠️ 数据目录下没有找到子组数据文件夹！")
        return

    print(f"[*] 共发现 {len(wave_folders)} 组内孤立波测试样本。即将开始 40 轮全量扫描...\n")

    # 外层循环：遍历 1% 到 40% 的截断比例
    for pct_int in range(1, 41):
        pct = pct_int / 100.0  # 将 1 转换为 0.01
        
        # 检测文件是否已存在，若存在则跳过该阈值
        output_filename = f"D:\\PYTHON\\layout design\\Analysis_Results_SwA_Lagrangian_Cut_Data\\analysis_results_swA_lagrangian_{pct_int}cut.csv"
        if os.path.exists(output_filename):
            print(f"⏭️  跳过截断阈值: {pct_int}% ({output_filename} 已存在)")
            continue
        
        print(f"▶ 正在处理截断阈值: {pct_int}% (w_threshold = {pct} * w_max) ...")
        
        results_for_current_threshold = []
        
        # 内层循环：遍历每一个滑翔机与波浪相遇的子组
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

                interp_w = RegularGridInterpolator((x_grid, y_grid, z), W_Vel_3D, bounds_error=False, fill_value=0.0)
                v_g = 0.22

                # 2. 生成拉格朗日时间跑道 (延长至 4000s 防止数组越界)
                V_rel = Cp + v_g 
                t_meet = thermocline_depth * (6000.0 / 1000.0)  
                x_init = v_g * t_meet + Cp * t_meet  

                half_window_time = max(4000.0, (8.0 * D) / V_rel)
                start_time = max(0, t_meet - half_window_time)
                end_time = t_meet + half_window_time
                dt = 5.0  

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
                    
                # 4. 根据当前设定的 % 动态寻点截断
                max_w_idx = np.argmax(w_isw_array)
                w_max = w_isw_array[max_w_idx]
                w_threshold = pct * w_max  # 动态计算当前截断极值

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
                
                # 收集该子组在该阈值下的指标
                results_for_current_threshold.append({
                    'wave_id': folder_name,
                    't_w0': t_w0,
                    't_U': t_U,
                    'duration': duration,
                    'dh_raw': dh_raw,
                    'dh': h0_corrected,
                    'true_h0': true_h0,
                    'abs_error': error_abs,
                    'error_pct': error_pct
                })

            except Exception as e:
                print(f"      [警告] 子组 {folder_name} 处理异常跳过: {e}")

        # 当一个阈值的全部子组跑完后，保存为一个独立的 CSV 文件
        if results_for_current_threshold:
            df = pd.DataFrame(results_for_current_threshold)
            # 添加局部误差密度指标用于后续深度分析
            df['error_density'] = df['error_pct'] / df['duration']
            
            output_filename = f"D:\\PYTHON\\layout design\\Analysis_Results_SwA_Lagrangian_Cut_Data\\analysis_results_swA_lagrangian_{pct_int}cut.csv"
            df.to_csv(output_filename, index=False)
            print(f"   ✅ 成功保存: {output_filename} (包含 {len(df)} 组观测数据，平均误差: {df['error_pct'].mean():.2f}%)")

    print(f"\n{'='*60}")
    print(f"🎉 全部 40 个阈值的数据生成完毕！")
    print(f"{'='*60}")

if __name__ == "__main__":
    batch_process_multiple_thresholds()
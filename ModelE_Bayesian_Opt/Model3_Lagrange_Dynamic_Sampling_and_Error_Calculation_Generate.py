"""
调用 process_30cut 函数处理 V_Wave_Data 中的所有流场数据
并将结果存储为 CSV 格式：wave_id, dh, true_h0, abs_error, error_pct
"""
import os
import json
import pandas as pd
from datetime import datetime
from Model3_Lagrange_Dynamic_Sampling_and_Error_Calculation import process_30cut

def main():
    # ========== 配置参数 ==========
    w_c_threshold = 9999        # 触发高频采样的垂直流速阈值 (m/s) - 永远不触发高频模式
    V_target = 0.276            # 目标静水滑翔速度 (m/s)
    zeta_target = -37.2         # 目标滑翔轨迹角 (度)
    f_s = 0.2                   # 目标采样频率 (Hz) - 与参考文件 dt=5s 等价
    
    # V_Wave_Data 根目录
    v_wave_data_root = r"d:\PYTHON\layout design\V_Wave_Data"
    
    # ========== 获取所有时间戳文件夹 ==========
    if not os.path.exists(v_wave_data_root):
        print(f"错误：V_Wave_Data 目录不存在: {v_wave_data_root}")
        return
    
    # 获取所有子目录（时间戳文件夹）
    data_dirs = sorted([
        d for d in os.listdir(v_wave_data_root) 
        if os.path.isdir(os.path.join(v_wave_data_root, d))
    ])
    
    if not data_dirs:
        print("警告：未找到任何数据文件夹")
        return
    
    print(f"找到 {len(data_dirs)} 个数据文件夹\n")
    
    # ========== 处理每个数据文件夹 ==========
    results = []
    
    for idx, data_dir_name in enumerate(data_dirs, 1):
        run_data_dir = os.path.join(v_wave_data_root, data_dir_name)
        
        try:
            print(f"[{idx}/{len(data_dirs)}] 处理: {data_dir_name} ...", end=" ", flush=True)
            
            # 验证必要的文件存在
            required_files = ['params.json', 'x_grid.npy', 'y_grid.npy', 'z.npy', 'W_Vel_3D.npy', 'W_profile.npy']
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(run_data_dir, f))]
            
            if missing_files:
                print(f"跳过 (缺失文件: {', '.join(missing_files)})")
                continue
            
            # 读取 params.json 获取真实的 h0
            with open(os.path.join(run_data_dir, 'params.json'), 'r') as f:
                params = json.load(f)
            true_h0 = params.get('h0', None)
            
            # 调用 process_30cut 函数
            result_dict = process_30cut(w_c_threshold, V_target, zeta_target, f_s, run_data_dir)
            
            # 提取需要的数据
            dh_calculated = result_dict['Delta_Z_calc']
            abs_error = result_dict['abs_error']
            error_pct = result_dict['rel_error']
            
            print(f"完成")
            
            # 按用户指定的格式记录结果
            results.append({
                'wave_id': data_dir_name,
                'dh': dh_calculated,
                'true_h0': true_h0,
                'abs_error': abs_error,
                'error_pct': error_pct
            })
        
        except Exception as e:
            print(f"错误: {str(e)}")
    
    # ========== 保存结果到 CSV ==========
    if results:
        # 转换为 DataFrame
        df_results = pd.DataFrame(results)
        
        # 保存为 CSV
        output_csv = r"D:\PYTHON\layout design\ModelE_Bayesian_Opt\Analysis_A_Bayesian_Opt\Model3_Unprovoke_Comparison.csv"
        df_results.to_csv(output_csv, index=False, encoding='utf-8')
        
        print(f"\n{'='*60}")
        print(f"处理完成！结果已保存到:")
        print(f"{output_csv}")
        print(f"{'='*60}")
        
        # 显示统计信息
        print(f"\n统计信息:")
        print(f"  总处理数: {len(results)} 个")
        
        if not df_results['abs_error'].isna().all():
            print(f"  平均绝对误差: {df_results['abs_error'].mean():.6f} m")
            print(f"  最小绝对误差: {df_results['abs_error'].min():.6f} m")
            print(f"  最大绝对误差: {df_results['abs_error'].max():.6f} m")
            
        if not df_results['error_pct'].isna().all():
            print(f"  平均相对误差: {df_results['error_pct'].mean():.4f} %")
            print(f"  最小相对误差: {df_results['error_pct'].min():.4f} %")
            print(f"  最大相对误差: {df_results['error_pct'].max():.4f} %")
        
        print(f"\n结果 DataFrame 预览:")
        print(df_results.head(10))
    else:
        print("\n未获得任何结果")

if __name__ == "__main__":
    main()

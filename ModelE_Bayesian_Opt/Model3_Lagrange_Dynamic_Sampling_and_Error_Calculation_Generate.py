"""
调用 process_30cut 函数处理 V_Wave_Data 中的所有流场数据
并将结果存储为 CSV 格式：wave_id, dh, true_h0, abs_error, error_pct
"""
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from Model3_Lagrange_Dynamic_Sampling_and_Error_Calculation import process_30cut


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = Path(__file__).resolve().parent
V_WAVE_DATA_ROOT = PROJECT_ROOT / "ModelA_Virtual_Internal_Solitary_Wave_Data_Generation" / "V_Wave_Data"
MODEL3_ANALYSIS_DIR = MODULE_DIR / "Analysis_Bayesian_Opt_Model3_Data"

def _process_one_dir(args):
    """处理单个数据文件夹的函数，用于多进程并行处理"""
    data_dir_name, v_wave_data_root, w_c_threshold, V_target, zeta_target, f_s = args
    run_data_dir = os.path.join(v_wave_data_root, data_dir_name)

    try:
        # 验证必要的文件存在
        required_files = ['params.json', 'x_grid.npy', 'y_grid.npy', 'z.npy', 'W_Vel_3D.npy', 'W_profile.npy']
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(run_data_dir, f))]

        if missing_files:
            return None  # 返回 None 表示跳过

        # 读取 params.json 获取真实的 h0
        with open(os.path.join(run_data_dir, 'params.json'), 'r') as f:
            params = json.load(f)
        true_h0 = params.get('h0', None)

        # 调用 process_30cut 函数
        result_dict = process_30cut(w_c_threshold, V_target, zeta_target, f_s, run_data_dir, return_diagnostic_data=True)

        # 提取需要的数据
        dh_calculated = result_dict['Delta_Z_calc']
        abs_error = result_dict['abs_error']
        error_pct = result_dict['rel_error']

        # 返回结果字典
        return {
            'wave_id': data_dir_name,
            'dh': dh_calculated,
            'true_h0': true_h0,
            'abs_error': abs_error,
            'error_pct': error_pct
        }

    except Exception as e:
        return None  # 返回 None 表示处理失败

def main():
    # ========== 配置参数 ==========
    w_c_threshold = 0.329        # 触发高频采样的垂直流速阈值 (m/s) - 永远不触发高频模式
    V_target = 0.343        # 目标静水滑翔速度 (m/s)
    zeta_target = -24.64         # 目标滑翔轨迹角 (度)
    f_s = 0.5                   # 目标采样频率 (Hz) - 与参考文件 dt=5s 等价

    # 多进程配置
    n_workers = max(1, multiprocessing.cpu_count() - 1)  # 使用 CPU 核心数 - 1 个工作进程
    
    # V_Wave_Data 根目录
    v_wave_data_root = V_WAVE_DATA_ROOT
    
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
    
    # ========== 用户选择处理数量 ==========
    while True:
        try:
            print(f"请选择要处理的数据组数 (1-{len(data_dirs)}, 或输入 0 处理全部):")
            num_to_process = int(input().strip())
            
            if num_to_process == 0:
                num_to_process = len(data_dirs)
                break
            elif 1 <= num_to_process <= len(data_dirs):
                break
            else:
                print(f"请输入 1-{len(data_dirs)} 之间的数字，或输入 0 处理全部。")
        except ValueError:
            print("请输入有效数字。")
    
    # 选择要处理的数据文件夹
    selected_dirs = data_dirs[:num_to_process]
    print(f"将处理前 {len(selected_dirs)} 个数据文件夹\n")
    
    # ========== 处理每个数据文件夹（多进程并行）==========
    print(f"使用 {n_workers} 个进程进行并行处理...\n")

    # 准备参数列表
    args_list = [(data_dir_name, str(v_wave_data_root), w_c_threshold, V_target, zeta_target, f_s)
                 for data_dir_name in selected_dirs]

    # 使用多进程并行处理
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # 提交所有任务
        future_to_dir = {executor.submit(_process_one_dir, args): args[0]
                        for args in args_list}

        # 实时显示进度
        completed = 0
        total = len(selected_dirs)
        for future in future_to_dir:
            dir_name = future_to_dir[future]
            try:
                result = future.result()
                completed += 1
                if result is not None:
                    results.append(result)
                    print(f"[{completed}/{total}] 处理: {dir_name} ... 完成")
                else:
                    print(f"[{completed}/{total}] 处理: {dir_name} ... 跳过")
            except Exception as e:
                completed += 1
                print(f"[{completed}/{total}] 处理: {dir_name} ... 错误: {str(e)}")
    
    # ========== 保存结果到 CSV ==========
    if results:
        # 转换为 DataFrame
        df_results = pd.DataFrame(results)
        
        # 保存为 CSV
        os.makedirs(MODEL3_ANALYSIS_DIR, exist_ok=True)
        output_csv = MODEL3_ANALYSIS_DIR / "Model3_Unprovoke_Comparison_try1.csv"
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

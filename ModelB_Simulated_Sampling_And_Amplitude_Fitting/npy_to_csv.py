import numpy as np
import pandas as pd
import os

# ==========================================
# 请将这里的路径替换为你截图中的实际路径！
# 例如: r"你的绝对路径/V_Wave_Data/20260310_145055"
# ==========================================
data_dir = r"D:\PYTHON\layout design\ModelA_Virtual_Internal_Solitary_Wave_Data_Generation\V_Wave_Data_Hor\20260422_150147"

def convert_u_to_csv(folder_path):
    u_file = os.path.join(folder_path, 'U_Vel_3D.npy')
    z_file = os.path.join(folder_path, 'z.npy')
    
    if not os.path.exists(u_file):
        print(f"❌ 找不到文件: {u_file}")
        return

    print("[*] 正在读取 NumPy 二进制数据...")
    u_data = np.load(u_file)
    
    # 尝试加载深度信息 z.npy，如果没有就用索引号代替
    try:
        z_data = np.load(z_file)
    except FileNotFoundError:
        print("[!] 未找到 z.npy，将使用数组索引作为深度替代。")
        z_data = np.arange(len(u_data))

    print(f"[*] 成功加载！U_profile 数组维度为: {u_data.ndim} 维, 形状: {u_data.shape}")

    # 情况 A: 这只是一个一维的深度剖面 (随 Z 变化)
    if u_data.ndim == 1:
        print("[*] 诊断: 这是一个 1D 深度剖面。代表该水平流场在 X 和 Y 方向上是均匀的。")
        df = pd.DataFrame({
            'Depth_Z (m)': z_data,
            'U_Velocity (m/s)': u_data
        })
        out_name = os.path.join(folder_path, 'U_profile_1D_readable.csv')
        df.to_csv(out_name, index=False)
        print(f"✅ 转换完成！已保存为列表格式 CSV: {out_name}")

    # 情况 B: 这是一个三维数据场 (随 X, Y, Z 变化)
    elif u_data.ndim == 3:
        print("[*] 诊断: 这是一个 3D 速度场。准备提取 Y=0 (中心) 的 X-Z 切片...")
        x_grid = np.load(os.path.join(folder_path, 'x_grid.npy'))
        y_grid = np.load(os.path.join(folder_path, 'y_grid.npy'))
        
        # 寻找 Y 轴中心点的索引
        y_center_idx = np.argmin(np.abs(y_grid - 0)) 
        print(f"[*] 锁定 Y=0 切片 (索引: {y_center_idx}, 实际 Y 值: {y_grid[y_center_idx]})")
        
        # 提取切片 (假设 numpy 数组排列为 [X, Y, Z])
        # 如果你的数组是 [Z, Y, X]，请改为 u_slice = u_data[:, y_center_idx, :]
        u_slice = u_data[:, y_center_idx, :] 
        
        # 将数据转置，使其行是深度 Z，列是位置 X，更符合人类直觉的矩阵阅读习惯
        df = pd.DataFrame(u_slice.T, index=np.round(z_data, 1), columns=np.round(x_grid, 1))
        df.index.name = 'Depth_Z(m) \ X(m)'
        
        out_name = os.path.join(folder_path, 'U_profile_slice_Y0_readable.csv')
        df.to_csv(out_name)
        print(f"✅ 转换完成！已保存为二维矩阵 CSV: {out_name}")
        
    else:
        print(f"❌ 无法处理的维度: {u_data.ndim} 维。")

if __name__ == '__main__':
    # 填入你想要转换的那个具体时间戳文件夹的路径
    convert_u_to_csv(data_dir)
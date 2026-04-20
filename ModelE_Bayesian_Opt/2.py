import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ================= 配置路径 =================
# CSV 文件路径
csv_path = r"D:\PYTHON\layout design\ModelB_Simulated_Sampling_And_Amplitude_Fitting\Analysis_Results_SwA_Lagrangian_Cut_Data\analysis_results_swA_lagrangian_30cut.csv"
# V_Wave_Data 根目录 (用于提取温跃层深度)
v_wave_root = r"D:\PYTHON\layout design\ModelA_Virtual_Internal_Solitary_Wave_Data_Generation\V_Wave_Data"

# ================= 数据读取与匹配 =================
df = pd.read_csv(csv_path)

ther_depths = []
valid_indices = []

# 依然需要通过文件夹去拿 ther_depth 数据，但不再作为绘图展示内容
for idx, row in df.iterrows():
    wave_id = str(row['wave_id'])
    json_path = os.path.join(v_wave_root, wave_id, "params.json")
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            params = json.load(f)
            ther_depths.append(params.get('thermocline_depth', 0))
            valid_indices.append(idx)

# 构建仅包含 X, Y, Z 绘图所需数据的干净 DataFrame
plot_df = df.iloc[valid_indices].copy()
plot_df['thermocline_depth'] = ther_depths

# ================= 3D 绘图 =================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# X: 真实振幅, Y: 温跃层深度, Z: 相对误差
sc = ax.scatter(plot_df['true_h0'], 
                plot_df['thermocline_depth'], 
                plot_df['error_pct'], 
                c=plot_df['error_pct'], # 颜色映射依然根据误差大小
                cmap='coolwarm',        # 使用冷暖色调，高误差更刺眼(红色)
                s=60, alpha=0.8, edgecolors='w')

# 极简坐标轴设定
ax.set_xlabel('Amplitude (m)')
ax.set_ylabel('Thermocline Depth (m)')
ax.set_zlabel('Error (%)')
plt.title('Amplitude vs Thermocline Depth vs Error')

# 添加颜色条
cbar = plt.colorbar(sc, shrink=0.5, pad=0.1)
cbar.set_label('Error (%)')

# 调整初始视角，以便更好地观察 XY 平面上的变化对 Z 的影响
ax.view_init(elev=30, azim=45)

plt.tight_layout()
plt.show()
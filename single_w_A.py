import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# ==========================================
# 1. 设定相对路径并加载环境与物理场数据
# ==========================================
# z轴逻辑：海面为0，向下为正方向（0到1000米），z数组升序从0到1000
# 使用 os.path.join 保证在 Windows/Mac/Linux 路径符的兼容性
data_dir = os.path.join("v_wave_data", "20260310_145559")

print(f"正在从相对路径加载数据: {data_dir} ...")
z = np.load(os.path.join(data_dir, 'z.npy'))                # 深度网格，升序：0（海面）到1000（海底）
x_grid = np.load(os.path.join(data_dir, 'x_grid.npy'))      # X轴空间网格
y_grid = np.load(os.path.join(data_dir, 'y_grid.npy'))      # Y轴空间网格
W_Vel_3D = np.load(os.path.join(data_dir, 'W_Vel_3D.npy'))  # 3D 垂直流速场

with open(os.path.join(data_dir, 'params.json'), 'r') as f:
    params = json.load(f)

# 提取波速，以及最重要的水下滑翔机目标相遇深度
Cp = params.get('c0')
thermocline_depth = params.get('thermocline_depth') 

# ==========================================
# 2. 构建 3D 空间插值器 (Interpolator)
# ==========================================
# z轴升序，确保RegularGridInterpolator正常工作
# 如果数据中z是降序的（罕见），则翻转以匹配升序要求
if z[0] > z[-1]:
    z = np.flip(z)
    W_Vel_3D = np.flip(W_Vel_3D, axis=2)

interp_w = RegularGridInterpolator((x_grid, y_grid, z), W_Vel_3D, bounds_error=False, fill_value=0.0)

# ==========================================
# 3. 定义滑翔机运动与【最大振幅深度】精准交汇逻辑
# ==========================================
def sampler_position(t):
    """水平速度 0.22m/s，6000秒下潜到1000米，呈锯齿状"""
    x = 0.22 * t
    t_mod = t % 12000
    if t_mod < 6000:
        d = t_mod * 1000 / 6000
    else:
        d = 1000 - (t_mod - 6000) * 1000 / 6000
    return x, d

# 核心计算：因为下潜速度是 1/6 m/s，到达 thermocline_depth 深度所需的时间刚好是 深度 * 6
t_meet = thermocline_depth * 6.0
x_g_meet, z_g_meet = sampler_position(t_meet)

print(f"对齐参数: 内孤立波最大振幅深度={thermocline_depth:.2f}m")
print(f"对齐参数: 滑翔机将在 t={t_meet:.1f}s 时到达该深度并与波峰相遇")

# 计算内孤立波的初始位置，确保在 t=t_meet 时，波峰恰好移动到滑翔机的X坐标处
x_init = x_g_meet + Cp * t_meet #调整相遇在x的位置,找找误差图
y_center = 0.0 

# ==========================================
# 4. 生成时间序列并执行纯净虚拟采样 (无噪声)
# ==========================================
# 自适应时间窗口：以相遇时间为中心，前后各取 1500 秒 (50分钟窗口)
start_time = max(0, t_meet - 1500)
end_time = t_meet + 1500
t_array = np.arange(start_time, end_time, 5) # 模拟每 5 秒采样一次

w_obs = np.zeros_like(t_array, dtype=float)
depth_obs = np.zeros_like(t_array, dtype=float)

for i, t in enumerate(t_array):
    x_g, z_g = sampler_position(t)
    depth_obs[i] = z_g
    
    # 时空转换(伽利略变换)：抵消内孤立波的传播速度
    X_wave_current = x_init - Cp * t
    x_eff = x_g - X_wave_current # 计算滑翔机相对于波峰的有效水平位置
    
    # 直接提取插值流速，【不加入任何高斯白噪声】
    w_obs[i] = interp_w((x_eff, y_center, z_g))

# ==========================================
# 5. 复刻 Anatomy Fig 3 绘图
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 6))

# 右侧 Y 轴：绘制滑翔机深度 (红线)
ax2 = ax1.twinx()
ax2.plot(t_array, depth_obs, color='tomato', linewidth=1.5, label='Glider Depth')
ax2.set_ylim(1000, 0)  # 翻转深度轴
ax2.set_ylabel('Depth (m)', color='tomato', fontsize=12)

# 左侧 Y 轴：绘制垂直水速 (深蓝线)
# 取消了marker，使用纯净的实线展现完美的无噪声物理波形
ax1.plot(t_array, w_obs, color='#005b96', linestyle='-', linewidth=2)

# 阴影填充：复刻 Anatomy 中下沉涂蓝、上升涂粉的效果
ax1.fill_between(t_array, 0, w_obs, where=(w_obs < 0), color='#6b9ac4', alpha=0.5) 
ax1.fill_between(t_array, 0, w_obs, where=(w_obs > 0), color='#f4a1c1', alpha=0.5)

# 标记 t_w0 零点和参考线
ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax1.axvline(t_meet, color='black', linestyle=':', linewidth=1.5, alpha=0.6)
ax1.text(t_meet + 20, max(w_obs)*0.8, f'Peak Encounter\nDepth: {thermocline_depth:.1f}m', color='black')

ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel(r'$w_{isw}$ (m s$^{-1}$)', color='#005b96', fontsize=12)
ax1.set_title('Reconstruction of Anatomy Fig 3 (Pure KdV Signal at Max Amplitude Depth)', fontsize=14)

# 动态调整左侧 Y 轴的上下限，使其对称美观
w_max = max(abs(np.min(w_obs)), abs(np.max(w_obs))) * 1.2
if w_max > 0:
    ax1.set_ylim(-w_max, w_max)

plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

import numpy as np

print("\n--- 开始执行直接积分法求最大振幅 (dh) ---")

# 1. 锁定相遇时间附近的有效窗口 (前后 1500 秒)
mask = (t_array > start_time) & (t_array < end_time)
t_win = t_array[mask]
w_win = w_obs[mask]

# ==========================================
# 核心修复：直接锁定“向上水流（w > 0）”的核心主峰
# 无论波形是先上后下，还是先下后上，寻找最大正向水流并向两侧扩张，
# 即可绝对安全地包络住整个上升阶段。
# ==========================================
max_w_idx = np.argmax(w_win)

# 2. 向前(左)回溯，寻找积分起点 t_w0 (水流刚开始变为向上的时刻)
tw0_idx = max_w_idx
while tw0_idx > 0 and w_win[tw0_idx] > 0:
    tw0_idx -= 1
t_w0 = t_win[tw0_idx]

# 3. 向后(右)推进，寻找积分终点 t_U (水流重新停止向上的时刻)
tu_idx = max_w_idx
while tu_idx < len(w_win) - 1 and w_win[tu_idx] > 0:
    tu_idx += 1
t_U = t_win[tu_idx]

# 4. 截取精确的单峰积分区间
t_integral = t_win[tw0_idx:tu_idx]
w_integral = w_win[tw0_idx:tu_idx]

# 5. 数值积分：使用 numpy 原生的梯形积分法则 (最稳定，完全规避 scipy bug)
dh = np.trapezoid(w_integral, x=t_integral)  

# ==========================================
# 评价报告
# ==========================================
true_h0 = params['h0']
error_pct = np.abs(dh - true_h0) / true_h0 * 100

print(f"积分区间锁定: 从 {t_w0:.1f} s 到 {t_U:.1f} s")
print(f"提取的有效向上水流数据点数: {len(w_integral)} 个")
print("\n================ 最终观测评价报告 ==================")
print(f"【真实基准值】 Ground Truth 最大振幅 h0 = {true_h0:.2f} m")
print(f"【积分观测值】 Inferred     最大振幅 dh = {dh:.2f} m")
print(f"【测量绝对误差】 Error = {np.abs(dh - true_h0):.2f} m ({error_pct:.2f}%)")
print("====================================================")
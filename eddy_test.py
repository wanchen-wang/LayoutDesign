import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 第 1 步：生成观测网络坐标 (x_r)
# ==========================================
# 设定网格大小和传感器间距
N_points = 4
spacing = 1.0  # 假设相邻传感器之间的距离为 1 个单位

# 生成 x, y, z 三个维度的坐标刻度
x_ticks = np.linspace(0, (N_points - 1) * spacing, N_points)
y_ticks = np.linspace(0, (N_points - 1) * spacing, N_points)
z_ticks = np.linspace(0, (N_points - 1) * spacing, N_points)

# 使用 meshgrid 生成 3D 网格
X, Y, Z = np.meshgrid(x_ticks, y_ticks, z_ticks, indexing='ij')

# 将网格展平，得到一个 N x 3 的矩阵。
# 每一行代表一个传感器的 [x, y, z] 坐标。总共有 4x4x4 = 64 行。
obs_coords = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

N_obs = obs_coords.shape[0]
print(f"成功生成观测网络！共有 {N_obs} 个传感器。")
print(f"前 5 个传感器的坐标为:\n{obs_coords[:5]}")

# ==========================================
# 第 2 步：设定物理场的统计假设条件
# ==========================================
# 参考文献第二章和图2、图3的示例，设定相关参数
R_c = 1.5  # 空间相关尺度 (Signal scale)。设为 1.5 意味着在距离 1.5 个单位内，温度比较相似。
E = 0.1    # 测量仪器的随机误差方差 (Noise variance)。代表仪器不完美。

# ==========================================
# 第 3 步：定义核心的“协方差函数” F(d)
# ==========================================
# 根据文献公式 (43b) 的简化形式（假设噪声完全是仪器带来的，即Rn趋于0时的局部白噪声）
def covariance_function(distance, R_c):
    """
    计算两个点之间的信号协方差。
    distance: 两点之间的欧几里得距离
    R_c: 空间相关尺度
    """
    return np.exp(-(distance**2) / (R_c**2))

print(f"\n物理场设定完毕: 相关尺度 R_c = {R_c}, 仪器误差 E = {E}")

from scipy.spatial import distance_matrix

# ==========================================
# 第 4 步：生成模拟的观测读数 (phi_s)
# ==========================================
# 因为我们没有真实的海洋数据，需要先模拟出这 64 个传感器的读数。
# 假设真实的温度场是一个简单的空间波动，测到的数据是真实温度加上随机误差。
np.random.seed(42)  # 固定随机种子，保证你我每次跑出的结果一样

# 假设真实温度随着空间坐标变化（比如一个简单的正弦/余弦场）
true_temp_at_obs = np.sin(obs_coords[:, 0]) + np.cos(obs_coords[:, 1])

# 加上均值为0，方差为E的高斯白噪声，代表仪器误差
instrument_noise = np.random.normal(0, np.sqrt(E), size=N_obs)

# 这就是我们手头拿到的最终观测数据（对应文献中的 phi_s）
phi_s = true_temp_at_obs + instrument_noise


# ==========================================
# 第 5 步：计算观测点协方差矩阵 A (对应文献公式 8)
# ==========================================
# 1. 计算这 64 个传感器两两之间的距离矩阵 (大小为 64 x 64)
dist_matrix = distance_matrix(obs_coords, obs_coords)

# 2. 将距离矩阵代入协方差函数，计算信号的空间相关性
signal_covariance = covariance_function(dist_matrix, R_c)

# 3. 加上对角线上的仪器误差 E（因为不同仪器的噪音互不相关，所以只加在对角线上）
# 这里的 np.eye(64) 生成了一个对角线为1，其余为0的单位矩阵（即文献中的克罗内克 delta 符号）
A_matrix = signal_covariance + E * np.eye(N_obs)


# ==========================================
# 第 6 步：计算观测点的全局权重向量 eta (对应文献公式 11)
# ==========================================
# 1. 求矩阵 A 的逆矩阵 (A^{-1})
A_inv = np.linalg.inv(A_matrix)

# 2. 用 A 的逆矩阵乘以我们的观测数据向量 phi_s
eta = np.dot(A_inv, phi_s)

print(f"成功计算协方差矩阵 A！它的形状是: {A_matrix.shape}")
print(f"成功计算全局权重向量 eta！它的形状是: {eta.shape}\n")

print(f"前 3 个传感器的观测读数 (phi_s) 为: {phi_s[:3].round(3)}")
print(f"前 3 个传感器提炼出的权重 (eta) 为: {eta[:3].round(3)}")

# ==========================================
# 第二阶段：开始预测 (对应文献 2.1 节)
# ==========================================

# 第 7 步：指定一个你想预测的“未知目标点” x
# 比如我们选在网格正中间的一个空隙位置
target_point = np.array([[1.5, 1.5, 1.5]]) 

# 第 8 步：计算目标点与所有 64 个观测点之间的距离
# 结果会是一个 1 x 64 的矩阵
dist_to_obs = distance_matrix(target_point, obs_coords) 

# 第 9 步：计算协方差向量 C (对应文献公式 9)
# 将距离代入协方差函数，计算目标点和 64 个观测点的空间相关性
C_vector = covariance_function(dist_to_obs, R_c) 

# 第 10 步：计算目标点的最优温度估计值 (对应文献公式 12)
# 把目标点协方差向量 C_vector 和 之前算好的权重 eta 做点乘
estimated_temp = np.dot(C_vector, eta)

# 第 11 步：计算该点预测值的误差方差 (对应文献公式 10)
# 公式: 误差方差 = C_xx - sum( C_xr * C_xs * A_rs^-1 )

# 1. C_xx 是目标点自身的基础方差（在没有任何观测数据时的盲猜误差）。
# 根据我们的协方差函数 exp(-d^2/R_c^2)，当距离 d=0 时，F(0) = 1.0
C_xx = 1.0 

# 2. 计算观测数据帮我们“消除”掉的误差。
# 在线性代数中，求和项 sum( C_xr * C_xs * A_rs^-1 ) 可以写成矩阵连乘： C * A_inv * C^T
error_reduction = np.dot(np.dot(C_vector, A_inv), C_vector.T)

# 3. 得出最终的预期误差
expected_error_variance = C_xx - error_reduction

# 将可能是 ndarray 的结果提取为标量以便格式化输出
est_scalar = float(np.squeeze(estimated_temp))
err_var_scalar = float(np.squeeze(expected_error_variance))

print(f"目标点坐标: {target_point}")
print(f"--> 【预测结果】最优温度估计值 (theta_hat): {est_scalar:.4f}")
print(f"--> 【置信度】预测的误差方差 (Error Variance): {err_var_scalar:.4f}")
print(f"    (注：在没有数据前，盲猜的原始误差是 1.0000。现在降到了 {err_var_scalar:.4f})")

# ==========================================
# 第三阶段：生成二维切片并绘制地图 (对应文献中的 Objective/Error Maps)
# ==========================================

# 第 12 步：定义一个高分辨率的二维预测网格
# 我们固定 Z = 1.5 (网格正中间高度)，让 X 和 Y 在 0 到 3 之间密集取样
grid_resolution = 50  # 50x50 的网格分辨率
x_pred = np.linspace(0, 3, grid_resolution)
y_pred = np.linspace(0, 3, grid_resolution)
X_pred, Y_pred = np.meshgrid(x_pred, y_pred)

# 将高度 Z_pred 全部固定为 1.5
Z_pred = np.full_like(X_pred, 1.5) 

# 把网格展平，得到 2500 个需要预测的 [x, y, z] 坐标点
pred_coords = np.vstack([X_pred.ravel(), Y_pred.ravel(), Z_pred.ravel()]).T

# 初始化存储预测温度和误差方差的数组（每个点一个标量）
predicted_field = np.zeros(pred_coords.shape[0])
error_field = np.zeros(pred_coords.shape[0])

# 第 13 步：对网格上的每一个点进行批量预测和误差评估
print("正在生成二维预测切片，请稍候...")
for i, target in enumerate(pred_coords):
    # 1. 算距离: 目标点和 64 个观测点的距离矩阵 (1 x 64)
    dist = distance_matrix([target], obs_coords)
    
    # 2. 算协方差 C_vector (公式 9)
    C_vec = covariance_function(dist, R_c)
    
    # 3. 算出温度预测值 (公式 12)
    predicted_field[i] = float(np.squeeze(np.dot(C_vec, eta)))
    
    # 4. 算出误差方差 (公式 10)
    C_xx = 1.0  # 自身方差
    error_reduction = np.dot(np.dot(C_vec, A_inv), C_vec.T)
    error_field[i] = float(np.squeeze(C_xx - error_reduction))

# 把一维的结果重新折叠回 50x50 的二维网格形状，准备画图
predicted_field_2d = predicted_field.reshape(grid_resolution, grid_resolution)
error_field_2d = error_field.reshape(grid_resolution, grid_resolution)

# ==========================================
# 第 14 步：绘制结果图！
# ==========================================
# 我们为了简化画面，把所有 64 个观测点的 X、Y 坐标投影到图上作为参考
obs_xy = obs_coords[:, :2] 

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ---- 图 1: 最优温度预测地图 (Objective Map) ----
cf1 = ax1.contourf(X_pred, Y_pred, predicted_field_2d, cmap='coolwarm', levels=30)
plt.colorbar(cf1, ax=ax1, label='Estimated Temperature')
# 画出传感器位置的投影
ax1.scatter(obs_xy[:, 0], obs_xy[:, 1], c='black', marker='x', alpha=0.5, label='Sensors (Projected)')
ax1.set_title("Objective Analysis: Temperature Field (Z=1.5)")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.legend()

# ---- 图 2: 预测误差方差地图 (Error Map) ----
cf2 = ax2.contourf(X_pred, Y_pred, error_field_2d, cmap='viridis_r', levels=30)
plt.colorbar(cf2, ax=ax2, label='Error Variance (Confidence)')
# 添加等高线，让它看起来更像文献中的图 (如 Fig. 1)
contours = ax2.contour(X_pred, Y_pred, error_field_2d, colors='black', linewidths=0.5, levels=8)
ax2.clabel(contours, inline=True, fontsize=8)
ax2.scatter(obs_xy[:, 0], obs_xy[:, 1], c='red', marker='x', alpha=0.5)
ax2.set_title("Expected Error Map")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")

plt.tight_layout()
plt.show()
print("绘图完成！")

# ==========================================
# 矢量场进阶：构建无辐散流场的 2Nx2N 协方差矩阵
# (对应文献 2.5 节 Vector Fields)
# ==========================================

# 我们继续使用之前生成的 64 个观测点位置 (obs_coords) 和 N_obs=64
# 并且假定只在二维平面 (Z=1.5) 上有速度分量，简化距离计算
xy_coords = obs_coords[:, :2] 

# 定义基础的流函数协方差 F(r) 及其导数推导出的 R(r) 和 S(r)
# 假设 F(r) = exp(-r^2 / R_c^2)
def calc_R_S(distance, R_c):
    """
    根据文献公式 28d, 28e 计算纵向协方差 R 和 横向协方差 S
    注意：为了避免距离为 0 时除以 0，这里需要进行极限处理
    """
    R_val = np.zeros_like(distance)
    S_val = np.zeros_like(distance)
    
    # 找出距离不为 0 的点
    nz = distance > 1e-10
    r_nz = distance[nz]
    
    # 根据 F(r) 的导数解析推导出的 R 和 S 公式 (数学求导结果)
    # R(r) = -(1/r) * dF/dr = (2/R_c^2) * exp(-r^2/R_c^2)
    R_val[nz] = (2.0 / R_c**2) * np.exp(-(r_nz**2) / R_c**2)
    
    # S(r) = -d^2F/dr^2 = (2/R_c^2 - 4*r^2/R_c^4) * exp(-r^2/R_c^2)
    S_val[nz] = (2.0 / R_c**2 - 4.0 * r_nz**2 / R_c**4) * np.exp(-(r_nz**2) / R_c**2)
    
    # 处理距离为 0 的自身协方差 (取 r->0 时的极限)
    R_val[~nz] = 2.0 / R_c**2
    S_val[~nz] = 2.0 / R_c**2
    
    return R_val, S_val

# 计算所有观测点两两之间的 dx, dy 和 距离 rho
# 通过广播机制生成 N x N 的坐标差矩阵
dx = xy_coords[:, 0:1] - xy_coords[:, 0:1].T  
dy = xy_coords[:, 1:2] - xy_coords[:, 1:2].T  
rho = np.sqrt(dx**2 + dy**2)

# 计算方向余弦 chi_1 和 chi_2 (文献 28b 之前的公式)
chi_1 = np.zeros_like(rho)
chi_2 = np.zeros_like(rho)
nz = rho > 1e-10
chi_1[nz] = dx[nz] / rho[nz]
chi_2[nz] = dy[nz] / rho[nz]

# 计算 R(rho) 和 S(rho)
R_mat, S_mat = calc_R_S(rho, R_c)

# ------------------------------------------
# 开始拼装 2N x 2N 的大分块矩阵 A
# ------------------------------------------
# 1. u 和 u 的协方差块 (A_uu)
A_uu = chi_1**2 * (R_mat - S_mat) + S_mat

# 2. v 和 v 的协方差块 (A_vv)
A_vv = chi_2**2 * (R_mat - S_mat) + S_mat

# 3. u 和 v 的交叉协方差块 (A_uv 和 A_vu，它们互为转置)
A_uv = chi_1 * chi_2 * (R_mat - S_mat)
A_vu = A_uv.T

# 使用 np.block 将四个 N x N 的矩阵拼成 2N x 2N 的大矩阵
A_vector_field = np.block([
    [A_uu, A_uv],
    [A_vu, A_vv]
])

# 最后，在对角线上加上仪器测量误差 E (模拟独立仪器的白噪声)
# 注意现在是 2N 个数据，所以要加在 128x128 的对角线上
A_vector_field += E * np.eye(2 * N_obs)

print(f"成功构建矢量场协方差矩阵 A！它的形状是: {A_vector_field.shape}")

# ==========================================
# 第四阶段：从速度场直接预测流函数 (对应文献公式 30)
# (重现文献中 Fig 5a 与 Fig 5b 的奇迹)
# ==========================================

# 第 1 步：生成模拟的“真实流函数”和“观测速度”
# 为了验证我们算得准不准，我们先假设真实的流函数是 psi(x,y) = sin(x) * cos(y)
# 根据无辐散物理定律： u = -d(psi)/dy,  v = d(psi)/dx
# 我们算出 64 个传感器所在位置的真实流速：
true_u_at_obs = np.sin(xy_coords[:, 0]) * np.sin(xy_coords[:, 1])
true_v_at_obs = np.cos(xy_coords[:, 0]) * np.cos(xy_coords[:, 1])

# 加上高斯白噪声，模拟不完美的流速计
obs_u = true_u_at_obs + np.random.normal(0, np.sqrt(E), N_obs)
obs_v = true_v_at_obs + np.random.normal(0, np.sqrt(E), N_obs)

# 把所有的 u 和 v 拼接成一个长度为 128 的超级数据向量 (文献中的 phi_s)
phi_vector = np.concatenate([obs_u, obs_v])


# 第 2 步：计算矢量场的全局权重 eta (共 128 个)
# 对应文献公式 (30) 中的核心操作：求 A_rs 的逆并乘以 phi_s
A_vec_inv = np.linalg.inv(A_vector_field)
eta_vector = np.dot(A_vec_inv, phi_vector)

# 把权重拆分为 u 的部分和 v 的部分，方便一会儿做加法
eta_u = eta_vector[:N_obs]
eta_v = eta_vector[N_obs:]


# 第 3 步：定义目标点(标量流函数)与观测点(矢量速度)之间的协方差 P
def calc_P_covariance(target_pt, obs_pts, R_c):
    """
    对应文献公式 (31) 之后的等式：
    P_u = - gamma_2 * (dF/drho)
    P_v =   gamma_1 * (dF/drho)
    """
    dx = obs_pts[:, 0] - target_pt[0]
    dy = obs_pts[:, 1] - target_pt[1]
    rho = np.sqrt(dx**2 + dy**2)
    
    # 基础的流函数协方差 F(rho) = exp(-rho^2 / R_c^2)
    F_val = np.exp(-(rho**2) / R_c**2)
    
    # 对 F(rho) 求导，dF/drho = (-2*rho / R_c^2) * F_val
    # 代入 gamma_1 = dx/rho, gamma_2 = dy/rho 之后，奇迹般地抵消了分母的 rho：
    P_u = (2.0 * dy / R_c**2) * F_val
    P_v = (-2.0 * dx / R_c**2) * F_val
    
    return P_u, P_v


# 第 4 步：在 2D 网格上批量预测流函数
grid_res = 40
x_grid = np.linspace(0, 3, grid_res)
y_grid = np.linspace(0, 3, grid_res)
X_g, Y_g = np.meshgrid(x_grid, y_grid)

predicted_psi = np.zeros((grid_res, grid_res))
true_psi = np.zeros((grid_res, grid_res))

print("正在吸入 128 个速度数据，反推全局流函数地图，请稍候...")
for i in range(grid_res):
    for j in range(grid_res):
        target_xy = np.array([X_g[i, j], Y_g[i, j]])
        
        # 记录该点的真实流函数（仅用来对比，算法其实“不知道”这个值）
        true_psi[i, j] = np.sin(target_xy[0]) * np.cos(target_xy[1])
        
        # 1. 计算目标点与所有观测点的相互关系 P_u 和 P_v
        P_u, P_v = calc_P_covariance(target_xy, xy_coords, R_c)
        
        # 2. 累加权重得出预测值 (严格对应文献公式 30)
        # psi_hat = sum(P_u * eta_u) + sum(P_v * eta_v)
        predicted_psi[i, j] = np.dot(P_u, eta_u) + np.dot(P_v, eta_v)


# 第 5 步：绘制华丽的对比图！
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ---- 图 1: 真实情况 (类似文献 Fig 5a) ----
cf1 = ax1.contourf(X_g, Y_g, true_psi, cmap='RdBu_r', levels=20)
# 用 quiver 画出观测点上真实的流速箭头
ax1.quiver(xy_coords[:, 0], xy_coords[:, 1], true_u_at_obs, true_v_at_obs, color='black', scale=15)
ax1.set_title("True Stream Function & True Velocities (Like Fig. 5a)")
plt.colorbar(cf1, ax=ax1)

# ---- 图 2: 客观分析预测结果 (类似文献 Fig 5b) ----
cf2 = ax2.contourf(X_g, Y_g, predicted_psi, cmap='RdBu_r', levels=20)
# 用 quiver 画出带噪音的观测流速箭头 (这就是算法唯一拥有的输入)
ax2.quiver(xy_coords[:, 0], xy_coords[:, 1], obs_u, obs_v, color='black', scale=15)
ax2.set_title("Objective Map Predicted from Velocities (Like Fig. 5b)")
plt.colorbar(cf2, ax=ax2)

plt.tight_layout()
plt.show()

# ==========================================
# 补充：计算并绘制流函数的误差地图 (对应文献公式 35 和 Fig. 10)
# ==========================================

# 1. 在初始化预测矩阵时，多加一个存储误差的矩阵
predicted_psi = np.zeros((grid_res, grid_res))
true_psi = np.zeros((grid_res, grid_res))
psi_error_field = np.zeros((grid_res, grid_res)) # [新增] 用于存储流函数的误差方差

print("正在吸入 128 个速度数据，反推全局流函数及误差地图，请稍候...")
for i in range(grid_res):
    for j in range(grid_res):
        target_xy = np.array([X_g[i, j], Y_g[i, j]])
        
        # 记录该点的真实流函数（仅用来对比）
        true_psi[i, j] = np.sin(target_xy[0]) * np.cos(target_xy[1])
        
        # 计算目标点与所有观测点的相互关系 P_u 和 P_v
        P_u, P_v = calc_P_covariance(target_xy, xy_coords, R_c)
        
        # 累加权重得出预测值
        predicted_psi[i, j] = np.dot(P_u, eta_u) + np.dot(P_v, eta_v)
        
        # ------------------------------------------
        # [新增] 计算该点流函数的预期误差 (对应公式 35)
        # ------------------------------------------
        # a. 将 P_u 和 P_v 拼接成一个长度为 128 的向量 P_vec
        P_vec = np.concatenate([P_u, P_v])
        
        # b. 计算观测数据消除的误差: P_vec * A_inv * P_vec^T
        error_reduction = np.dot(np.dot(P_vec, A_vec_inv), P_vec.T)
        
        # c. 用初始基础方差 1.0 减去消除的误差
        psi_error_field[i, j] = 1.0 - error_reduction


# ==========================================
# [新增] 第 6 步：绘制流函数的误差地图
# ==========================================
fig3, ax3 = plt.subplots(figsize=(7, 6))

# 绘制误差等值线 (类似文献中的 Fig. 10)
cf3 = ax3.contourf(X_g, Y_g, psi_error_field, cmap='viridis_r', levels=30)
plt.colorbar(cf3, ax=ax3, label='Stream Function Error Variance')

# 画出黑色等高线让层次更清晰
contours_err = ax3.contour(X_g, Y_g, psi_error_field, colors='black', linewidths=0.5, levels=10)
ax3.clabel(contours_err, inline=True, fontsize=8)

# 标记出传感器所在的位置
ax3.scatter(xy_coords[:, 0], xy_coords[:, 1], c='red', marker='x', alpha=0.8, label='Velocity Sensors')

ax3.set_title("Expected Error Map of Stream Function (Like Fig. 10)")
ax3.set_xlabel("X")
ax3.set_ylabel("Y")
ax3.legend()

plt.tight_layout()
plt.show()
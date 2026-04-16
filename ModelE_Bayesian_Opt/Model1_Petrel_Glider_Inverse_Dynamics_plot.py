import numpy as np
import matplotlib.pyplot as plt
# 假设前面的逆向动力学解算程序保存为 PetrelGliderInverseDynamics 类
# 如果在同一个文件中，可直接运行；如果在不同文件，请使用: 
# from your_module import PetrelGliderInverseDynamics
from Model1_Petrel_Glider_Inverse_Dynamics import PetrelGliderInverseDynamics
# 1. 实例化我们前面写好的逆向动力学解算器
petrel = PetrelGliderInverseDynamics()

# 2. 数据生成：直接遍历滑翔角（包含无法配平的 0° 附近区域）
zeta_array = np.linspace(-60, 60, 2000)
theta_list = []
alpha_list = []

# 3. 循环调用模块一的解算核心
for zeta in zeta_array:
    try:
        # 直接调用写好的解析函数
        theta_rad, alpha_rad = petrel.solve_angles_analytical(zeta)
        theta_list.append(np.degrees(theta_rad))
        alpha_list.append(np.degrees(alpha_rad))
    except ValueError:
        # 巧妙利用我们在模块一中写的物理失速报错，将无解区域置为 NaN
        theta_list.append(np.nan)
        alpha_list.append(np.nan)

# 4. 绘制图表 (完美复刻图3-1)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(8, 6))

# 由于数据中包含了 NaN，Matplotlib 会自动将曲线在失速死区断开
plt.plot(theta_list, alpha_list, 'k-', linewidth=2, label='解析稳态滑翔曲线')

plt.title('俯仰角与攻角关系图 ', fontsize=14)
plt.xlabel(r'俯仰角 $\theta$ (°)', fontsize=12)
plt.ylabel(r'攻角 $\alpha$ (°)', fontsize=12)

# 限制坐标轴范围以匹配原论文比例
plt.xlim(-60, 60)
plt.ylim(-15, 15)

# 设置更详细的刻度
plt.xticks(np.arange(-60, 61, 10))
plt.yticks(np.arange(-15, 16, 2.5))

# 添加主次网格虚线
plt.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.7)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.5)

plt.axhline(0, color='gray', linewidth=1)
plt.axvline(0, color='gray', linewidth=1)
plt.legend(loc='upper left')
plt.tight_layout()

plt.show()

# 1. 生成攻角数组
# 依据海试中经济滑翔状态的实际范围，攻角通常在 ±6° 以内
# 这里我们生成 -7° 到 7° 的数组，并转化为论文图谱中使用的弧度(rad)单位
alpha_deg_array = np.linspace(-7, 7, 200)
alpha_rad_array = np.radians(alpha_deg_array)

# 2. 核心计算：直接调用模块一中写好的 calc_hydrodynamic_coeffs 函数
# 得益于 numpy 的矢量化运算能力，我们可以直接传入整个数组
C_L_array, C_D_array, C_M_array = petrel.calc_hydrodynamic_coeffs(alpha_rad_array)

# 3. 绘制图表 (1x3 并排子图)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# --- 绘制图2-21：升力系数与攻角关系 ---
axes[0].plot(alpha_rad_array, C_L_array, 'b-', linewidth=2, label='升力系数 $C_L$')
axes[0].set_title('升力系数与攻角关系', fontsize=14)
axes[0].set_xlabel(r'攻角 $\alpha$ (rad)', fontsize=12)
axes[0].set_ylabel(r'$C_L$', fontsize=12)
axes[0].grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.7)
axes[0].minorticks_on()
axes[0].grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.5)
axes[0].legend(loc='upper left')

# --- 绘制图2-22：阻力系数与攻角关系 ---
axes[1].plot(alpha_rad_array, C_D_array, 'r-', linewidth=2, label='阻力系数 $C_D$')
axes[1].set_title('阻力系数与攻角关系', fontsize=14)
axes[1].set_xlabel(r'攻角 $\alpha$ (rad)', fontsize=12)
axes[1].set_ylabel(r'$C_D$', fontsize=12)
axes[1].grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.7)
axes[1].minorticks_on()
axes[1].grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.5)
axes[1].legend(loc='upper center')

# --- 绘制图2-23：俯仰力矩系数与攻角关系 ---
axes[2].plot(alpha_rad_array, C_M_array, 'g-', linewidth=2, label='俯仰力矩系数 $C_M$')
axes[2].set_title('俯仰水动力矩系数与攻角关系', fontsize=14)
axes[2].set_xlabel(r'攻角 $\alpha$ (rad)', fontsize=12)
axes[2].set_ylabel(r'$C_M$', fontsize=12)
axes[2].grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.7)
axes[2].minorticks_on()
axes[2].grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.5)
axes[2].legend(loc='upper left')

# 调整子图间距并显示
plt.tight_layout()
plt.show()
import numpy as np

class PetrelGliderInverseDynamics:
    def __init__(self):
        # 1. "海燕-II" 基础物理参数设计
        self.mass_total = 70.0            # 滑翔机总质量 (kg)
        self.mass_p = 16.0                # 姿态调节单元(电池包)质量 (kg)
        self.diameter = 0.22              # 滑翔机最大直径 (m)
        self.Area = np.pi * (self.diameter / 2)**2  
        self.g = 9.81                     
        self.rho = 1024.0                 
        self.M_static = 0.0               
        
    def calc_hydrodynamic_coeffs(self, alpha_rad):
        """计算水动力系数 (标准未搭载EVS设备)"""
        C_L = 12.13 * alpha_rad + 0.0512
        C_D = 8.514 * (alpha_rad**2) + 0.318
        C_M = 10.09 * alpha_rad - 0.019
        return C_L, C_D, C_M

    def solve_angles_analytical(self, zeta_target_deg):
        """
        【彻底重构的解算器：摒弃错误的经验公式，采用严密的物理受力解析解】
        核心依据：稳态滑翔时阻力与升力之比严格等于滑翔角的负正切 C_D / C_L = -tan(zeta)
        """
        zeta_rad = np.radians(zeta_target_deg)
        tan_zeta = np.tan(zeta_rad)

        # 构建一元二次方程: a*alpha^2 + b*alpha + c = 0
        a = 8.514
        b = 12.13 * tan_zeta
        c = 0.318 + 0.0512 * tan_zeta

        # 求解判别式
        delta = b**2 - 4 * a * c
        if delta < 0:
            raise ValueError(f"无法在滑翔角 {zeta_target_deg}° 下实现物理配平 (失速或无解)")

        # 求根：流体力学中取绝对值较小的那一个作为实际未失速的攻角
        alpha1 = (-b + np.sqrt(delta)) / (2 * a)
        alpha2 = (-b - np.sqrt(delta)) / (2 * a)
        alpha_rad = alpha1 if abs(alpha1) < abs(alpha2) else alpha2
        
        # 严密的几何关系：俯仰角 = 滑翔角 + 攻角
        theta_rad = zeta_rad + alpha_rad
        
        return theta_rad, alpha_rad

    def inverse_kinematics_solver(self, V_target, zeta_target_deg):
        """主程序：通过目标速度和轨迹直接反推所需的油量和电池位移"""
        # 步骤 1：使用严密的解析方程求出俯仰角与攻角
        theta_rad, alpha_rad = self.solve_angles_analytical(zeta_target_deg)
        
        # 步骤 2：计算升力、阻力与俯仰力矩
        C_L, C_D, C_M = self.calc_hydrodynamic_coeffs(alpha_rad)
        dynamic_pressure = 0.5 * self.rho * (V_target**2) * self.Area
        L = dynamic_pressure * C_L
        D = dynamic_pressure * C_D
        M_DL = dynamic_pressure * C_M 

        # 步骤 3：受力平衡反推所需净浮力
        Delta_B = - (L * np.cos(alpha_rad) + D * np.sin(alpha_rad)) / np.cos(theta_rad)
        
        # 步骤 4：力矩平衡反推所需电池包位移
        r_p1 = - (M_DL + self.M_static) / (self.mass_p * self.g * np.cos(theta_rad))
        
        return {
            "目标速度 (m/s)": round(V_target, 3),
            "目标滑翔角 (°)": round(zeta_target_deg, 2),
            "解算俯仰角_theta (°)": round(np.degrees(theta_rad), 2),
            "解析攻角_alpha (°)": round(np.degrees(alpha_rad), 2),
            "所需净浮力_Delta_B (N)": round(Delta_B, 2),
            "所需电池包位移_rp1 (m)": round(r_p1, 4)
        }

if __name__ == "__main__":
    petrel = PetrelGliderInverseDynamics()
    # 我们测试一个标准的下潜目标滑翔角 -45度
    results = petrel.inverse_kinematics_solver(V_target=0.35, zeta_target_deg=-45.0)
    for k, v in results.items():
        print(f"{k}: {v}")
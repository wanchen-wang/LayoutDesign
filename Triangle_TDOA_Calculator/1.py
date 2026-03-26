import os
import json
import numpy as np

# ==========================================
# 1. 配置读取模块
# ==========================================
class ConfigManager:
    @staticmethod
    def load_params(json_path):
        """读取 V_Wave_Data_Line 目录下的 params.json"""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"找不到参数文件: {json_path}")
        with open(json_path, 'r') as f:
            params = json.load(f)
        return params

# ==========================================
# 2. 阵列规划与运动学反推模块 (Step 1)
# ==========================================
class DeploymentPlanner:
    def __init__(self, L, H, v_g, v_z, z_max):
        self.L = L            # 底边长 (y方向间距)
        self.H = H            # 高 (x方向间距)
        self.v_g = v_g        # 滑翔机水平速度 (与波法向反向迎面)
        self.v_z = v_z        # 滑翔机垂直下潜速度
        self.z_max = z_max    # 最大振幅深度 (即相遇深度)

    def get_local_encounter_points(self):
        """获取在 z_max 深度处，构成等腰三角形的局部相遇点坐标"""
        # 节点1(底边下), 节点2(底边上), 节点3(尖端前锋)
        return {
            1: np.array([0, -self.L / 2.0]),
            2: np.array([0, self.L / 2.0]),
            3: np.array([self.H, 0])
        }

    def calculate_surface_deployment(self, t_encounter_dict, theta_true_rad):
        """反推海面初始发车时间 T0 和 全局坐标 X0, Y0 (为后续对接 single_W_A_Lagrangian 准备)"""
        # 下潜耗时
        t_dive = self.z_max / self.v_z 
        deploy_cmds = {}
        
        for idx, t_enc in t_encounter_dict.items():
            # 1. 初始发车时间
            T0 = t_enc - t_dive
            # 2. 这里的坐标反推留作下一步接口：
            # 需要将局部坐标旋转 theta_true_rad 映射到全局 3D 坐标系，
            # 并减去滑翔机在下潜过程中水平飞行的距离。
            deploy_cmds[idx] = {'T0': T0, 't_enc': t_enc} # 暂存时间
            
        return deploy_cmds

# ==========================================
# 3. 虚拟采样模块 (Step 2)
# ==========================================
class VirtualSampler:
    def __init__(self, planner):
        self.planner = planner

    def generate_theoretical_times(self, C_p_true, theta_true_deg, t0=0.0):
        """正向运动学：生成理想无偏环境下的理论到达时间 (Baseline Test)"""
        theta_rad = np.radians(theta_true_deg)
        C_app = C_p_true + self.planner.v_g # 视在相速度
        
        # 严格按照等腰三角形迎面逆行的法向投影推导
        t1 = t0 + (-self.planner.L * np.sin(theta_rad) / 2.0) / C_app
        t2 = t0 + ( self.planner.L * np.sin(theta_rad) / 2.0) / C_app
        t3 = t0 + (-self.planner.H * np.cos(theta_rad)) / C_app
        
        return t1, t2, t3

    def extract_from_3d_data(self):
        """
        [预留接口]
        调用您的 single_W_A_Lagrangian 读取 W_Vel_3D.npy
        提取垂向流速极大值的真实仿真时间
        """
        pass

# ==========================================
# 4. TDOA 逆向解算核心模块 (Step 3)
# ==========================================
class TdoaInverter:
    def __init__(self, L, H, v_g):
        self.L = L
        self.H = H
        self.v_g = v_g

    def solve(self, t1, t2, t3):
        """解耦时空维度，消除多普勒涂抹，纯数学反推"""
        # 1. 提取时间差
        delta_t_y = t2 - t1
        delta_t_x = 0.5 * (t1 + t2) - t3
        
        # 2. 解算相对偏角 theta
        y_comp = delta_t_y / self.L
        x_comp = delta_t_x / self.H
        theta_calc_rad = np.arctan2(y_comp, x_comp)
        theta_calc_deg = np.degrees(theta_calc_rad)
        
        # 3. 解算真实的内波相速度 (视在速度减去滑翔机自身水平速度)
        C_app_calc = 1.0 / np.sqrt(y_comp**2 + x_comp**2)
        C_p_calc = C_app_calc - self.v_g
        
        return C_p_calc, theta_calc_deg


# ==========================================
# 主执行入口
# ==========================================
if __name__ == "__main__":
    # --- A. 模拟读取 params.json (参考您的截图数据) ---
    # 实际使用时：params = ConfigManager.load_params('V_Wave_Data_Line/run_xxxx/params.json')
    params_mock = {
        "c0": 2.748380010527255,
        "thermocline_depth": 240.48096192384767
    }
    
    C_p_real = params_mock["c0"]
    z_max_real = params_mock["thermocline_depth"]
    theta_real = 15.0 # 测试偏角：15度
    
    # --- B. 阵列与滑翔机参数配置 ---
    L_spacing = 4000.0   # 等腰三角形底边 (y方向间距 4km)
    H_spacing = 2000.0   # 等腰三角形高 (x方向间距 2km)
    v_glider_h = 0.35    # 滑翔机水平速度 (迎面逆行)
    v_glider_z = 0.15    # 滑翔机垂直下潜速度
    
    print("====== 纯数学零误差基准测试 (Baseline Model) ======")
    print(f"输入真实波速: {C_p_real:.6f} m/s")
    print(f"输入真实偏角: {theta_real:.6f} 度\n")
    
    # --- C. 初始化各个模块 ---
    planner = DeploymentPlanner(L_spacing, H_spacing, v_glider_h, v_glider_z, z_max_real)
    sampler = VirtualSampler(planner)
    inverter = TdoaInverter(L_spacing, H_spacing, v_glider_h)
    
    # --- D. 运行测试 ---
    # 1. 获得上帝视角的理论到达时间
    t1, t2, t3 = sampler.generate_theoretical_times(C_p_real, theta_real, t0=10000.0)
    print(f"模拟波达时间(s): t1={t1:.3f}, t2={t2:.3f}, t3={t3:.3f}")
    
    # 2. 盲测反推
    Cp_calc, theta_calc = inverter.solve(t1, t2, t3)
    print("\n====== 解算结果 ======")
    print(f"解算波速: {Cp_calc:.6f} m/s (绝对误差: {abs(Cp_calc - C_p_real):.2e} m/s)")
    print(f"解算偏角: {theta_calc:.6f} 度 (绝对误差: {abs(theta_calc - theta_real):.2e} 度)")
    
    if abs(Cp_calc - C_p_real) < 1e-6 and abs(theta_calc - theta_real) < 1e-6:
        print("\n✅ 验证通过！运动学解耦与 TDOA 数学模型绝对正确。")
    else:
        print("\n❌ 验证失败！请检查公式推导。")
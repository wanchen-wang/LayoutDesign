"""
三角形 TDOA 内波探测 — 核心模型模块
=====================================
提供从阵列部署规划到 TDOA 反解的完整流水线。

  模块 / 类:
    ConfigManager      — 读取波场参数、筛选数据组
    DeploymentPlanner  — 等腰三角形阵列布置与运动学回退（得到 X0/Y0/T0）
    VirtualSampler     — 调用 30cut 进行虚拟轨迹采样
    TdoaInverter       — 根据三节点峰值时刻反解波速与偏角

  流水线函数:
    run_tdoa_group     — 单波组完整 TDOA 流程（对 30cut 调用 3 次）
    run_tdoa_batch     — 批量波组处理，结果写入 CSV
    run_cut30_pipeline — 单滑翔机独立振幅分析（保留用途）
"""
import os
import json
import numpy as np
import pandas as pd
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1] 
SWA_DIR = PROJECT_ROOT / "Simulated_Sampling_And_Amplitude_Fitting"
ANALYSIS_OUTPUT_DIR = PROJECT_ROOT / "Analysis_C_Data"
TRAJECTORY_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "Trajectories"
DEFAULT_TDOA_SUMMARY_CSV = ANALYSIS_OUTPUT_DIR / "TDOA_Metrics_Summary.csv"


def _load_cut30_module():
    """Load an available 30cut module from the sibling sampling folder."""
    candidate_files = [
        "Single_W_A_Lagrangian_30Cut.py",
        "Single_W_A_Lagrangian_30Cut_Diffstart.py",
    ]

    for file_name in candidate_files:
        module_path = SWA_DIR / file_name
        if not module_path.exists():
            continue

        spec = importlib.util.spec_from_file_location("single_w_a_lagrangian_30cut", str(module_path))
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return None


# 动态加载 30cut 模块，同步使用其滑翔机配置以确保两侧参数一致；
# 若加载失败则退回内置默认值（不影响类结构，仅单元测试时失效）。
_CUT30_MODULE = _load_cut30_module()
if _CUT30_MODULE is not None:
    run_batch_30cut  = _CUT30_MODULE.run_batch_30cut
    get_glider_config = _CUT30_MODULE.get_glider_config
else:
    run_batch_30cut = None

    def get_glider_config():
        return {"v_g": 0.22, "v_z": 1000.0 / 6000.0, "depth_max": 1000.0}

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

    @staticmethod
    def list_groups(base_data_dir):
        """列出数据组目录，按名称排序。"""
        if not os.path.isdir(base_data_dir):
            return []
        groups = [
            d for d in os.listdir(base_data_dir)
            if os.path.isdir(os.path.join(base_data_dir, d))
        ]
        return sorted(groups)

    @staticmethod
    def select_groups(base_data_dir, start_idx=None, end_idx=None, max_groups=None):
        """按 1-based 组号范围和组数筛选处理组。"""
        groups = ConfigManager.list_groups(base_data_dir)
        if not groups:
            return []

        if start_idx is None:
            start_idx = 1
        if end_idx is None:
            end_idx = len(groups)

        start_idx = max(1, int(start_idx))
        end_idx = min(len(groups), int(end_idx))
        if start_idx > end_idx:
            return []

        selected = groups[start_idx - 1:end_idx]
        if max_groups is not None:
            selected = selected[:max(0, int(max_groups))]
        return selected

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
        # 尖端为(0,0)，底边两台滑翔机在它的后方距离 H 的位置展开
        return {
            1: np.array([0.0, 0.0]),  # 尖端滑翔机，牢牢锚定在 (0,0)
            2: np.array([-self.H, -self.L / 2.0]),  # 底边下 (落后于尖端)
            3: np.array([-self.H, self.L / 2.0]),   # 底边上 (落后于尖端)
        }

    def calculate_surface_deployment(self, t_encounter_dict, theta_true_rad, X_offset=0.0, Y_offset=0.0):
        """
        反推海面初始发车时间 T0 和 全局发车坐标 X0, Y0
        :param t_encounter_dict: 各节点在 z_max 深度与波峰相遇的绝对时间
        :param theta_true_rad: 您预设的相对传播偏角 (弧度)
        :param X_offset, Y_offset: 您的 3D 流场网格中心点的坐标偏移量
        :return: 包含每个节点 (T0, X0, Y0) 的字典
        """
        # 1. 取得在局部等腰三角形坐标系下，各节点在 z_max 处的相遇点坐标
        local_points = self.get_local_encounter_points()
        
        # 2. 滑翔机从海面下潜到最大振幅深度所需的耗时
        t_dive = self.z_max / self.v_z 
        
        deploy_cmds = {}
        
        for idx, t_enc in t_encounter_dict.items():
            # ==========================================
            # A. 初始发车时间 T0
            # ==========================================
            T0 = t_enc - t_dive
            
            # ==========================================
            # B. 坐标空间旋转 (Local to Global)
            # ==========================================
            # 提取局部坐标
            x_loc, y_loc = local_points[idx]
            
            # 乘以旋转矩阵，将局部三角形旋转 theta 角度，并加上流场网格中心的平移偏移
            # （假设全局流场中波向 -X 传播）
            X_enc_global = x_loc * np.cos(theta_true_rad) - y_loc * np.sin(theta_true_rad) + X_offset
            Y_enc_global = x_loc * np.sin(theta_true_rad) + y_loc * np.cos(theta_true_rad) + Y_offset
            
            # ==========================================
            # C. 运动学回退：计算海面发车点 (X0, Y0)
            # ==========================================
            # 核心物理逻辑：因为我们设定整个阵列向着波的法向(迎面)逆行，
            # 既然波在全局中始终向 -X 走，滑翔机在全局中就必然严格向 +X 飞。
            # 因此，在 t_dive 这段时间内，滑翔机在全局 X 方向上飞过了 (v_g * t_dive) 的距离，Y 方向没有位移。
            # 海面的发车点 X0 需要把这段水平飞行的距离减掉！
            
            X0_global = X_enc_global - (self.v_g * t_dive)
            Y0_global = Y_enc_global  # Y 方向不受水平迎面飞行的影响
            
            deploy_cmds[idx] = {
                'T0': T0, 
                'X0': X0_global, 
                'Y0': Y0_global,
                't_enc': t_enc
            }
            
        return deploy_cmds

# ==========================================
# 3. 虚拟采样模块 (Step 2)
# ==========================================
class VirtualSampler:
    def __init__(self, planner):
        self.planner = planner

    def generate_theoretical_times(self, C_p_true, theta_true_deg, t0=0.0):
        """正向运动学：生成理想无偏环境下的理论到达时间 (Baseline Test)
        返回顺序: (t_node1=尖端, t_node2=底边下, t_node3=底边上)
        """
        theta_rad = np.radians(theta_true_deg)
        C_app = C_p_true + self.planner.v_g # 视在相速度
        
        # 尖端 (node1): 局部坐标 (0, 0)
        t_tip   = t0 + (-self.planner.H * np.cos(theta_rad)) / C_app
        # 底边下 (node2): 局部坐标 (-H, -L/2)
        t_lower = t0 + (-self.planner.L * np.sin(theta_rad) / 2.0) / C_app
        # 底边上 (node3): 局部坐标 (-H, +L/2)
        t_upper = t0 + ( self.planner.L * np.sin(theta_rad) / 2.0) / C_app
        
        return t_tip, t_lower, t_upper

    def extract_from_3d_data(
        self,
        data_dir,
        X0_global,
        Y0_global,
        T0_global,
        cut_percentage=30,
        enable_amplitude_calc=False,
    ):
        """调用 30cut 单组处理，返回与组网调度对齐的真实仿真时间。"""
        if _CUT30_MODULE is None:
            raise ImportError("无法导入 Single_W_A_Lagrangian_30Cut.py，请检查路径配置。")

        run_single_group_30cut = getattr(_CUT30_MODULE, "run_single_group_30cut", None)
        if run_single_group_30cut is None:
            raise AttributeError("30cut 模块缺少 run_single_group_30cut 接口。")

        result = run_single_group_30cut(
            data_dir=data_dir,
            X0_global=X0_global,
            Y0_global=Y0_global,
            T0_global=T0_global,
            cut_percentage=cut_percentage,
            glider_config=get_glider_config(),
            enable_amplitude_calc=enable_amplitude_calc,
        )

        t_w0 = float(result["t_w0"])
        t_peak = float(result.get("t_peak", 0.5 * (result["t_w0"] + result["t_U"])))
        t_U = float(result["t_U"])

        return {
            "X0": float(result.get("X0", X0_global)),
            "Y0": float(result.get("Y0", Y0_global)),
            "T0": float(result.get("T0", T0_global)),
            "t_w0": t_w0,
            "t_peak": t_peak,
            "t_U": t_U,
            "dh": float(result.get("dh", np.nan)),
            "true_h0": float(result.get("true_h0", np.nan)),
            "error_pct": float(result.get("error_pct", np.nan)),
            "raw": result,
        }


# ==========================================
# 4. TDOA 逆向解算核心模块
# ==========================================
class TdoaInverter:
    """根据三节点峰值到达时刻，反解内波相速度与传播偏角。"""

    def __init__(self, L, H, v_g):
        self.L   = L    # 等腰三角形底边长 (y 方向, m)
        self.H   = H    # 等腰三角形高     (x 方向, m)
        self.v_g = v_g  # 滑翔机水平速度   (m/s)

    def solve(self, t1, t2, t3):
        """
        解耦时空维度，消除多普勒涂抹，纯数学反推。

        参数约定（与节点物理编号对应）:
          t1 — node1 尖端节点的峰値到达时刻 (s)
          t2 — node2 底边下节点的峰値到达时刻 (s)
          t3 — node3 底边上节点的峰値到达时刻 (s)
        返回:
          C_p_calc       — 内波真实相速度 (m/s)
          theta_calc_deg — 传播偏角       (deg)
        """
        # 底边两节点时差 → 感知 y 方向；底边中点与尖端时差 → 感知 x 方向
        delta_t_y = t3 - t2                  # node3(底边上) - node2(底边下)
        delta_t_x = 0.5 * (t2 + t3) - t1    # 底边中点 - 尖端(node1)

        # 归一化：时差 / 空间间距 = 视在慢度在该方向的投影
        y_comp = delta_t_y / self.L
        x_comp = delta_t_x / self.H

        # 传播偏角（atan2 自动给出象限正确的角度）
        theta_calc_rad = np.arctan2(y_comp, x_comp)
        theta_calc_deg = np.degrees(theta_calc_rad)

        # 视在相速度 → 减去平台自身速度 → 真实相速度
        C_app_calc = 1.0 / np.sqrt(y_comp**2 + x_comp**2)
        C_p_calc   = C_app_calc - self.v_g

        return C_p_calc, theta_calc_deg


# ==========================================
# 5. 流水线接口函数
# ==========================================

def dock_virtual_sampler_with_30cut(sampler, data_dir, deploy_cmds, cut_percentage=30):
    """
    将组网调度输出的 X0/Y0/T0 逐节点分发到 30cut 单机采样接口。
    每个节点独立调用一次 run_single_group_30cut，互不干扰。
    """
    node_results = {}
    for node_id, cmd in deploy_cmds.items():
        if not all(k in cmd for k in ("X0", "Y0", "T0")):
            raise KeyError(f"节点 {node_id} 缺少 X0/Y0/T0 发车参数。")

        node_results[node_id] = sampler.extract_from_3d_data(
            data_dir=data_dir,
            X0_global=cmd["X0"],
            Y0_global=cmd["Y0"],
            T0_global=cmd["T0"],
            cut_percentage=cut_percentage,
            enable_amplitude_calc=False,
        )
    return node_results


def _build_trajectory_file_name(wave_id, L_spacing, H_spacing, theta_real_deg):
    """构造单组轨迹压缩包文件名。"""
    theta_text = f"{float(theta_real_deg):g}".replace("-", "neg")
    return f"traj_{wave_id}_L{int(round(L_spacing))}_H{int(round(H_spacing))}_Th{theta_text}.npz"


def _extract_node_encounter_position(node_result):
    """从单节点采样结果中提取峰值相遇位置。"""
    raw = node_result["raw"]
    t_series = np.asarray(raw["t_global_array"], dtype=float)
    x_series = np.asarray(raw["x_track_global"], dtype=float)
    y_series = np.asarray(raw["y_track_global"], dtype=float)
    z_series = np.asarray(raw["z_track"], dtype=float)
    peak_idx = int(np.argmin(np.abs(t_series - float(node_result["t_peak"]))))

    return {
        "x_enc": float(x_series[peak_idx]),
        "y_enc": float(y_series[peak_idx]),
        "z_enc": float(z_series[peak_idx]),
    }


def _save_group_trajectories(result, trajectory_dir, wave_id, L_spacing, H_spacing, theta_real_deg):
    """将三台滑翔机的轨迹与关键元数据保存为单个 npz 文件。"""
    trajectory_dir = Path(trajectory_dir)
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    file_name = _build_trajectory_file_name(wave_id, L_spacing, H_spacing, theta_real_deg)
    file_path = trajectory_dir / file_name
    node_results = result["node_results"]

    npz_payload = {
        "wave_id": np.array(wave_id),
        "L_spacing": np.array(float(L_spacing)),
        "H_spacing": np.array(float(H_spacing)),
        "theta_true_deg": np.array(float(result["theta_true"])),
        "theta_calc_deg": np.array(float(result["theta_calc"])),
        "C_p_true": np.array(float(result["C_p_true"])),
        "C_p_calc": np.array(float(result["C_p_calc"])),
        "t1_ref": np.array(float(result["t1_ref"])),
        "t2_ref": np.array(float(result["t2_ref"])),
        "t3_ref": np.array(float(result["t3_ref"])),
        "t1_obs": np.array(float(result["t1_obs"])),
        "t2_obs": np.array(float(result["t2_obs"])),
        "t3_obs": np.array(float(result["t3_obs"])),
    }

    for node_id, node_result in node_results.items():
        raw = node_result["raw"]
        prefix = f"node{node_id}"
        npz_payload[f"{prefix}_X0"] = np.array(float(node_result["X0"]))
        npz_payload[f"{prefix}_Y0"] = np.array(float(node_result["Y0"]))
        npz_payload[f"{prefix}_T0"] = np.array(float(node_result["T0"]))
        npz_payload[f"{prefix}_t_w0"] = np.array(float(node_result["t_w0"]))
        npz_payload[f"{prefix}_t_peak"] = np.array(float(node_result["t_peak"]))
        npz_payload[f"{prefix}_t_U"] = np.array(float(node_result["t_U"]))
        npz_payload[f"{prefix}_dh"] = np.array(float(node_result["dh"]))
        npz_payload[f"{prefix}_error_pct"] = np.array(float(node_result["error_pct"]))
        npz_payload[f"{prefix}_t_global_array"] = np.asarray(raw["t_global_array"], dtype=float)
        npz_payload[f"{prefix}_x_track_global"] = np.asarray(raw["x_track_global"], dtype=float)
        npz_payload[f"{prefix}_y_track_global"] = np.asarray(raw["y_track_global"], dtype=float)
        npz_payload[f"{prefix}_z_track"] = np.asarray(raw["z_track"], dtype=float)
        if "w_sampled" in raw:
            npz_payload[f"{prefix}_w_sampled"] = np.asarray(raw["w_sampled"], dtype=float)

    np.savez_compressed(file_path, **npz_payload)
    return file_path


def _build_summary_row(result, wave_id, trajectory_path):
    """构造汇总 CSV 行，包含标量、关键坐标和轨迹文件位置。"""
    row = {
        "wave_id": wave_id,
        "trajectory_file": str(Path(trajectory_path).relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "C_p_true": float(result["C_p_true"]),
        "theta_true": float(result["theta_true"]),
        "C_p_calc": float(result["C_p_calc"]),
        "theta_calc": float(result["theta_calc"]),
        "C_p_error": float(result["C_p_error"]),
        "theta_error": float(result["theta_error"]),
        "t1_ref": float(result["t1_ref"]),
        "t2_ref": float(result["t2_ref"]),
        "t3_ref": float(result["t3_ref"]),
        "t1_obs": float(result["t1_obs"]),
        "t2_obs": float(result["t2_obs"]),
        "t3_obs": float(result["t3_obs"]),
    }

    for node_id in [1, 2, 3]:
        node_result = result["node_results"][node_id]
        encounter_pos = _extract_node_encounter_position(node_result)
        row[f"node{node_id}_X0"] = float(node_result["X0"])
        row[f"node{node_id}_Y0"] = float(node_result["Y0"])
        row[f"node{node_id}_T0"] = float(node_result["T0"])
        row[f"node{node_id}_t_w0"] = float(node_result["t_w0"])
        row[f"node{node_id}_t_peak"] = float(node_result["t_peak"])
        row[f"node{node_id}_t_U"] = float(node_result["t_U"])
        row[f"node{node_id}_dh"] = float(node_result["dh"])
        row[f"node{node_id}_error_pct"] = float(node_result["error_pct"])
        row[f"node{node_id}_x_enc"] = encounter_pos["x_enc"]
        row[f"node{node_id}_y_enc"] = encounter_pos["y_enc"]
        row[f"node{node_id}_z_enc"] = encounter_pos["z_enc"]

    return row


def run_tdoa_group(
    data_dir,
    L_spacing,
    H_spacing,
    glider_cfg,
    theta_real_deg,
    t0_ref=10000.0,
    cut_percentage=30,
):
    """
    单组 TDOA 完整流程（对应一个波场数据目录）：

      步骤 1  从 params.json 读取波场真实参数 (C_p, z_max)
      步骤 2  正向运动学 → 三节点理论相遇时刻
      步骤 3  运动学回退 → 三滑翔机各自独立的海面发车参数 (X0, Y0, T0)
      步骤 4  调用 30cut 三次，每个滑翔机传入自己的 X0/Y0/T0
      步骤 5  TDOA 反解波速与传播偏角

    返回含真实值、解算值、三节点采样细节的字典。
    """
    params   = ConfigManager.load_params(os.path.join(data_dir, "params.json"))
    C_p_real = params["c0"]
    z_max    = params["thermocline_depth"]
    v_g      = float(glider_cfg["v_g"])
    v_z      = float(glider_cfg["v_z"])

    planner  = DeploymentPlanner(L_spacing, H_spacing, v_g, v_z, z_max)
    sampler  = VirtualSampler(planner)
    inverter = TdoaInverter(L_spacing, H_spacing, v_g)

    # 步骤 2: 正向运动学 → 理论相遇时刻
    # t1_ref=尖端(node1), t2_ref=底边下(node2), t3_ref=底边上(node3)
    t1_ref, t2_ref, t3_ref = sampler.generate_theoretical_times(
        C_p_real, theta_real_deg, t0=t0_ref
    )

    # 步骤 3: 运动学回退 → 三滑翔机各自的海面发车参数
    deploy_cmds = planner.calculate_surface_deployment(
        t_encounter_dict={1: t1_ref, 2: t2_ref, 3: t3_ref},
        theta_true_rad=np.radians(theta_real_deg),
    )

    # 步骤 4: 调用 30cut 三次，每次对应一个滑翔机节点
    node_results = dock_virtual_sampler_with_30cut(
        sampler=sampler,
        data_dir=data_dir,
        deploy_cmds=deploy_cmds,
        cut_percentage=cut_percentage,
    )

    # 步骤 5: TDOA 反解
    # inverter.solve(t1=尖端, t2=底边下, t3=底边上)
    t1_obs = node_results[1]["t_peak"]  # node1 = 尖端
    t2_obs = node_results[2]["t_peak"]  # node2 = 底边下
    t3_obs = node_results[3]["t_peak"]  # node3 = 底边上
    C_p_calc, theta_calc = inverter.solve(t1_obs, t2_obs, t3_obs)

    return {
        "C_p_true":        C_p_real,
        "theta_true":      theta_real_deg,
        "C_p_calc":        C_p_calc,
        "theta_calc":      theta_calc,
        "C_p_error":       abs(C_p_calc - C_p_real),
        "theta_error":     abs(theta_calc - theta_real_deg),
        "t1_ref": t1_ref, "t2_ref": t2_ref, "t3_ref": t3_ref,
        "t1_obs": t1_obs, "t2_obs": t2_obs, "t3_obs": t3_obs,
        "dh_node1":        node_results[1]["dh"],
        "error_pct_node1": node_results[1]["error_pct"],
        "dh_node2":        node_results[2]["dh"],
        "error_pct_node2": node_results[2]["error_pct"],
        "dh_node3":        node_results[3]["dh"],
        "error_pct_node3": node_results[3]["error_pct"],
        "node_results":    node_results,
    }


def run_tdoa_batch(
    base_data_dir,
    output_csv,
    L_spacing,
    H_spacing,
    glider_cfg,
    theta_real_deg,
    t0_ref=10000.0,
    cut_percentage=30,
    start_idx=None,
    end_idx=None,
    max_groups=None,
):
    """
    批处理入口：遍历所有波组，每组调用 run_tdoa_group（内含 3 次 30cut 调用），
    将每组的 TDOA 解算结果写入汇总 CSV，并将三台滑翔机轨迹保存为单组 npz 文件。
    """
    selected_groups = ConfigManager.select_groups(
        base_data_dir=base_data_dir,
        start_idx=start_idx,
        end_idx=end_idx,
        max_groups=max_groups,
    )
    if not selected_groups:
        raise RuntimeError(f"未找到可处理组别，请检查目录: {base_data_dir}")

    print(f"\n{'='*60}")
    print(f"🚀 三滑翔机 TDOA 批处理: {int(cut_percentage)}% cut, θ={theta_real_deg}°")
    print(f"📂 数据目录: {base_data_dir}  共 {len(selected_groups)} 组")
    print(f"{'='*60}")

    rows = []
    first_result = None
    output_csv_path = Path(output_csv)
    trajectory_dir = output_csv_path.parent / "Trajectories"
    for folder_name in selected_groups:
        data_dir = os.path.join(base_data_dir, folder_name)
        try:
            result = run_tdoa_group(
                data_dir=data_dir,
                L_spacing=L_spacing,
                H_spacing=H_spacing,
                glider_cfg=glider_cfg,
                theta_real_deg=theta_real_deg,
                t0_ref=t0_ref,
                cut_percentage=cut_percentage,
            )
            trajectory_path = _save_group_trajectories(
                result=result,
                trajectory_dir=trajectory_dir,
                wave_id=folder_name,
                L_spacing=L_spacing,
                H_spacing=H_spacing,
                theta_real_deg=theta_real_deg,
            )
            row = _build_summary_row(result, folder_name, trajectory_path)
            rows.append(row)
            if first_result is None:
                first_result = result
            print(f"  ✓ {folder_name}  Cp_err={result['C_p_error']:.2e}  θ_err={result['theta_error']:.4f}°")
        except Exception as e:
            print(f"  [警告] {folder_name} 处理异常，已跳过: {e}")

    if not rows:
        raise RuntimeError("没有可写入的数据。")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)
    print(f"\n✅ 汇总 CSV 已保存: {output_csv_path}  样本数: {len(df)}")
    print(f"✅ 轨迹文件目录: {trajectory_dir}")
    return str(output_csv_path), first_result


def run_cut30_pipeline(
    base_data_dir,
    output_csv,
    start_idx=None,
    end_idx=None,
    max_groups=None,
    deployment_cmds=None,
):
    """
    单滑翔机模式批处理（保留用于独立振幅误差分析）。
    每组仅对单个 X0/Y0/T0 调用一次 30cut，不执行 TDOA 解算。
    """
    if run_batch_30cut is None:
        raise ImportError("无法导入 Single_W_A_Lagrangian_30Cut.py，请检查路径配置。")

    return run_batch_30cut(
        base_data_dir=base_data_dir,
        output_csv=output_csv,
        start_idx=start_idx,
        end_idx=end_idx,
        max_groups=max_groups,
        cut_percentage=30,
        deployment_cmds=deployment_cmds,
    )


# ==========================================
# 主执行入口
# ==========================================
if __name__ == "__main__":
    base_data_dir = str(PROJECT_ROOT / "V_Wave_Data_Line")
    output_csv = str(DEFAULT_TDOA_SUMMARY_CSV)

    start_text = input("处理起始组编号 start_idx (默认 1): ").strip()
    end_text = input("处理结束组编号 end_idx (默认最后一组): ").strip()
    count_text = input("最多处理组数 max_groups (默认不限): ").strip()

    start_idx = int(start_text) if start_text else None
    end_idx = int(end_text) if end_text else None
    max_groups = int(count_text) if count_text else None

    selected_groups = ConfigManager.select_groups(
        base_data_dir=base_data_dir,
        start_idx=start_idx,
        end_idx=end_idx,
        max_groups=max_groups,
    )
    if not selected_groups:
        raise RuntimeError(f"未找到可处理组别，请检查目录: {base_data_dir}")

    first_group = selected_groups[0]
    params_path = os.path.join(base_data_dir, first_group, "params.json")
    params_real = ConfigManager.load_params(params_path)

    C_p_real = params_real["c0"]
    theta_real = -15.0 # 测试偏角：-15度

    # --- B. 阵列与滑翔机参数配置 ---
    L_spacing = 6800.0   # 等腰三角形底边 (y方向间距 20km)
    H_spacing = 2000.0   # 等腰三角形高 (x方向间距 2km)
    glider_cfg = get_glider_config()
    
    print("====== 纯数学零误差基准测试 (Baseline Model) ======")
    print(f"参考数据组: {first_group}")
    print(f"输入真实波速: {C_p_real:.6f} m/s")
    print(f"输入真实偏角: {theta_real:.6f} 度\n")

    # 3) 每组调用 30cut 三次（三个滑翔机各传入不同 X0/Y0/T0），合并为一行输出 CSV
    csv_path, first_result = run_tdoa_batch(
        base_data_dir=base_data_dir,
        output_csv=output_csv,
        L_spacing=L_spacing,
        H_spacing=H_spacing,
        glider_cfg=glider_cfg,
        theta_real_deg=theta_real,
        t0_ref=10000.0,
        cut_percentage=30,
        start_idx=start_idx,
        end_idx=end_idx,
        max_groups=max_groups,
    )
    print(f"TDOA 批处理结果 CSV: {csv_path}")
'''
    # --- D. 第一组单组验证输出 ---
    print(f"\n====== 第一组 ({first_group}) 基准验证 ======")
    print(f"理论波达时间(s): t1={first_result['t1_ref']:.3f}, t2={first_result['t2_ref']:.3f}, t3={first_result['t3_ref']:.3f}")
    print(f"观测峰值时间(s): t1={first_result['t1_obs']:.3f}, t2={first_result['t2_obs']:.3f}, t3={first_result['t3_obs']:.3f}")

    Cp_calc = first_result["C_p_calc"]
    theta_calc = first_result["theta_calc"]
    print("\n====== 解算结果 ======")
    print(f"解算波速: {Cp_calc:.6f} m/s (绝对误差: {first_result['C_p_error']:.2e} m/s)")
    print(f"解算偏角: {theta_calc:.6f} 度 (绝对误差: {first_result['theta_error']:.2e} 度)")

    if first_result["C_p_error"] < 1e-6 and first_result["theta_error"] < 1e-6:
        print("\n✅ 验证通过！运动学解耦与 TDOA 数学模型绝对正确。")
    else:
        print("\n❌ 验证失败！请检查公式推导。")
'''
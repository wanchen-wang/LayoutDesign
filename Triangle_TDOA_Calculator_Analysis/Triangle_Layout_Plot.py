import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")   # 强制使用交互式窗口后端
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Arc

# ==========================================
# 1. 字体与路径配置
# ==========================================
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "Analysis_C_Data"
WAVE_DATA_DIR = PROJECT_ROOT / "V_Wave_Data_Line"
CSV_PATH = ANALYSIS_DIR / "TDOA_Metrics_Summary.csv"
OUTPUT_PIC_DIR = PROJECT_ROOT / "Pic_New"  # 新图表输出目录

NODE_COLORS = {1: "#d1495b", 2: "#2b59c3", 3: "#2a9d8f"}

# ==========================================
# 2. 辅助工具
# ==========================================
def _annotate_point(ax, x, y, text, dx, dy, color='black', fontsize=8):
    """在数据点旁绘制带白色底框的标注箭头。"""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords='offset points',
        fontsize=fontsize,
        color=color,
        arrowprops=dict(arrowstyle='->', color=color, lw=0.8),
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=color, alpha=0.85),
    )


# ==========================================
# 3. 可视化函数 I：三台滑翔机采样过程及特征点
# ==========================================
def plot_sampling_kinematics(npz_data, wave_id):
    """绘制三架滑翔机的拉格朗日采样过程（深度轨迹与垂直流速）"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
    fig.suptitle(f"[{wave_id}] 三台滑翔机拉格朗日采样与 30% 截断特征",
                 fontsize=15, fontweight='bold', y=0.98)

    # 绘制顺序：node1=1号(尖端), node2=2号(底边下), node3=3号(底边上)
    PLOT_ORDER = [(1, "1号 (尖端)"), (2, "2号 (底边下)"), (3, "3号 (底边上)")]

    for i, (node_id, glider_label) in enumerate(PLOT_ORDER):
        ax_w = axes[i]       # 左轴：垂直流速（主轴）
        ax_z = ax_w.twinx()  # 右轴：深度轨迹

        t = npz_data[f"node{node_id}_t_global_array"]
        z = npz_data[f"node{node_id}_z_track"]
        w = npz_data[f"node{node_id}_w_sampled"]
        idx_arr = np.arange(len(t))

        # —— 右轴：深度轨迹（橙色实线）
        ax_z.plot(t, z, color='darkorange', lw=1.5, ls='-', label='Glider Depth')
        ax_z.invert_yaxis()
        ax_z.set_ylabel('Depth (m)', color='darkorange', fontsize=9)
        ax_z.tick_params(axis='y', labelcolor='darkorange')

        # —— 特征点计算
        max_idx        = int(np.argmax(w))
        t_peak, w_peak = t[max_idx], w[max_idx]
        cut_val        = 0.3 * w_peak

        _lc = np.where(w[:max_idx] <= cut_val)[0]
        left_cut  = int(_lc[-1])  if len(_lc) > 0 else 0
        _rc = np.where(w[max_idx:] <= cut_val)[0]
        right_cut = max_idx + int(_rc[0]) if len(_rc) > 0 else len(w) - 1
        _lz = np.where(w[:max_idx] <= 0)[0]
        left_zero  = int(_lz[-1])  if len(_lz) > 0 else 0
        _rz = np.where(w[max_idx:] <= 0)[0]
        right_zero = max_idx + int(_rz[0]) if len(_rz) > 0 else len(w) - 1

        # —— 左轴：蓝色流速主线
        ax_w.plot(t, w, color='#005b96', lw=2, label=f'{glider_label} $w_{{isw}}$')
        ax_w.axhline(0, color='gray', ls='--', lw=0.8)

        # —— 填充区：负速度=蓝色，积分区（截断内正速度）=粉色，截断外正速度=绿色
        ax_w.fill_between(t, 0, w,
                          where=(w < 0),
                          color='#6b9ac4', alpha=0.45, label='Negative velocity')
        ax_w.fill_between(t, 0, w,
                          where=(idx_arr >= left_cut) & (idx_arr <= right_cut) & (w > 0),
                          color='#f4a1c1', alpha=0.55, label='Kept for integration')
        cut_mask = (idx_arr >= left_zero) & (idx_arr <= right_zero) \
                   & ~((idx_arr >= left_cut) & (idx_arr <= right_cut)) & (w > 0)
        ax_w.fill_between(t, 0, w,
                          where=cut_mask,
                          color='green', alpha=0.35, label='Cut-off area')

        # —— 截断阈值线
        ax_w.axhline(cut_val, color='green', ls=':', lw=1.5, alpha=0.8,
                     label=f'Cut threshold (30% of peak)')

        # —— Peak 标注（红色散点）
        ax_w.scatter([t_peak], [w_peak], color='red', s=40, zorder=6)
        _annotate_point(ax_w, t_peak, w_peak,
                        f"Peak\n({t_peak:.1f}s, {w_peak:.4f}m/s)",
                        18, 18, color='red')

        # —— 过零点标注（黑色圆点）
        for idx, tag, dx in [(left_zero, 'L', -95), (right_zero, 'R', 24)]:
            ax_w.scatter([t[idx]], [w[idx]], color='black', s=24, zorder=6)
            _annotate_point(ax_w, t[idx], w[idx],
                            f"Zero {tag}\n({t[idx]:.1f}s, {w[idx]:.4f})",
                            dx, 22, color='black')

        # —— 截断点标注（绿色散点）
        for idx, tag, dx in [(left_cut, 'L', -95), (right_cut, 'R', 24)]:
            ax_w.scatter([t[idx]], [w[idx]], color='green', s=28, zorder=6)
            _annotate_point(ax_w, t[idx], w[idx],
                            f"Cut {tag}\n({t[idx]:.1f}s, {w[idx]:.4f})",
                            dx, -42, color='green')

        # —— 深度峰值标注（橙色）
        peak_depth = z[max_idx]
        ax_z.scatter([t_peak], [peak_depth], color='darkorange', s=28, zorder=6)
        _annotate_point(ax_z, t_peak, peak_depth,
                        f"Depth@Peak\n({t_peak:.1f}s, {peak_depth:.1f}m)",
                        24, -38, color='darkorange')

        # —— 轴设置
        w_lim = max(abs(float(w.min())), abs(float(w.max()))) * 1.35
        if w_lim > 0:
            ax_w.set_ylim(-w_lim, w_lim)
        ax_w.set_ylabel(f"{glider_label}  $w_{{isw}}$ (m/s)", color='#005b96', fontsize=10)
        ax_w.tick_params(axis='y', labelcolor='#005b96')
        ax_w.grid(True, linestyle=':', alpha=0.5)

        # —— 图例（合并两轴）
        h1, l1 = ax_w.get_legend_handles_labels()
        h2, l2 = ax_z.get_legend_handles_labels()
        ax_w.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8, framealpha=0.85)

    axes[-1].set_xlabel("绝对时间 T (s)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ==========================================
# 4. 可视化函数 II：二维阵列俯视图 (带渐变波前)
# ==========================================
def plot_gradient_top_view(csv_row, npz_data, params):
    """绘制带物理渐变内波背景的组网俯视图"""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_aspect("equal")  # 等比例显示，确保物理距离不失真
    
    # 1. 提取物理参数
    D = float(params["D"])       # 特征半波宽
    theta = float(csv_row["theta_true"])
    
    # 提取坐标
    starts_x, starts_y, encs_x, encs_y = [], [], [], []
    for node_id in [1, 2, 3]:
        # 水面发车点
        starts_x.append(float(csv_row[f"node{node_id}_X0"]))
        starts_y.append(float(csv_row[f"node{node_id}_Y0"]))
        
        # 水下相遇点 (根据 w_sampled 最大值位置提取)
        w_arr = npz_data[f"node{node_id}_w_sampled"]
        max_idx = int(np.argmax(w_arr))
        encs_x.append(float(npz_data[f"node{node_id}_x_track_global"][max_idx]))
        encs_y.append(float(npz_data[f"node{node_id}_y_track_global"][max_idx]))
    
    # 2. 绘制内孤立波渐变背景 (KdV 波形剖面)
    # 我们假定以 1 号机（尖端）相遇的时刻作为当前快照时间，此时波峰刚好处在 1 号机的相遇点 X
    wave_crest_x = encs_x[0] 
    
    # 构建绘图网格区域 (在滑翔机运动范围外扩充一定边距)
    x_min, x_max = min(starts_x + encs_x) - 300, max(starts_x + encs_x) + 300
    y_min, y_max = min(starts_y + encs_y) - 500, max(starts_y + encs_y) + 500
    
    xx = np.linspace(x_min, x_max, 300)
    yy = np.linspace(y_min, y_max, 100)
    X_mesh, Y_mesh = np.meshgrid(xx, yy)
    
    # 根据 KdV 平面波理论公式生成渐变场: Amplitude ~ sech^2((x - x_crest) / D)
    Z_mesh = np.cosh((X_mesh - wave_crest_x) / D)**(-2)
    
    # 绘制蓝色渐变带 (模拟内波从右向左打来的形态)
    contour = ax.contourf(X_mesh, Y_mesh, Z_mesh, levels=50, cmap='Blues', alpha=0.6)
    cbar = fig.colorbar(contour, ax=ax, pad=0.02)
    cbar.set_label('波形归一化强度 $\\propto sech^2(x/D)$', rotation=270, labelpad=15)
    
    # 3. 绘制滑翔机运动与阵列
    for i, node_id in enumerate([1, 2, 3]):
        color = NODE_COLORS[node_id]
        # 发车点 -> 相遇点 的水平投影轨迹
        ax.annotate("", xy=(encs_x[i], encs_y[i]), xytext=(starts_x[i], starts_y[i]),
                    arrowprops=dict(arrowstyle="->", color='gray', lw=1.5, ls='--'))
        # 标记水面发车点
        ax.plot(starts_x[i], starts_y[i], marker='s', color=color, markersize=7, label=f"Node {node_id} 水面" if i==0 else "")
        # 标记水下相遇点
        ax.plot(encs_x[i], encs_y[i], marker='o', color=color, markersize=9, label=f"Node {node_id} 相遇" if i==0 else "")

    # 绘制此时的等腰三角形阵型连线
    triangle_x = encs_x + [encs_x[0]]
    triangle_y = encs_y + [encs_y[0]]
    ax.plot(triangle_x, triangle_y, 'k-', lw=2, alpha=0.5, label="观测阵型连线")
    
    # 4. 辅助线与偏角标注
    # 理论平直波前（Y轴平行线，过 3 号机）
    ax.axvline(wave_crest_x, color='r', linestyle='-.', lw=1.5, alpha=0.8, label="理论波前垂线")
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"内孤立波组网观测二维俯视图\nWave ID: {csv_row['wave_id']} | 偏角 $\\theta$ = {theta}°", fontsize=14, fontweight='bold')
    ax.set_xlabel("全局坐标 X (m)", fontsize=12)
    ax.set_ylabel("全局坐标 Y (m)", fontsize=12)
    
    # 整理自定义图例
    custom_lines = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8, label='水面发车位置'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='水下相遇位置'),
        Line2D([0], [0], color='gray', ls='--', lw=1.5, label='下潜水平投影'),
        Line2D([0], [0], color='k', ls='-', lw=2, alpha=0.5, label='滑翔机阵型'),
        Line2D([0], [0], color='r', ls='-.', lw=1.5, label='理论波峰线')
    ]
    ax.legend(handles=custom_lines, loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    return fig


# ==========================================
# 5. 主执行流：选择记录并生成绘图
# ==========================================
def main():
    if not CSV_PATH.exists():
        print(f"❌ 找不到核心数据表: {CSV_PATH}")
        print("请先运行 Basic_Horizonal_Models.py 跑通数据！")
        return

    df = pd.read_csv(CSV_PATH)
    if df.empty:
        print("⚠️ 数据表为空。")
        return

    # —— 列出所有可用记录
    total = len(df)
    print(f"\n共找到 {total} 条实验记录：")
    print(f"  {'编号':<5}  {'wave_id':<28}  {'θ_true':>8}  {'C_p_true':>10}")
    print("  " + "-" * 58)
    for pos, (_, row) in enumerate(df.iterrows()):
        print(f"  {pos+1:<5}  {str(row['wave_id']):<28}  "
              f"{row['theta_true']:>8.1f}°  {row['C_p_true']:>10.4f}")

    # —— 交互选择
    while True:
        try:
            choice = input(f"\n请输入编号 [1-{total}，直接回车取最后一条]: ").strip()
            if choice == '':
                row_idx = total - 1
                break
            row_idx = int(choice) - 1
            if 0 <= row_idx < total:
                break
            print(f"  请输入 1~{total} 之间的数字。")
        except ValueError:
            print("  请输入有效数字。")

    selected_run = df.iloc[row_idx]
    wave_id      = str(selected_run['wave_id'])
    traj_rel_path = selected_run['trajectory_file']

    # —— 定位并读取 .npz 和 params.json
    npz_path    = PROJECT_ROOT / traj_rel_path
    params_path = WAVE_DATA_DIR / wave_id / "params.json"

    if not npz_path.exists() or not params_path.exists():
        print(f"❌ 缺失底层数据文件。请检查:\n1. {npz_path}\n2. {params_path}")
        return

    npz_data = np.load(npz_path)
    with open(params_path, 'r') as f:
        params = json.load(f)

    l_spacing = float(npz_data['L_spacing']) if 'L_spacing' in npz_data.files else '?'
    h_spacing = float(npz_data['H_spacing']) if 'H_spacing' in npz_data.files else '?'
    print(f"\n✅ 成功读取实验数据: {wave_id}")
    print(f"-> 阵型配置: L={l_spacing}m, H={h_spacing}m, 偏角={selected_run['theta_true']}°")

    # —— 绘图
    plot_sampling_kinematics(npz_data, wave_id)
    plot_gradient_top_view(selected_run, npz_data, params)
    plt.show()

if __name__ == "__main__":
    main()
"""
Driver script for batch data generation using V_Wave.run_simulation.
Supports generating multiple straight-line wavefront datasets in one run.
"""
from V_Wave_Line import run_simulation
import argparse
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


def resolve_target_dir(target_dir):
    """Resolve output path relative to project dir, not current terminal cwd."""
    if os.path.isabs(target_dir):
        return target_dir
    return os.path.abspath(os.path.join(PROJECT_DIR, target_dir))

def generate_v_wave_data(n_runs=2, save=True, target_dir="V_Wave_Data_Line"):
    """Run run_simulation repeatedly for straight wavefronts."""
    if n_runs < 1:
        raise ValueError("n_runs 必须大于等于 1")

    target_dir = resolve_target_dir(target_dir)

    print(f"🚀 开始批量生成 {n_runs} 个标准直线波前的三维内孤立波数据...")
    print(f"📁 目标存储文件夹: {target_dir}")
    
    for i in range(n_runs):
        print(f"\n--- 正在生成第 {i+1}/{n_runs} 个数据样本 ---")
        # 显式传递 base_folder 参数
        run_simulation(save=save, base_folder=target_dir)
        
    print(f"\n✅ 批量生成完成！所有 {n_runs} 个样本均已存入 {target_dir} 文件夹中。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量生成 V_Wave_Data_Line 数据")
    parser.add_argument("--n-runs", type=int, default=2, help="一次生成的数据组数量，默认 2")
    parser.add_argument(
        "--target-dir",
        type=str,
        default="V_Wave_Data_Line",
        help="输出目录；相对路径将基于项目根目录解析，默认 V_Wave_Data_Line",
    )
    args = parser.parse_args()

    generate_v_wave_data(n_runs=args.n_runs, target_dir=args.target_dir)
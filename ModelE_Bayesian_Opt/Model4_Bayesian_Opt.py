import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

# 导入模块四底层评估函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Model3_Lagrange_Dynamic_Sampling_and_Error_Calculation import process_18cut

# ==========================================================
# 全局配置
# ==========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = Path(__file__).resolve().parent

ANALYSIS_DIR = MODULE_DIR / "Analysis_Bayesian_Opt_Model4_Hor_Data"
HISTORY_CSV_PATH = os.path.join(ANALYSIS_DIR, "model4_HorMAE.csv")
SUMMARY_JSON_PATH = os.path.join(ANALYSIS_DIR, "model4_HorMAE.json")

BASE_DATA_DIR = PROJECT_ROOT / "ModelA_Virtual_Internal_Solitary_Wave_Data_Generation" / "V_Wave_Data_Hor"
DATA_SPLIT_SEED = 42
OPTIMIZER_SEED = 42
TARGET_MAX_EVALS = 60

TRAIN_SAMPLE_SIZE = 30
TEST_SAMPLE_SIZE = 30
PENALTY_LOSS = 9999.0

# W_C_THRESHOLD_CHOICES = [0.05, 0.10, 0.15, 0.20, 9999.0]
# V_RATIO_CHOICES = [0.2, 0.4, 0.6, 0.8, 1.0]
F_S_CHOICES = [0.2, 1, 2, 3, 4, 5]

W_MAE = 1
W_CI = 0

space = {
    "w_c_threshold": hp.uniform("w_c_threshold", 0.05, 0.5),
    "zeta_target": hp.uniform("zeta_target", -46.2, -19.7),
    "V_ratio": hp.uniform("V_ratio", 0.1, 1.0),
    "f_s": hp.choice("f_s", F_S_CHOICES),
}

# 运行期全局变量（由 main 初始化）
train_dirs = []
test_dirs = []


# ==========================================================
# 基础工具函数
# ==========================================================
def _calc_v_target(zeta_target, v_ratio):
    if zeta_target <= -23.64:
        v_max = 0.5
    else:
        v_max = -0.0157 * zeta_target + 0.1293
    return float(v_ratio) * float(v_max)


def _decode_choice_value(raw_value, choices):
    raw = float(raw_value)
    if raw.is_integer():
        idx = int(raw)
        if 0 <= idx < len(choices):
            return float(choices[idx])
    return raw


def _value_to_choice_idx(value, choices):
    value_f = float(value)
    if value_f.is_integer():
        idx = int(value_f)
        if 0 <= idx < len(choices):
            return idx

    for i, c in enumerate(choices):
        if abs(float(c) - value_f) < 1e-12:
            return i
    raise ValueError(f"值 {value} 不在候选集合 {choices} 中")


def _extract_error_value(result):
    if isinstance(result, dict):
        rel = result.get("rel_error")
        if rel is None:
            dz = float(result.get("Delta_Z_true", 0.0))
            if dz == 0.0:
                return PENALTY_LOSS
            rel = float(result.get("abs_error", result.get("J", PENALTY_LOSS))) / dz
            if not np.isfinite(rel):
                return PENALTY_LOSS
            return rel * 100.0
        else:
            rel = float(rel)
            if not np.isfinite(rel):
                return PENALTY_LOSS
            return rel
    return float(result)


def _run_one_dir(args):
    run_dir, w_c_threshold, v_target, zeta_target, f_s = args
    try:
        result = process_18cut(w_c_threshold, v_target, zeta_target, f_s, run_dir)
        return _extract_error_value(result)
    except Exception:
        return PENALTY_LOSS


# ==========================================================
# 数据划分与 Trials 编解码
# ==========================================================
def prepare_data_split():
    all_run_dirs = sorted(
        [
            os.path.join(BASE_DATA_DIR, d)
            for d in os.listdir(BASE_DATA_DIR)
            if os.path.isdir(os.path.join(BASE_DATA_DIR, d))
        ]
    )

    n_total = len(all_run_dirs)
    if n_total < 200:
        raise ValueError(
            f"当前仅有 {n_total} 组流场目录，需要至少 200 组以实现「前100组 / 后100组」划分。"
        )

    train_pool = all_run_dirs[:100]
    test_pool = all_run_dirs[100:200]

    rng_split = np.random.default_rng(DATA_SPLIT_SEED)
    train_pick = rng_split.choice(100, size=TRAIN_SAMPLE_SIZE, replace=False)
    test_pick = rng_split.choice(100, size=TEST_SAMPLE_SIZE, replace=False)

    train_dirs_local = [train_pool[i] for i in sorted(train_pick)]
    test_dirs_local = [test_pool[i] for i in sorted(test_pick)]
    return train_dirs_local, test_dirs_local


def build_history_dataframe(trials):
    rows = []
    for i, trial in enumerate(trials.trials, start=1):
        vals = trial["misc"]["vals"]
        result = trial["result"]

        loss = float(result["loss"])
        mae_pct = float(result.get("MAE_pct", np.nan))
        ci_width = float(result.get("CI_width", np.nan))
        w_mae = float(result.get("w_MAE", W_MAE))
        w_ci = float(result.get("w_CI", W_CI))

        w_c_threshold = float(vals["w_c_threshold"][0])
        zeta_target = float(vals["zeta_target"][0])
        v_ratio = float(vals["V_ratio"][0])
        f_s = _decode_choice_value(vals["f_s"][0], F_S_CHOICES)
        v_target = _calc_v_target(zeta_target, v_ratio)

        rows.append(
            {
                "eval_id": i,
                "w_c_threshold": w_c_threshold,
                "zeta_target": zeta_target,
                "V_ratio": v_ratio,
                "f_s": f_s,
                "V_target": v_target,
                "MAE_pct": mae_pct,
                "CI_width": ci_width,
                "w_MAE": w_mae,
                "w_CI": w_ci,
                "loss": loss,
                "is_penalty": int(loss >= PENALTY_LOSS),
            }
        )
    return pd.DataFrame(rows)


def _load_trials_from_history_csv(csv_path):
    trials = Trials()
    if not os.path.exists(csv_path):
        return trials, 0

    hist_df = pd.read_csv(csv_path)
    if hist_df.empty:
        return trials, 0

    docs = []
    for i, row in hist_df.reset_index(drop=True).iterrows():
        loss = float(row["loss"])
        mae_pct = row["MAE_pct"] if "MAE_pct" in row.index else np.nan
        ci_width = row["CI_width"] if "CI_width" in row.index else np.nan
        w_mae = float(row["w_MAE"]) if "w_MAE" in row.index and pd.notna(row["w_MAE"]) else W_MAE
        w_ci = float(row["w_CI"]) if "w_CI" in row.index and pd.notna(row["w_CI"]) else W_CI

        w_c_val = float(row["w_c_threshold"])  # 直接取数值
        v_ratio_val = float(row["V_ratio"])    # 直接取数值
        f_s_idx = _value_to_choice_idx(row["f_s"], F_S_CHOICES)

        docs.append(
            {
                "state": 2,
                "tid": int(i),
                "spec": None,
                "result": {
                    "loss": loss,
                    "status": STATUS_OK,
                    "MAE_pct": float(mae_pct) if pd.notna(mae_pct) else np.nan,
                    "CI_width": float(ci_width) if pd.notna(ci_width) else np.nan,
                    "w_MAE": w_mae,
                    "w_CI": w_ci,
                },
                "misc": {
                    "tid": int(i),
                    "cmd": ("domain_attachment", "FMinIter_Domain"),
                    "workdir": None,
                    "idxs": {
                        "w_c_threshold": [int(i)],
                        "zeta_target": [int(i)],
                        "V_ratio": [int(i)],
                        "f_s": [int(i)],
                    },
                    "vals": {
                        "w_c_threshold": [w_c_val],       # 改为刚才定义的数值变量
                        "zeta_target": [float(row["zeta_target"])],
                        "V_ratio": [v_ratio_val],         # 改为刚才定义的数值变量
                        "f_s": [f_s_idx],
                    },
                },
                "exp_key": None,
                "owner": None,
                "version": 0,
                "book_time": None,
                "refresh_time": None,
            }
        )

    trials.insert_trial_docs(docs)
    trials.refresh()
    return trials, len(docs)


# ==========================================================
# 目标函数与评估流程
# ==========================================================
def objective(params):
    w_c_threshold = params["w_c_threshold"]
    zeta_target = params["zeta_target"]
    f_s = params["f_s"]
    v_ratio = params["V_ratio"]
    v_target = _calc_v_target(zeta_target, v_ratio)

    n_workers = max(1, multiprocessing.cpu_count() - 1)
    args_list = [(run_dir, w_c_threshold, v_target, zeta_target, f_s) for run_dir in train_dirs]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_run_one_dir, args_list))

    raw_errors = [v for v in results if v != PENALTY_LOSS]
    if len(raw_errors) < 3:
        return {
            "loss": PENALTY_LOSS,
            "status": STATUS_OK,
            "MAE_pct": np.nan,
            "CI_width": np.nan,
            "w_MAE": W_MAE,
            "w_CI": W_CI,
        }

    errors = np.array(raw_errors)
    n_valid = len(errors)
    mae_pct = float(np.mean(errors))
    std_err = np.std(errors, ddof=1) if n_valid > 1 else 99.0
    ci_width = float(2 * 1.96 * (std_err / np.sqrt(n_valid)))
    weighted_loss = W_MAE * mae_pct + W_CI * ci_width

    print(
        f"[Model4] 本轮完成 | "
        f"w_c_threshold={w_c_threshold:.3f} m/s, "
        f"zeta_target={zeta_target:.2f}°, "
        f"V_ratio={v_ratio:.3f} -> V_target={v_target:.3f} m/s, "
        f"f_s={float(f_s):.1f} Hz | "
        f"N={n_valid} | 平均相对误差={mae_pct:.4f}% | 95%CI宽度={ci_width:.4f}% | loss={weighted_loss:.4f}"
    )

    return {
        "loss": weighted_loss,
        "status": STATUS_OK,
        "MAE_pct": mae_pct,
        "CI_width": ci_width,
        "w_MAE": W_MAE,
        "w_CI": W_CI,
    }


def run_test_evaluation(best_w_c, best_v_target, best_zeta, best_f_s):
    n_workers = max(1, multiprocessing.cpu_count() - 1)
    test_args_list = [(run_dir, best_w_c, best_v_target, best_zeta, best_f_s) for run_dir in test_dirs]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        test_results = list(executor.map(_run_one_dir, test_args_list))

    test_errors = np.array([v for v in test_results if v != PENALTY_LOSS])
    test_n = len(test_errors)
    if test_n == 0:
        return None

    test_mae_pct = float(np.mean(test_errors))
    test_std = float(np.std(test_errors, ddof=1)) if test_n > 1 else 0.0
    test_ci_width = float(2 * 1.96 * (test_std / np.sqrt(test_n)))
    test_weighted_loss = float(0.5 * test_mae_pct + 0.5 * test_ci_width)

    return {
        "test_N": int(test_n),
        "test_MAE_pct": test_mae_pct,
        "test_CI_width": test_ci_width,
        "test_weighted_loss": test_weighted_loss,
    }


def save_summary(
    history_df,
    loaded_evals,
    best_w_c,
    best_zeta,
    best_v_ratio,
    best_f_s,
    best_v_target,
    test_metrics,
):
    summary = {
        "max_evals": int(len(history_df)),
        "data_split_seed": int(DATA_SPLIT_SEED),
        "optimizer_seed": int(OPTIMIZER_SEED),
        "resumed_from_existing_csv": bool(loaded_evals > 0),
        "loaded_evals": int(loaded_evals),
        "best_params": {
            "w_c_threshold": float(best_w_c),
            "zeta_target": float(best_zeta),
            "V_ratio": float(best_v_ratio),
            "f_s": float(best_f_s),
            "V_target": float(best_v_target),
        },
        "best_loss": float(history_df["loss"].min()) if not history_df.empty else PENALTY_LOSS,
        "loss_formula": f"loss = {W_MAE:.2f} * MAE_pct + {W_CI:.2f} * CI_width",
        "loss_weights": {"w_MAE": W_MAE, "w_CI": W_CI},
        "train_sample_dirs": train_dirs,
        "test_sample_dirs": test_dirs,
        "test_metrics": test_metrics,
    }

    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    global train_dirs, test_dirs
    train_dirs, test_dirs = prepare_data_split()

    print("\n================= 阶段一：训练集优化 =================")
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    trials, loaded_evals = _load_trials_from_history_csv(HISTORY_CSV_PATH)
    if loaded_evals > 0:
        print(f"[+] 检测到已有历史记录：{loaded_evals} 轮，将在同一 CSV 上续跑。")

    if loaded_evals < TARGET_MAX_EVALS:
        fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=TARGET_MAX_EVALS,
            trials=trials,
            rstate=np.random.default_rng(OPTIMIZER_SEED),
        )
    else:
        print(f"[+] 已达到目标评估轮数 {TARGET_MAX_EVALS}，跳过新增评估。")

    history_df = build_history_dataframe(trials)
    history_df.to_csv(HISTORY_CSV_PATH, index=False, encoding="utf-8-sig")

    best_row = history_df.loc[history_df["loss"].idxmin()]
    best_w_c = float(best_row["w_c_threshold"])
    best_zeta = float(best_row["zeta_target"])
    best_v_ratio = float(best_row["V_ratio"])
    best_f_s = float(best_row["f_s"])
    best_v_target = _calc_v_target(best_zeta, best_v_ratio)

    print("\n[+] 训练完成！找到的最优抗流观测策略参数为:")
    print(f"  w_c_threshold : {best_w_c:.3f} m/s")
    print(f"  V_target      : {best_v_target:.3f} m/s")
    print(f"  zeta_target   : {best_zeta:.2f} °")
    print(f"  f_s           : {best_f_s:.1f} Hz")

    print("\n================= 阶段二：在后100组池中随机抽中的20组上验证 =================")
    test_metrics = run_test_evaluation(best_w_c, best_v_target, best_zeta, best_f_s)
    if test_metrics is None:
        print("[-] 测试集评估失败。")
    else:
        print("\n[+] 测试集盲测得分（百分比相对误差）:")
        print(f"  测试集 平均相对误差     : {test_metrics['test_MAE_pct']:.3f} %")
        print(f"  测试集 95% 置信区间宽度 : {test_metrics['test_CI_width']:.3f} %")
        print(f"  测试集综合 Loss         : {test_metrics['test_weighted_loss']:.3f}")

    save_summary(
        history_df=history_df,
        loaded_evals=loaded_evals,
        best_w_c=best_w_c,
        best_zeta=best_zeta,
        best_v_ratio=best_v_ratio,
        best_f_s=best_f_s,
        best_v_target=best_v_target,
        test_metrics=test_metrics,
    )

    print("\n[+] 贝叶斯优化结果已保存：")
    print(f"  - {HISTORY_CSV_PATH}")
    print(f"  - {SUMMARY_JSON_PATH}")


if __name__ == "__main__":
    main()

"""ModelF sampling strategy core.

This module defines the four principle-check sampling strategies used in
ModelF and the shared amplitude-estimation routine. It is intentionally
independent from ModelA generation code so the self-consistent ideal wave data
can be tested without changing earlier models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "Generated_Ideal_Wave_Data"
RESULTS_DIR = Path(__file__).resolve().parent / "Results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GLIDER_HORIZONTAL_SPEED = 0.22
DT = 1.0


@dataclass(frozen=True)
class Strategy:
    code: str
    label: str
    horizontal_speed: float
    mode: str


STRATEGIES = {
    "A": Strategy(
        code="A",
        label="A_Pre_Positioned_Zstar",
        horizontal_speed=0.0,
        mode="pre_positioned",
    ),
    "B1": Strategy(
        code="B1",
        label="B1_T0_Zstar",
        horizontal_speed=GLIDER_HORIZONTAL_SPEED,
        mode="t0_zstar",
    ),
    "B2": Strategy(
        code="B2",
        label="B2_T0_Zstar_Plus_Atrue",
        horizontal_speed=GLIDER_HORIZONTAL_SPEED,
        mode="t0_zstar_plus_a",
    ),
    "B3": Strategy(
        code="B3",
        label="B3_Early_Zstar_Then_Drift",
        horizontal_speed=GLIDER_HORIZONTAL_SPEED,
        mode="early_zstar",
    ),
}


def sorted_run_dirs(data_dir: Path = DEFAULT_DATA_DIR) -> list[Path]:
    return sorted([p for p in data_dir.iterdir() if p.is_dir()])


def load_wave_case(run_dir: Path) -> dict:
    z = np.load(run_dir / "z.npy")
    x_grid = np.load(run_dir / "x_grid.npy")
    y_grid = np.load(run_dir / "y_grid.npy")
    w_field = np.load(run_dir / "W_Vel_3D.npy")
    w_profile = np.load(run_dir / "W_profile.npy")

    with open(run_dir / "params.json", "r", encoding="utf-8") as fp:
        params = json.load(fp)

    if z[0] > z[-1]:
        z = np.flip(z)
        w_field = np.flip(w_field, axis=2)
        w_profile = np.flip(w_profile)

    interp_w = RegularGridInterpolator(
        (x_grid, y_grid, z),
        w_field,
        bounds_error=False,
        fill_value=0.0,
    )

    idx_star = int(np.argmax(np.abs(w_profile)))
    z_star = float(z[idx_star])
    w_star = float(w_profile[idx_star])
    h0 = float(params["h0"])
    cp = float(params["c0"])
    d_scale = float(params.get("D", 1000.0))
    a_true = h0 * abs(w_star)

    return {
        "run_dir": run_dir,
        "wave_id": run_dir.name,
        "z": z,
        "x_grid": x_grid,
        "y_grid": y_grid,
        "w_profile": w_profile,
        "interp_w": interp_w,
        "params": params,
        "z_star": z_star,
        "w_star": w_star,
        "h0": h0,
        "cp": cp,
        "d_scale": d_scale,
        "a_true": a_true,
        "z_min": float(z.min()),
        "z_max": float(z.max()),
    }


def sample_vertical_speed(case: dict, x_eff: float, z_pos: float) -> float:
    return float(case["interp_w"]((x_eff, 0.0, z_pos)))


def positive_lobe_bounds(t: np.ndarray, w: np.ndarray) -> tuple[int, int]:
    after_zero = np.where(t >= 0.0)[0]
    if len(after_zero) == 0:
        return 0, len(t)

    start = int(after_zero[0])
    positive = np.where((t >= 0.0) & (w > 0.0))[0]
    if len(positive) == 0:
        return start, start

    idx_max = int(positive[np.argmax(w[positive])])
    end = idx_max
    while end < len(w) - 1 and w[end] > 0.0:
        end += 1
    return start, end


def simulate_strategy(case: dict, strategy: Strategy, return_track: bool = False) -> dict:
    cp = case["cp"]
    d_scale = case["d_scale"]
    h0 = case["h0"]
    z_star = case["z_star"]
    w_star = case["w_star"]
    z_min = case["z_min"]
    z_max = case["z_max"]
    v_g = strategy.horizontal_speed
    v_rel = cp + v_g

    half_window = max(2600.0, 4.5 * d_scale / max(v_rel, 1e-6))
    t = np.arange(-half_window, half_window + DT, DT)
    x_eff = v_rel * t

    z_track = np.empty_like(t)
    w_track = np.empty_like(t)
    active = np.ones_like(t, dtype=bool)

    if strategy.mode == "pre_positioned":
        z_current = z_star
        for i, x_now in enumerate(x_eff):
            w_now = sample_vertical_speed(case, float(x_now), z_current)
            z_track[i] = z_current
            w_track[i] = w_now
            z_current = float(np.clip(z_current - w_now * DT, z_min, z_max))

    elif strategy.mode == "t0_zstar":
        z_current = z_star
        started = False
        for i, time_now in enumerate(t):
            if time_now < 0.0:
                z_track[i] = z_star
                w_track[i] = sample_vertical_speed(case, float(x_eff[i]), z_track[i])
                active[i] = False
            elif not started:
                started = True
                z_current = z_star
                w_now = sample_vertical_speed(case, float(x_eff[i]), z_current)
                z_track[i] = z_current
                w_track[i] = w_now
            else:
                w_now = sample_vertical_speed(case, float(x_eff[i]), z_current)
                z_track[i] = z_current
                w_track[i] = w_now
                z_current = float(np.clip(z_current - w_now * DT, z_min, z_max))

    elif strategy.mode == "t0_zstar_plus_a":
        z_target = float(np.clip(z_star + case["a_true"], z_min, z_max))
        z_current = z_target
        started = False
        for i, time_now in enumerate(t):
            if time_now < 0.0:
                z_track[i] = z_target
                w_track[i] = sample_vertical_speed(case, float(x_eff[i]), z_track[i])
                active[i] = False
            elif not started:
                started = True
                z_current = z_target
                w_now = sample_vertical_speed(case, float(x_eff[i]), z_current)
                z_track[i] = z_current
                w_track[i] = w_now
            else:
                w_now = sample_vertical_speed(case, float(x_eff[i]), z_current)
                z_track[i] = z_current
                w_track[i] = w_now
                z_current = float(np.clip(z_current - w_now * DT, z_min, z_max))

    elif strategy.mode == "early_zstar":
        lead_time = d_scale / max(v_rel, 1e-6)
        z_current = z_star
        for i, time_now in enumerate(t):
            if time_now < -lead_time:
                z_track[i] = z_star
                w_track[i] = sample_vertical_speed(case, float(x_eff[i]), z_star)
                active[i] = False
                continue

            w_now = sample_vertical_speed(case, float(x_eff[i]), z_current)
            z_track[i] = z_current
            w_track[i] = w_now
            z_current = float(np.clip(z_current - w_now * DT, z_min, z_max))

    else:
        raise ValueError(f"Unknown strategy mode: {strategy.mode}")

    start, end = positive_lobe_bounds(t, w_track)
    if end <= start + 1:
        dh_raw = 0.0
        duration = 0.0
        h_calc = 0.0
    else:
        dh_raw = float(np.trapezoid(w_track[start:end], x=t[start:end]))
        duration = float(t[end - 1] - t[start])
        h_calc = abs(dh_raw * (v_rel / cp) / w_star)

    abs_error = abs(h_calc - h0)
    rel_error = abs_error / h0 * 100.0 if h0 else float("inf")
    z_t0 = float(z_track[int(np.argmin(np.abs(t)))])

    result = {
        "wave_id": case["wave_id"],
        "strategy": strategy.code,
        "strategy_label": strategy.label,
        "true_h0": h0,
        "calc_h0": h_calc,
        "abs_error": abs_error,
        "rel_error": rel_error,
        "z_star": z_star,
        "w_star": w_star,
        "a_true": case["a_true"],
        "z_at_t0": z_t0,
        "z_t0_minus_zstar": z_t0 - z_star,
        "v_rel": v_rel,
        "duration": duration,
        "dh_raw": dh_raw,
        "integral_start_time": float(t[start]) if len(t) else np.nan,
        "integral_end_time": float(t[end - 1]) if end > start else np.nan,
    }

    if return_track:
        result["track"] = pd.DataFrame(
            {
                "Time": t,
                "X_eff": x_eff,
                "Z": z_track,
                "W": w_track,
                "Active": active,
            }
        )
        result["integral_slice"] = (start, end)

    return result


def run_batch(limit: int = 20, data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    rows = []
    for run_dir in sorted_run_dirs(data_dir)[:limit]:
        case = load_wave_case(run_dir)
        for strategy in STRATEGIES.values():
            rows.append(simulate_strategy(case, strategy, return_track=False))
    return pd.DataFrame(rows)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["strategy", "strategy_label"], as_index=False)["rel_error"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )

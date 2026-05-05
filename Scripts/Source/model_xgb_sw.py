"""XGBoost with Optuna, sliding-window validation, and sequential holdout.

The script is aligned with the article notebooks. It reports sliding-window
metrics, final sequential-holdout metrics, and two COS variants:

    COS_MAE  = 0.5 * (MAE_SW / MAE_final)  + 0.5 * (STD_SW / STD_final)
    COS_RMSE = 0.5 * (RMSE_SW / RMSE_final) + 0.5 * (STD_SW / STD_final)

Set TARGET_GP_NAME to select the Grand Prix:

    $env:TARGET_GP_NAME = "United States Grand Prix"
    python Scripts/Source/model_xgb_sw.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import scipy.stats as stats
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments.
    stats = None


TARGET_COL = "LapTime_seconds"
LAP_COL = "LapNumber"

NUM_COLS_BASE = [
    "TyreLife",
    "LapNumber",
    "Humidity_RBF_Median",
    "Pressure_RBF_Median",
    "TrackTemp_RBF_Median",
    "WindSpeed_RBF_Median",
    "TempDelta_RBF_Median",
    "LapTime_prev",
]

CAT_COLS = ["Driver", "Team", "pirelliCompound", "Year"]

HOLDOUT_RATIO = 0.20
WINDOW_RATIO = 0.20
WINDOW_TRAIN_RATIO = 0.80
WINDOW_STEP_RATIO = 0.20

USE_SAVED_XGB_PARAMS = True
OPTUNA_TRIALS = int(os.environ.get("OPTUNA_TRIALS", "100"))
BASE_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "eval_metric": "rmse",
    "seed": 42,
    "nthread": -1,
}


def safe_gp_name(gp_name: str) -> str:
    return gp_name.lower().replace(" ", "_")


def calc_stats(values):
    values = np.asarray(values, dtype=float)
    mean_value = float(np.mean(values))
    if len(values) > 1 and stats is not None:
        ci = stats.t.interval(0.95, len(values) - 1, loc=mean_value, scale=stats.sem(values))
    elif len(values) > 1:
        margin = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(len(values))
        ci = (mean_value - margin, mean_value + margin)
    else:
        ci = (mean_value, mean_value)
    return mean_value, float(ci[0]), float(ci[1])


def calc_holdout_ci(y_true, y_pred, n_bootstrap=1000, alpha=0.05, seed=42):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    rmse_point = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae_point = float(mean_absolute_error(y_true, y_pred))
    r2_point = float(r2_score(y_true, y_pred))

    if n < 2:
        return {"rmse": (rmse_point, rmse_point), "mae": (mae_point, mae_point), "r2": (r2_point, r2_point)}

    rng = np.random.default_rng(seed)
    rmse_samples, mae_samples, r2_samples = [], [], []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        pb = y_pred[idx]
        rmse_samples.append(np.sqrt(mean_squared_error(yb, pb)))
        mae_samples.append(mean_absolute_error(yb, pb))
        r2_value = r2_score(yb, pb)
        if np.isfinite(r2_value):
            r2_samples.append(r2_value)

    def percentile_ci(samples, point_value):
        if not samples:
            return point_value, point_value
        lower = float(np.percentile(samples, 100 * (alpha / 2)))
        upper = float(np.percentile(samples, 100 * (1 - alpha / 2)))
        return lower, upper

    return {
        "rmse": percentile_ci(rmse_samples, rmse_point),
        "mae": percentile_ci(mae_samples, mae_point),
        "r2": percentile_ci(r2_samples, r2_point),
    }


def calc_cos_metric(error_sliding, error_final, std_sliding, std_final, alpha=0.5, beta=0.5):
    error_sliding = float(error_sliding)
    error_final = float(error_final)
    std_sliding = float(std_sliding)
    std_final = float(std_final)

    if np.isclose(error_final, 0) or np.isclose(std_final, 0):
        return np.nan, error_sliding, error_final, std_sliding, std_final

    cos_value = alpha * (error_sliding / error_final) + beta * (std_sliding / std_final)
    return cos_value, error_sliding, error_final, std_sliding, std_final


def build_sliding_windows(n_laps, window_ratio, train_ratio, step_ratio):
    if n_laps < 2:
        raise ValueError("Insufficient data for sliding window validation.")

    window_size = max(2, min(int(np.ceil(n_laps * window_ratio)), n_laps))
    train_size = max(1, int(np.floor(window_size * train_ratio)))
    if train_size >= window_size:
        train_size = window_size - 1

    val_size = window_size - train_size
    step_size = max(1, int(np.ceil(window_size * step_ratio)))

    windows = []
    start = 0
    while start + window_size <= n_laps:
        windows.append((start, start + train_size, start + window_size))
        start += step_size

    last_start = n_laps - window_size
    if not windows or windows[-1][0] != last_start:
        windows.append((last_start, last_start + train_size, last_start + window_size))

    return windows, window_size, train_size, val_size, step_size


def build_sequential_split(df_base, valid_indices, holdout_ratio, lap_col):
    if lap_col not in df_base.columns:
        raise KeyError(f"Column '{lap_col}' not found.")

    lap_series = df_base.loc[valid_indices, lap_col]
    if lap_series.dropna().empty:
        raise ValueError("No valid lap values are available.")

    lap_min = int(np.floor(lap_series.min()))
    lap_max = int(np.floor(lap_series.max()))
    total_laps = lap_max - lap_min + 1

    holdout_laps = max(1, int(np.ceil(total_laps * holdout_ratio)))
    holdout_laps = min(holdout_laps, total_laps - 1)
    holdout_start_lap = lap_max - holdout_laps + 1
    model_end_lap = holdout_start_lap - 1

    model_mask = (lap_series >= lap_min) & (lap_series <= model_end_lap)
    holdout_mask = (lap_series >= holdout_start_lap) & (lap_series <= lap_max)
    model_idx = lap_series[model_mask].index
    holdout_idx = lap_series[holdout_mask].index

    if len(model_idx) == 0 or len(holdout_idx) == 0:
        raise ValueError("Invalid sequential split: modeling or holdout block is empty.")

    return lap_series, lap_min, lap_max, model_idx, holdout_idx, holdout_start_lap, model_end_lap, total_laps


def load_cleaned_data():
    target_gp_name = os.environ.get("TARGET_GP_NAME", "Bahrain Grand Prix")
    script_dir = Path(__file__).resolve().parent
    scripts_dir = script_dir.parent
    input_csv_path = scripts_dir / "ModelData" / target_gp_name / f"{safe_gp_name(target_gp_name)}_cleaned_data.csv"

    print(f"Loading cleaned data from:\n{input_csv_path}")
    if not input_csv_path.exists():
        raise FileNotFoundError(f"File not found: {input_csv_path}")

    return target_gp_name, scripts_dir, pd.read_csv(input_csv_path)


def prepare_modeling_frame(df_base):
    num_cols = [col for col in NUM_COLS_BASE if col in df_base.columns]
    cat_cols = [col for col in CAT_COLS if col in df_base.columns]

    X_raw = df_base[num_cols + cat_cols].copy()
    y_raw = df_base[TARGET_COL].copy()
    valid_indices = y_raw.dropna().index
    X_raw = X_raw.loc[valid_indices]
    y_raw = y_raw.loc[valid_indices]

    X_proc = X_raw.copy()
    X_proc[cat_cols] = X_proc[cat_cols].fillna("Missing")
    X_proc = pd.get_dummies(X_proc, columns=cat_cols, drop_first=False)

    return X_proc, y_raw, valid_indices, num_cols, cat_cols


def tune_or_load_params(params_path, windows, unique_laps, lap_model_sorted, X_model, y_model):
    if USE_SAVED_XGB_PARAMS and params_path.exists():
        print(f"Using saved XGBoost parameters: {params_path}")
        with params_path.open("r", encoding="utf-8") as file:
            loaded_params = json.load(file)

        best_n = int(loaded_params.get("n_estimators", 0))
        if best_n < 1:
            raise ValueError("Invalid saved XGBoost parameter file: missing n_estimators.")

        train_params = {key: value for key, value in loaded_params.items() if key != "n_estimators"}
        train_params = {**BASE_XGB_PARAMS, **train_params}
        return train_params, best_n, {**train_params, "n_estimators": best_n}

    def objective(trial):
        params = {
            **BASE_XGB_PARAMS,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        scores = []
        for start, split, end in windows:
            train_laps = unique_laps[start:split]
            val_laps = unique_laps[split:end]
            train_mask = lap_model_sorted.isin(train_laps)
            val_mask = lap_model_sorted.isin(val_laps)
            X_train, y_train = X_model.loc[train_mask], y_model.loc[train_mask]
            X_val, y_val = X_model.loc[val_mask], y_model.loc[val_mask]
            if len(X_train) == 0 or len(X_val) == 0:
                raise optuna.exceptions.TrialPruned()

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=2000,
                evals=[(dval, "validation")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )
            preds = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
            scores.append(np.sqrt(mean_squared_error(y_val, preds)))

        return float(np.mean(scores))

    print(f"Running Optuna tuning with {OPTUNA_TRIALS} trials...")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    train_params = {**BASE_XGB_PARAMS, **study.best_params}

    print("Calibrating n_estimators through sliding-window early stopping...")
    best_iterations = []
    for start, split, end in windows:
        train_laps = unique_laps[start:split]
        val_laps = unique_laps[split:end]
        train_mask = lap_model_sorted.isin(train_laps)
        val_mask = lap_model_sorted.isin(val_laps)
        dtrain = xgb.DMatrix(X_model.loc[train_mask], label=y_model.loc[train_mask])
        dval = xgb.DMatrix(X_model.loc[val_mask], label=y_model.loc[val_mask])
        booster = xgb.train(
            params=train_params,
            dtrain=dtrain,
            num_boost_round=5000,
            evals=[(dval, "validation")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        best_iterations.append(booster.best_iteration + 1)

    best_n = max(1, int(np.median(best_iterations)))
    final_params = {**train_params, "n_estimators": best_n}
    params_path.parent.mkdir(parents=True, exist_ok=True)
    with params_path.open("w", encoding="utf-8") as file:
        json.dump(final_params, file, indent=4)
    print(f"Saved XGBoost parameters to: {params_path}")
    return train_params, best_n, final_params


def main():
    target_gp_name, scripts_dir, laps_cleaned = load_cleaned_data()
    df_base = laps_cleaned.copy()
    X_proc, y_raw, valid_indices, num_cols, cat_cols = prepare_modeling_frame(df_base)

    print("--- XGBOOST: OPTUNA + SLIDING WINDOW + SEQUENTIAL HOLDOUT ---")
    print(f"Grand Prix: {target_gp_name}")
    print(f"Numerical features: {num_cols}")
    print(f"Categorical features: {cat_cols}")

    (
        lap_series,
        lap_min,
        lap_max,
        model_idx,
        holdout_idx,
        holdout_start_lap,
        model_end_lap,
        total_laps,
    ) = build_sequential_split(df_base, valid_indices, HOLDOUT_RATIO, LAP_COL)

    X_model = X_proc.loc[model_idx].copy()
    y_model = y_raw.loc[model_idx].copy()
    X_holdout = X_proc.loc[holdout_idx].copy()
    y_holdout = y_raw.loc[holdout_idx].copy()

    medians = X_model[num_cols].median()
    X_model[num_cols] = X_model[num_cols].fillna(medians)
    X_holdout[num_cols] = X_holdout[num_cols].fillna(medians)

    model_laps = lap_series.loc[model_idx]
    model_order_idx = model_laps.sort_values(kind="mergesort").index
    X_model = X_model.loc[model_order_idx].reset_index(drop=True)
    y_model = y_model.loc[model_order_idx].reset_index(drop=True)
    lap_model_sorted = model_laps.loc[model_order_idx].reset_index(drop=True)
    unique_laps = np.sort(pd.to_numeric(lap_model_sorted, errors="coerce").dropna().unique())

    windows, window_size, train_size, val_size, step_size = build_sliding_windows(
        len(unique_laps), WINDOW_RATIO, WINDOW_TRAIN_RATIO, WINDOW_STEP_RATIO
    )

    print("\n--- Sequential split ---")
    print(f"Total laps: {total_laps} (LapNumber {lap_min}-{lap_max})")
    print(f"Modeling block: laps {lap_min}-{model_end_lap} | records={len(X_model)}")
    print(f"Holdout block: laps {holdout_start_lap}-{lap_max} | records={len(X_holdout)}")
    print(f"Sliding windows: {len(windows)} | window={window_size} | train/val={train_size}/{val_size} | step={step_size}")

    params_path = scripts_dir / "Results" / "xgboost" / "sw" / "params" / f"{safe_gp_name(target_gp_name)}_xgb_params_sw.json"
    train_params, best_n, final_params = tune_or_load_params(params_path, windows, unique_laps, lap_model_sorted, X_model, y_model)

    print("\n--- Training final XGBoost model ---")
    dmodel_full = xgb.DMatrix(X_model, label=y_model)
    final_model = xgb.train(params=train_params, dtrain=dmodel_full, num_boost_round=best_n, verbose_eval=False)
    print(f"Selected n_estimators: {best_n}")
    print(final_params)

    results = {"window": [], "rmse": [], "mae": [], "r2": [], "std": []}

    print("\n--- Sliding-window validation ---")
    for i, (start, split, end) in enumerate(windows, start=1):
        train_laps = unique_laps[start:split]
        val_laps = unique_laps[split:end]
        train_mask = lap_model_sorted.isin(train_laps)
        val_mask = lap_model_sorted.isin(val_laps)
        X_train, y_train = X_model.loc[train_mask], y_model.loc[train_mask]
        X_val, y_val = X_model.loc[val_mask], y_model.loc[val_mask]
        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError(f"Window {i}: empty train or validation fold.")

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        booster = xgb.train(
            params=train_params,
            dtrain=dtrain,
            num_boost_round=best_n,
            evals=[(dval, "validation")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        preds = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))

        rmse_value = float(np.sqrt(mean_squared_error(y_val, preds)))
        mae_value = float(mean_absolute_error(y_val, preds))
        r2_value = float(r2_score(y_val, preds))
        std_value = float(np.std(np.asarray(y_val) - np.asarray(preds), ddof=1)) if len(y_val) > 1 else 0.0

        results["window"].append(i)
        results["rmse"].append(rmse_value)
        results["mae"].append(mae_value)
        results["r2"].append(r2_value)
        results["std"].append(std_value)

        print(
            f"Window {i:02d} | train laps {int(train_laps[0])}-{int(train_laps[-1])} | "
            f"val laps {int(val_laps[0])}-{int(val_laps[-1])} | "
            f"RMSE={rmse_value:.4f} | MAE={mae_value:.4f} | R2={r2_value:.4f}"
        )

    rmse_m, rmse_l, rmse_u = calc_stats(results["rmse"])
    mae_m, mae_l, mae_u = calc_stats(results["mae"])
    r2_m, r2_l, r2_u = calc_stats(results["r2"])
    std_m, _, _ = calc_stats(results["std"])

    dholdout = xgb.DMatrix(X_holdout, label=y_holdout)
    preds_holdout = final_model.predict(dholdout)

    holdout_ci = calc_holdout_ci(y_holdout.to_numpy(), preds_holdout)
    rmse_holdout = float(np.sqrt(mean_squared_error(y_holdout, preds_holdout)))
    mae_holdout = float(mean_absolute_error(y_holdout, preds_holdout))
    r2_holdout = float(r2_score(y_holdout, preds_holdout))
    std_holdout = float(np.std(np.asarray(y_holdout) - np.asarray(preds_holdout), ddof=1)) if len(y_holdout) > 1 else 0.0

    cos_mae, mae_sw, mae_final, std_sw, std_final = calc_cos_metric(mae_m, mae_holdout, std_m, std_holdout)
    cos_rmse, rmse_sw, rmse_final, _, _ = calc_cos_metric(rmse_m, rmse_holdout, std_m, std_holdout)

    cos_mae_windows = 0.5 * (np.array(results["mae"]) / mae_holdout) + 0.5 * (np.array(results["std"]) / std_holdout)
    cos_rmse_windows = 0.5 * (np.array(results["rmse"]) / rmse_holdout) + 0.5 * (np.array(results["std"]) / std_holdout)
    _, cos_mae_l, cos_mae_u = calc_stats(cos_mae_windows)
    _, cos_rmse_l, cos_rmse_u = calc_stats(cos_rmse_windows)

    print("\n--- Sliding-window summary (indicative CI) ---")
    print("NOTE: sliding windows overlap; these confidence intervals are descriptive.")
    print(f"RMSE: {rmse_m:.4f} | 95% CI: [{rmse_l:.4f}, {rmse_u:.4f}]")
    print(f"MAE:  {mae_m:.4f} | 95% CI: [{mae_l:.4f}, {mae_u:.4f}]")
    print(f"R2:   {r2_m:.4f} | 95% CI: [{r2_l:.4f}, {r2_u:.4f}]")

    print("\n--- Sequential holdout ---")
    print(f"RMSE: {rmse_holdout:.4f} | 95% CI: [{holdout_ci['rmse'][0]:.4f}, {holdout_ci['rmse'][1]:.4f}]")
    print(f"MAE:  {mae_holdout:.4f} | 95% CI: [{holdout_ci['mae'][0]:.4f}, {holdout_ci['mae'][1]:.4f}]")
    print(f"R2:   {r2_holdout:.4f} | 95% CI: [{holdout_ci['r2'][0]:.4f}, {holdout_ci['r2'][1]:.4f}]")
    print(f"COS_MAE:  {cos_mae:.4f} | 95% CI: [{cos_mae_l:.4f}, {cos_mae_u:.4f}]")
    print(f"          MAE SW/final={mae_sw:.4f}/{mae_final:.4f} | STD SW/final={std_sw:.4f}/{std_final:.4f}")
    print(f"COS_RMSE: {cos_rmse:.4f} | 95% CI: [{cos_rmse_l:.4f}, {cos_rmse_u:.4f}]")
    print(f"          RMSE SW/final={rmse_sw:.4f}/{rmse_final:.4f} | STD SW/final={std_sw:.4f}/{std_final:.4f}")


if __name__ == "__main__":
    main()

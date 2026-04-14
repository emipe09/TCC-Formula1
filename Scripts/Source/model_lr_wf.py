import pandas as pd
import numpy as np
import os
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure the target Grand Prix to load corresponding data
target_gp_name = os.environ.get('TARGET_GP_NAME', 'Bahrain Grand Prix')
safe_gp_name = target_gp_name.lower().replace(' ', '_')

# Define paths to read the ModelData file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
model_data_dir = os.path.join(parent_dir, 'ModelData', target_gp_name)
input_csv_path = os.path.join(model_data_dir, f"{safe_gp_name}_cleaned_data.csv")

print(f"Loading cleaned data from:\n{input_csv_path}")
if not os.path.exists(input_csv_path):
    raise FileNotFoundError(f"File not found: {input_csv_path}. Run script_model_data.py first.")

# Load the DataFrame
laps_cleaned = pd.read_csv(input_csv_path)

# Base feature preparation
target_col = 'LapTime_seconds'
df_base = laps_cleaned.copy() 

num_cols_base = [
    'TyreLife', 'LapNumber',
    'Humidity_RBF_Median','Pressure_RBF_Median', 'TrackTemp_RBF_Median', 
    'WindSpeed_RBF_Median',
    'TempDelta_RBF_Median', 'LapTime_prev'
]

cat_cols = ['Driver', 'Team', 'pirelliCompound', 'Year']

num_cols_base = [c for c in num_cols_base if c in df_base.columns]
cat_cols = [c for c in cat_cols if c in df_base.columns]

X_raw = df_base[num_cols_base + cat_cols].copy()
y_raw = df_base[target_col].copy()

valid_indices = y_raw.dropna().index
X_raw = X_raw.loc[valid_indices]
y_raw = y_raw.loc[valid_indices]

X_proc = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
X_proc = X_proc.fillna(X_proc.mean())

LAP_COL = "LapNumber"
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
SLIDE_STEP = 1

lap_min = int(X_raw[LAP_COL].min())
lap_max = int(X_raw[LAP_COL].max())
total_laps = lap_max - lap_min + 1

window_ratios_to_test = [0.05, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.60]
comparison_results_lr = []
fold_details_lr = []

confidence = 0.95
z_score = stats.norm.ppf((1 + confidence) / 2)

results_root_dir = os.path.join(parent_dir, "Results", "linear_regression", "wf", safe_gp_name)

def save_folds_by_ratio_txt(df_folds, output_dir, section_title):
    if df_folds.empty:
        return

    os.makedirs(output_dir, exist_ok=True)

    for ratio in sorted(df_folds["Ratio"].unique(), key=lambda x: float(x.replace("%", ""))):
        df_ratio = df_folds[df_folds["Ratio"] == ratio].sort_values(by="RMSE", ascending=True)
        ratio_file = ratio.replace("%", "pct")
        file_path = os.path.join(output_dir, f"folds_ratio_{ratio_file}.txt")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"{section_title}\n")
            f.write(f"Ratio: {ratio}\n")
            f.write(df_ratio.to_string(index=False))
            f.write("\n")

    print(f"Text files saved at: {output_dir}")

for ratio in window_ratios_to_test:
    window_laps = max(2, int(np.round(total_laps * ratio)))
    train_size = max(1, int(np.floor(window_laps * TRAIN_RATIO)))
    test_size = max(1, window_laps - train_size)
    
    metrics = {"rmse": [], "mae": [], "r2": []}
    
    window_start = lap_min
    window_last_start = lap_max - window_laps + 1
    
    while window_start <= window_last_start:
        window_end = window_start + window_laps - 1
        train_start = window_start
        train_end = train_start + train_size - 1
        test_start = train_end + 1
        test_end = window_end

        mask_train = (X_raw[LAP_COL] >= train_start) & (X_raw[LAP_COL] <= train_end)
        mask_test = (X_raw[LAP_COL] >= test_start) & (X_raw[LAP_COL] <= test_end)

        if mask_train.any() and mask_test.any():
            X_train_wf = X_proc.loc[mask_train]
            y_train_wf = y_raw.loc[mask_train]
            X_test_wf = X_proc.loc[mask_test]
            y_test_wf = y_raw.loc[mask_test]

            model = LinearRegression()
            model.fit(X_train_wf, y_train_wf)
            
            preds = model.predict(X_test_wf)

            rmse_fold = np.sqrt(mean_squared_error(y_test_wf, preds))
            mae_fold = mean_absolute_error(y_test_wf, preds)
            r2_fold = r2_score(y_test_wf, preds)

            metrics["rmse"].append(rmse_fold)
            metrics["mae"].append(mae_fold)
            metrics["r2"].append(r2_fold)

            fold_details_lr.append({
                "Ratio": f"{ratio:.0%}",
                "Fold": len(metrics["rmse"]),
                "Voltas_Train": f"{train_start}-{train_end}",
                "Voltas_Test": f"{test_start}-{test_end}",
                "Train_Regs": len(X_train_wf),
                "Test_Regs": len(X_test_wf),
                "RMSE": round(rmse_fold, 4),
                "MAE": round(mae_fold, 4),
                "R2": round(r2_fold, 4)
            })
            
        window_start += SLIDE_STEP
    
    if len(metrics["rmse"]) > 1:
        res_row = {
            "Ratio": f"{ratio:.0%}",
            "Laps": window_laps,
            "Folds": len(metrics["rmse"])
        }
        
        for m in ["rmse", "mae", "r2"]:
            arr = np.array(metrics[m])
            mean_val = np.mean(arr)
            std_val = np.std(arr, ddof=1)
            err = z_score * (std_val / np.sqrt(len(arr)))
            
            res_row[f"{m.upper()}_Media"] = round(mean_val, 4)
            res_row[f"IC_95%_{m.upper()}"] = [round(mean_val - err, 4), round(mean_val + err, 4)]
            
        comparison_results_lr.append(res_row)

df_comparison_lr = pd.DataFrame(comparison_results_lr)
print("\n--- COMPARAÃ‡ÃƒO LINEAR REGRESSION WALK FORWARD (WF) ---")
print(df_comparison_lr.to_string(index=False))

if len(fold_details_lr) > 0:
    df_folds_lr = pd.DataFrame(fold_details_lr).sort_values(by="RMSE", ascending=True)
    print("\n--- TODOS OS FOLDS (WF GERAL) | MELHOR -> PIOR RMSE ---")
    print(df_folds_lr.to_string(index=False))

    general_output_dir = os.path.join(results_root_dir, "wf_geral")
    save_folds_by_ratio_txt(df_folds_lr, general_output_dir, "Folds WF Geral")

print("\n--- COMPARAÃ‡ÃƒO LINEAR REGRESSION WALK FORWARD (WF) | APENAS 2025 ---")
if "Year" not in df_base.columns:
    print("Column 'Year' not found. Could not run the 2025 section.")
else:
    year_series = df_base.loc[valid_indices, "Year"].astype(str).str.strip()
    mask_2025 = year_series == "2025"

    if not mask_2025.any():
        print("There are no records de 2025 after limpeza para executar a evaluation dedicada.")
    else:
        X_raw_2025 = X_raw.loc[mask_2025]

        lap_min_2025 = int(X_raw_2025[LAP_COL].min())
        lap_max_2025 = int(X_raw_2025[LAP_COL].max())
        total_laps_2025 = lap_max_2025 - lap_min_2025 + 1

        comparison_results_lr_2025 = []
        fold_details_lr_2025 = []

        for ratio in window_ratios_to_test:
            window_laps = max(2, int(np.round(total_laps_2025 * ratio)))
            train_size = max(1, int(np.floor(window_laps * TRAIN_RATIO)))
            test_size = max(1, window_laps - train_size)

            metrics = {"rmse": [], "mae": [], "r2": []}

            window_start = lap_min_2025
            window_last_start = lap_max_2025 - window_laps + 1

            while window_start <= window_last_start:
                window_end = window_start + window_laps - 1
                train_start = window_start
                train_end = train_start + train_size - 1
                test_start = train_end + 1
                test_end = window_end

                mask_test = mask_2025 & (X_raw[LAP_COL] >= test_start) & (X_raw[LAP_COL] <= test_end)

                if mask_test.any():
                    # Train temporalmente justo:
                    # - Usa 2022/2023/2024 completos.
                    # - Em 2025, usa apenas laps anteriores ao start da janela de test.
                    mask_train_temporal = (~mask_2025) | (mask_2025 & (X_raw[LAP_COL] < test_start))

                    test_idx = X_raw.index[mask_test]
                    train_idx = X_raw.index[mask_train_temporal]

                    if len(train_idx) == 0:
                        window_start += SLIDE_STEP
                        continue

                    X_train_wf = X_proc.loc[train_idx]
                    y_train_wf = y_raw.loc[train_idx]
                    X_test_wf = X_proc.loc[test_idx]
                    y_test_wf = y_raw.loc[test_idx]

                    train_2025_count = int((mask_2025.loc[train_idx]).sum())
                    train_prev_years_count = len(train_idx) - train_2025_count

                    model = LinearRegression()
                    model.fit(X_train_wf, y_train_wf)

                    preds = model.predict(X_test_wf)

                    rmse_fold = np.sqrt(mean_squared_error(y_test_wf, preds))
                    mae_fold = mean_absolute_error(y_test_wf, preds)
                    r2_fold = r2_score(y_test_wf, preds)

                    metrics["rmse"].append(rmse_fold)
                    metrics["mae"].append(mae_fold)
                    metrics["r2"].append(r2_fold)

                    fold_details_lr_2025.append({
                        "Ratio": f"{ratio:.0%}",
                        "Fold": len(metrics["rmse"]),
                        "Voltas_Test_2025": f"{test_start}-{test_end}",
                        "Train_Regs_Total": len(X_train_wf),
                        "Train_Regs_2022_2024": train_prev_years_count,
                        "Train_Regs_2025_Passado": train_2025_count,
                        "Test_Regs_2025": len(X_test_wf),
                        "RMSE": round(rmse_fold, 4),
                        "MAE": round(mae_fold, 4),
                        "R2": round(r2_fold, 4)
                    })

                window_start += SLIDE_STEP

            if len(metrics["rmse"]) > 1:
                res_row = {
                    "Ratio": f"{ratio:.0%}",
                    "Laps": window_laps,
                    "Folds": len(metrics["rmse"])
                }

                for m in ["rmse", "mae", "r2"]:
                    arr = np.array(metrics[m])
                    mean_val = np.mean(arr)
                    std_val = np.std(arr, ddof=1)
                    err = z_score * (std_val / np.sqrt(len(arr)))

                    res_row[f"{m.upper()}_Media"] = round(mean_val, 4)
                    res_row[f"IC_95%_{m.upper()}"] = [round(mean_val - err, 4), round(mean_val + err, 4)]

                comparison_results_lr_2025.append(res_row)

        if len(comparison_results_lr_2025) == 0:
            print("Could not generate enough valid windows for 2025.")
        else:
            df_comparison_lr_2025 = pd.DataFrame(comparison_results_lr_2025)
            print(
                f"Evaluated year: 2025 | Laps considered: {total_laps_2025} "
                f"({lap_min_2025}-{lap_max_2025}) | Train: todos os anos "
                f"(except the 2025 test window in each fold)"
            )
            print(df_comparison_lr_2025.to_string(index=False))

            if len(fold_details_lr_2025) > 0:
                df_folds_lr_2025 = pd.DataFrame(fold_details_lr_2025).sort_values(by="RMSE", ascending=True)
                print("\n--- TODOS OS FOLDS (WF 2025) | MELHOR -> PIOR RMSE ---")
                print(df_folds_lr_2025.to_string(index=False))

                y2025_output_dir = os.path.join(results_root_dir, "wf_2025")
                save_folds_by_ratio_txt(df_folds_lr_2025, y2025_output_dir, "Folds WF 2025")





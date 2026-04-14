import pandas as pd
import numpy as np
import os
import scipy.stats as stats
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

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

# Base feature preparation para XGBoost
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

X_proc = X_raw.copy()
X_proc[cat_cols] = X_proc[cat_cols].fillna("Missing")
X_proc = pd.get_dummies(X_proc, columns=cat_cols, drop_first=True)
X_proc[num_cols_base] = X_proc[num_cols_base].fillna(X_proc[num_cols_base].median())

LAP_COL = "LapNumber"
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
SLIDE_STEP = 1


window_ratios_to_test = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
confidence = 0.95
z_score = stats.norm.ppf((1 + confidence) / 2)

comparison_results_xgb = []

lap_min = int(X_raw[LAP_COL].min())
lap_max = int(X_raw[LAP_COL].max())
total_laps = lap_max - lap_min + 1

json_candidates = [
    os.path.join(parent_dir, 'Results', 'xgboost', 'cv', 'params', f"{safe_gp_name}_xgb_params_cv.json"),
    os.path.join(parent_dir, 'Results', 'xgboost', 'cv', 'params', f"{safe_gp_name}_xgb_params.json"),
    os.path.join(parent_dir, 'Utils', f"{safe_gp_name}_xgb_params.json"),
]

json_path = None
for candidate in json_candidates:
    if os.path.exists(candidate):
        json_path = candidate
        break

if json_path is not None:
    with open(json_path, 'r') as f:
        optuna_params_train = json.load(f)
    num_boost_round = optuna_params_train.pop("n_estimators", 500)
    print(f"Parameters Optuna carregados de: {json_path}")
else:
    print("No previous parameter file found. Using defaults.")
    optuna_params_train = {"objective": "reg:squarederror", "tree_method": "hist", "eval_metric": "rmse", "seed": 42}
    num_boost_round = 500

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

            dtr = xgb.DMatrix(X_train_wf, label=y_train_wf)
            dte = xgb.DMatrix(X_test_wf, label=y_test_wf)
            
            bst = xgb.train(
                params=optuna_params_train, 
                dtrain=dtr, 
                num_boost_round=num_boost_round
            )
            
            preds = bst.predict(dte)
            
            metrics["rmse"].append(np.sqrt(mean_squared_error(y_test_wf, preds)))
            metrics["mae"].append(mean_absolute_error(y_test_wf, preds))
            metrics["r2"].append(r2_score(y_test_wf, preds))
            
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
            err = z_score * (np.std(arr, ddof=1) / np.sqrt(len(arr)))
            res_row[f"{m.upper()}_Media"] = round(mean_val, 4)
            res_row[f"IC_95%_{m.upper()}"] = [round(mean_val - err, 4), round(mean_val + err, 4)]
            
        comparison_results_xgb.append(res_row)

df_comparison_xgb = pd.DataFrame(comparison_results_xgb)
print("\n--- COMPARAÃ‡ÃƒO XGBOOST WALK FORWARD (WF) ---")
print(df_comparison_xgb.to_string(index=False))




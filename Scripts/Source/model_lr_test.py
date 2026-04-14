import pandas as pd
import numpy as np
import os
import scipy.stats as stats
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure the target Grand Prix to load data
target_gp_name = os.environ.get('TARGET_GP_NAME', 'Bahrain Grand Prix')
safe_gp_name = target_gp_name.lower().replace(' ', '_')

# Caminhos
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
model_data_dir = os.path.join(parent_dir, 'ModelData', target_gp_name)
input_csv_path = os.path.join(model_data_dir, f"{safe_gp_name}_cleaned_data.csv")

print(f"Loading cleaned data from:\n{input_csv_path}")
if not os.path.exists(input_csv_path):
    raise FileNotFoundError(f"File not found: {input_csv_path}. Run script_model_data.py first.")

laps_cleaned = pd.read_csv(input_csv_path)

def calc_stats(values):
    mean_v = np.mean(values)
    if len(values) > 1:
        ci = stats.t.interval(0.95, len(values)-1, loc=mean_v, scale=stats.sem(values))
    else:
        ci = (mean_v, mean_v)
    return mean_v, ci[0], ci[1]

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

# Generate identical random seeds for a fair N_SPLITS comparison
N_SPLITS = 5
np.random.seed(42)
seeds_do_baseline = [np.random.randint(0, 100000) for _ in range(N_SPLITS)] 

print("--- PREPARATION FOR LINEAR REGRESSION ---")

X_proc = X_raw.copy()
X_proc[cat_cols] = X_proc[cat_cols].fillna("Missing")
X_proc = pd.get_dummies(X_proc, columns=cat_cols, drop_first=True)

X = X_proc
y = y_raw.copy()

# Sequential split by lap range: 80% for modeling and 20% final holdout.
LAP_COL = "LapNumber"
HOLDOUT_RATIO = 0.20

if LAP_COL not in df_base.columns:
    raise KeyError(f"Column '{LAP_COL}' not found for sequential split.")

lap_series = df_base.loc[valid_indices, LAP_COL]

if lap_series.dropna().empty:
    raise ValueError("No valid LapNumber values available for sequential split.")

lap_min = int(np.floor(lap_series.min()))
lap_max = int(np.floor(lap_series.max()))
total_laps_track = lap_max - lap_min + 1

if total_laps_track < 2:
    raise ValueError("Insufficient number of laps to split train and holdout.")

holdout_laps = max(1, int(np.ceil(total_laps_track * HOLDOUT_RATIO)))
holdout_laps = min(holdout_laps, total_laps_track - 1)
holdout_start_lap = lap_max - holdout_laps + 1
model_end_lap = holdout_start_lap - 1

model_mask = (lap_series >= lap_min) & (lap_series <= model_end_lap)
holdout_mask = (lap_series >= holdout_start_lap) & (lap_series <= lap_max)

model_idx = lap_series[model_mask].index
holdout_idx = lap_series[holdout_mask].index

if len(model_idx) == 0 or len(holdout_idx) == 0:
    raise ValueError(
        "Invalid sequential split: train/modeling ou holdout ficou vazio."
    )

X_model = X.loc[model_idx]
y_model = y.loc[model_idx]
X_holdout = X.loc[holdout_idx]
y_holdout = y.loc[holdout_idx]

print("\n--- SEQUENTIAL SPLIT DEFINED ---")
print(f"Track: {target_gp_name}")
print(f"Total laps considered: {total_laps_track} (LapNumber {lap_min}-{lap_max})")
print(f"Holdout (final 20%): laps {holdout_start_lap}-{lap_max} | Total laps: {holdout_laps} | Records: {len(X_holdout)}")
print(f"Modeling (initial 80%): laps {lap_min}-{model_end_lap} | Total laps: {total_laps_track - holdout_laps} | Records: {len(X_model)}")


print("\n--- CROSS-VALIDATION (3-FOLD) ON MODELING SET (80%) ---")
kf = KFold(n_splits=3, shuffle=True, random_state=42)
cv_rmse = []
cv_mae = []
cv_r2 = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_model), start=1):
    X_tr_fold = X_model.iloc[train_idx]
    y_tr_fold = y_model.iloc[train_idx]
    X_va_fold = X_model.iloc[val_idx]
    y_va_fold = y_model.iloc[val_idx]

    model_fold = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ]
    )
    model_fold.fit(X_tr_fold, y_tr_fold)
    preds_fold = model_fold.predict(X_va_fold)

    rmse_fold = np.sqrt(mean_squared_error(y_va_fold, preds_fold))
    mae_fold = mean_absolute_error(y_va_fold, preds_fold)
    r2_fold = r2_score(y_va_fold, preds_fold)

    cv_rmse.append(rmse_fold)
    cv_mae.append(mae_fold)
    cv_r2.append(r2_fold)
    print(f"Fold {fold} | RMSE={rmse_fold:.4f} | MAE={mae_fold:.4f} | R2={r2_fold:.4f}")

rmse_cv_m, rmse_cv_l, rmse_cv_u = calc_stats(cv_rmse)
mae_cv_m, mae_cv_l, mae_cv_u = calc_stats(cv_mae)
r2_cv_m, r2_cv_l, r2_cv_u = calc_stats(cv_r2)

print("\nCV Summary (80% modeling):")
print(f"RMSE Mean: {rmse_cv_m:.4f} IC95%: [{rmse_cv_l:.4f}, {rmse_cv_u:.4f}]")
print(f"MAE Mean: {mae_cv_m:.4f} IC95%: [{mae_cv_l:.4f}, {mae_cv_u:.4f}]")
print(f"R2 Mean: {r2_cv_m:.4f} IC95%: [{r2_cv_l:.4f}, {r2_cv_u:.4f}]")

print("\n--- TRAINING FINAL MODEL (LINEAR REGRESSION) ---")
modelo_final = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ]
)
modelo_final.fit(X_model, y_model)

print("\n--- STARTING PAIRED COMPARISON (LINEAR REGRESSION) ---")
results_lr = {"seed_used": [], "rmse": [], "mae": [], "r2": []}

for i, seed in enumerate(seeds_do_baseline):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_model, y_model, test_size=0.20, random_state=seed, shuffle=True
    )
    print(
        f"Split {i+1} | Seed {seed} | Internal train (80% of modeling set): {len(X_tr)} records | "
        f"Test interno (20% de modeling): {len(X_te)} records"
    )
    model_split = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ]
    )
    model_split.fit(X_tr, y_tr)
    preds = model_split.predict(X_te)
    rmse_val = np.sqrt(mean_squared_error(y_te, preds))
    mae_val  = mean_absolute_error(y_te, preds)
    r2_val   = r2_score(y_te, preds)

    results_lr["seed_used"].append(seed)
    results_lr["rmse"].append(rmse_val)
    results_lr["mae"].append(mae_val)
    results_lr["r2"].append(r2_val)
    print(f"Split {i+1} | Seed {seed} | RMSE={rmse_val:.4f} | R2={r2_val:.4f}")

rmse_m_lr, rmse_l_lr, rmse_u_lr = calc_stats(results_lr["rmse"])
mae_m_lr, mae_l_lr, mae_u_lr = calc_stats(results_lr["mae"])
r2_m_lr, r2_l_lr, r2_u_lr = calc_stats(results_lr["r2"])

print("\n--- FINAL RESULT LINEAR REGRESSION ---")
print(f"RMSE Mean: {rmse_m_lr:.4f} IC95%: [{rmse_l_lr:.4f}, {rmse_u_lr:.4f}]")
print(f"MAE Mean: {mae_m_lr:.4f} IC95%: [{mae_l_lr:.4f}, {mae_u_lr:.4f}]")
print(f"R2 Mean: {r2_m_lr:.4f} IC95%: [{r2_l_lr:.4f}, {r2_u_lr:.4f}]")

print("\n--- FINAL TEST ON SEQUENTIAL HOLDOUT (20%) ---")
print(f"Final test (holdout): laps {holdout_start_lap}-{lap_max} | Records: {len(X_holdout)}")
preds_holdout = modelo_final.predict(X_holdout)
rmse_holdout = np.sqrt(mean_squared_error(y_holdout, preds_holdout))
mae_holdout = mean_absolute_error(y_holdout, preds_holdout)
r2_holdout = r2_score(y_holdout, preds_holdout)

print(f"Holdout RMSE: {rmse_holdout:.4f}")
print(f"Holdout MAE: {mae_holdout:.4f}")
print(f"Holdout R2: {r2_holdout:.4f}")

print("\n--- HOLDOUT APENAS DE 2025 ---")
X_holdout_2025 = X_holdout[X_holdout['Year_2025'] == 1]
y_holdout_2025 = y_holdout[X_holdout['Year_2025'] == 1]
preds_holdout_2025 = modelo_final.predict(X_holdout_2025)
rmse_holdout_2025 = np.sqrt(mean_squared_error(y_holdout_2025, preds_holdout_2025))
mae_holdout_2025 = mean_absolute_error(y_holdout_2025, preds_holdout_2025)
r2_holdout_2025 = r2_score(y_holdout_2025, preds_holdout_2025)

print(f"Holdout 2025 RMSE: {rmse_holdout_2025:.4f}")
print(f"Holdout 2025 MAE: {mae_holdout_2025:.4f}")
print(f"Holdout 2025 R2: {r2_holdout_2025:.4f}")





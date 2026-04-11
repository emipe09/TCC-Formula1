import pandas as pd
import numpy as np
import os
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configurar o Grande Prêmio desejado para carregar os dados correspondentes
target_gp_name = 'Bahrain Grand Prix'
safe_gp_name = target_gp_name.lower().replace(' ', '_')

# Definir os caminhos para ler o arquivo do ModelData
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
model_data_dir = os.path.join(parent_dir, 'ModelData', target_gp_name)
input_csv_path = os.path.join(model_data_dir, f"{safe_gp_name}_cleaned_data.csv")

print(f"Carregando dados limpos de:\n{input_csv_path}")
if not os.path.exists(input_csv_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {input_csv_path}. Execute o script_model_data.py primeiro.")

# Carregar o DataFrame
laps_cleaned = pd.read_csv(input_csv_path)

# Preparação das colunas de base
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

confidence = 0.95
z_score = stats.norm.ppf((1 + confidence) / 2)

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
            std_val = np.std(arr, ddof=1)
            err = z_score * (std_val / np.sqrt(len(arr)))
            
            res_row[f"{m.upper()}_Media"] = round(mean_val, 4)
            res_row[f"IC_95%_{m.upper()}"] = [round(mean_val - err, 4), round(mean_val + err, 4)]
            
        comparison_results_lr.append(res_row)

df_comparison_lr = pd.DataFrame(comparison_results_lr)
print("\n--- COMPARAÇÃO LINEAR REGRESSION WALK FORWARD (WF) ---")
print(df_comparison_lr.to_string(index=False))
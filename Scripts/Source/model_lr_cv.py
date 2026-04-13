import pandas as pd
import numpy as np
import os
import scipy.stats as stats
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configurar o Grande Premio para carregar os dados
target_gp_name = os.environ.get('TARGET_GP_NAME', 'Bahrain Grand Prix')
safe_gp_name = target_gp_name.lower().replace(' ', '_')

# Caminhos
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
model_data_dir = os.path.join(parent_dir, 'ModelData', target_gp_name)
input_csv_path = os.path.join(model_data_dir, f"{safe_gp_name}_cleaned_data.csv")

print(f"Carregando dados limpos de:\n{input_csv_path}")
if not os.path.exists(input_csv_path):
    raise FileNotFoundError(f"Arquivo nao encontrado: {input_csv_path}. Execute o script_model_data.py primeiro.")

laps_cleaned = pd.read_csv(input_csv_path)

def calc_stats(values):
    mean_v = np.mean(values)
    if len(values) > 1:
        ci = stats.t.interval(0.95, len(values)-1, loc=mean_v, scale=stats.sem(values))
    else:
        ci = (mean_v, mean_v)
    return mean_v, ci[0], ci[1]

def calc_holdout_ci(y_true, y_pred, n_bootstrap=1000, alpha=0.05, seed=42):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    if n < 2:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {
            'rmse': (rmse, rmse),
            'mae': (mae, mae),
            'r2': (r2, r2),
        }

    rng = np.random.default_rng(seed)
    rmse_samples = []
    mae_samples = []
    r2_samples = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        pb = y_pred[idx]

        rmse_samples.append(np.sqrt(mean_squared_error(yb, pb)))
        mae_samples.append(mean_absolute_error(yb, pb))

        try:
            r2b = r2_score(yb, pb)
        except ValueError:
            r2b = np.nan
        if np.isfinite(r2b):
            r2_samples.append(r2b)

    def percentile_ci(samples, point_value):
        if len(samples) == 0:
            return point_value, point_value
        lower = float(np.percentile(samples, 100 * (alpha / 2)))
        upper = float(np.percentile(samples, 100 * (1 - alpha / 2)))
        return lower, upper

    rmse_point = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_point = mean_absolute_error(y_true, y_pred)
    r2_point = r2_score(y_true, y_pred)

    return {
        'rmse': percentile_ci(rmse_samples, rmse_point),
        'mae': percentile_ci(mae_samples, mae_point),
        'r2': percentile_ci(r2_samples, r2_point),
    }

def fit_predict_linear_regression(X_train, y_train, X_eval):
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train)
    X_eval_imp = imputer.transform(X_eval)

    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_eval_scaled = scaler.transform(X_eval_imp)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    return model.predict(X_eval_scaled), model, imputer, scaler

target_col = 'LapTime_seconds'
df_base = laps_cleaned.copy()

num_cols_base = [
    'TyreLife', 'LapNumber',
    'Humidity_RBF_Median', 'Pressure_RBF_Median', 'TrackTemp_RBF_Median',
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

print("--- PREPARACAO PARA REGRESSAO LINEAR (DRIVER ONE-HOT) ---")

X_proc = X_raw.copy()
X_proc[cat_cols] = X_proc[cat_cols].fillna('Missing')
X_proc = pd.get_dummies(X_proc, columns=cat_cols, drop_first=True)

X = X_proc
y = y_raw.copy()

# Split sequencial por faixa de voltas: 80% para modelagem e 20% holdout final.
LAP_COL = 'LapNumber'
HOLDOUT_RATIO = 0.20

if LAP_COL not in df_base.columns:
    raise KeyError(f"Coluna '{LAP_COL}' nao encontrada para split sequencial.")

lap_series = df_base.loc[valid_indices, LAP_COL]

if lap_series.dropna().empty:
    raise ValueError('Nao ha valores validos de LapNumber para aplicar split sequencial.')

lap_min = int(np.floor(lap_series.min()))
lap_max = int(np.floor(lap_series.max()))
total_laps_track = lap_max - lap_min + 1

if total_laps_track < 2:
    raise ValueError('Numero de voltas insuficiente para separar treino e holdout.')

holdout_laps = max(1, int(np.ceil(total_laps_track * HOLDOUT_RATIO)))
holdout_laps = min(holdout_laps, total_laps_track - 1)
holdout_start_lap = lap_max - holdout_laps + 1
model_end_lap = holdout_start_lap - 1

model_mask = (lap_series >= lap_min) & (lap_series <= model_end_lap)
holdout_mask = (lap_series >= holdout_start_lap) & (lap_series <= lap_max)

model_idx = lap_series[model_mask].index
holdout_idx = lap_series[holdout_mask].index

if len(model_idx) == 0 or len(holdout_idx) == 0:
    raise ValueError('Split sequencial invalido: treino/modelagem ou holdout ficou vazio.')

X_model = X.loc[model_idx]
y_model = y.loc[model_idx]
X_holdout = X.loc[holdout_idx]
y_holdout = y.loc[holdout_idx]

print('\n--- SPLIT SEQUENCIAL DEFINIDO ---')
print(f'Pista: {target_gp_name}')
print(f'Total de voltas consideradas: {total_laps_track} (LapNumber {lap_min}-{lap_max})')
print(f'Holdout (20% final): voltas {holdout_start_lap}-{lap_max} | Total de voltas: {holdout_laps} | Registros: {len(X_holdout)}')
print(f'Modelagem (80% iniciais): voltas {lap_min}-{model_end_lap} | Total de voltas: {total_laps_track - holdout_laps} | Registros: {len(X_model)}')

print('\n--- AVALIACAO INTERNA POR K-FOLD (DENTRO DOS 80%) ---')
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results_lr = {'fold': [], 'rmse': [], 'mae': [], 'r2': []}

for i, (train_idx, val_idx) in enumerate(kf.split(X_model), start=1):
    X_tr = X_model.iloc[train_idx]
    y_tr = y_model.iloc[train_idx]
    X_va = X_model.iloc[val_idx]
    y_va = y_model.iloc[val_idx]

    preds, _, _, _ = fit_predict_linear_regression(X_tr, y_tr, X_va)

    rmse_val = np.sqrt(mean_squared_error(y_va, preds))
    mae_val = mean_absolute_error(y_va, preds)
    r2_val = r2_score(y_va, preds)

    results_lr['fold'].append(i)
    results_lr['rmse'].append(rmse_val)
    results_lr['mae'].append(mae_val)
    results_lr['r2'].append(r2_val)

    print(f'Fold {i} | treino={len(X_tr)} | val={len(X_va)} | RMSE={rmse_val:.4f} | R2={r2_val:.4f}')

print('\n--- TREINANDO MODELO FINAL (BASE DE MODELAGEM 80%) ---')
_, modelo_final, imputer_final, scaler_final = fit_predict_linear_regression(X_model, y_model, X_model)

X_holdout_imp = imputer_final.transform(X_holdout)
X_holdout_scaled = scaler_final.transform(X_holdout_imp)
preds_holdout = modelo_final.predict(X_holdout_scaled)

rmse_holdout = np.sqrt(mean_squared_error(y_holdout, preds_holdout))
mae_holdout = mean_absolute_error(y_holdout, preds_holdout)
r2_holdout = r2_score(y_holdout, preds_holdout)
holdout_ci = calc_holdout_ci(y_holdout.to_numpy(), preds_holdout)

rmse_m, rmse_l, rmse_u = calc_stats(results_lr['rmse'])
mae_m, mae_l, mae_u = calc_stats(results_lr['mae'])
r2_m, r2_l, r2_u = calc_stats(results_lr['r2'])

print('\n--- RESULTADO FINAL REGRESSAO LINEAR ---')
print(f'RMSE Medio: {rmse_m:.4f} IC95%: [{rmse_l:.4f}, {rmse_u:.4f}]')
print(f'MAE Medio: {mae_m:.4f} IC95%: [{mae_l:.4f}, {mae_u:.4f}]')
print(f'R2 Medio: {r2_m:.4f} IC95%: [{r2_l:.4f}, {r2_u:.4f}]')

print('\n--- TESTE FINAL NO HOLDOUT SEQUENCIAL (20%) ---')
print(f'Teste final (holdout): voltas {holdout_start_lap}-{lap_max} | Registros: {len(X_holdout)}')
print(f'Holdout RMSE: {rmse_holdout:.4f}')
print(f'Holdout MAE: {mae_holdout:.4f}')
print(f'Holdout R2: {r2_holdout:.4f}')
print(f"Holdout RMSE IC95%: [{holdout_ci['rmse'][0]:.4f}, {holdout_ci['rmse'][1]:.4f}]")
print(f"Holdout MAE IC95%: [{holdout_ci['mae'][0]:.4f}, {holdout_ci['mae'][1]:.4f}]")
print(f"Holdout R2 IC95%: [{holdout_ci['r2'][0]:.4f}, {holdout_ci['r2'][1]:.4f}]")

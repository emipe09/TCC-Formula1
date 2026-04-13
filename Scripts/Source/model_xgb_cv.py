import pandas as pd
import numpy as np
import os
import json
import scipy.stats as stats
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configurar o Grande Prêmio para carregar os dados
target_gp_name = os.environ.get('TARGET_GP_NAME', 'Bahrain Grand Prix')
safe_gp_name = target_gp_name.lower().replace(' ', '_')

# Caminhos
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
model_data_dir = os.path.join(parent_dir, 'ModelData', target_gp_name)
input_csv_path = os.path.join(model_data_dir, f"{safe_gp_name}_cleaned_data.csv")

print(f"Carregando dados limpos de:\n{input_csv_path}")
if not os.path.exists(input_csv_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {input_csv_path}. Execute o script_model_data.py primeiro.")

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
            "rmse": (rmse, rmse),
            "mae": (mae, mae),
            "r2": (r2, r2),
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
        "rmse": percentile_ci(rmse_samples, rmse_point),
        "mae": percentile_ci(mae_samples, mae_point),
        "r2": percentile_ci(r2_samples, r2_point),
    }

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

# Gerar sementes aleatórias idênticas para a mesma comparação de N_SPLITS
N_SPLITS = 5
np.random.seed(42)
seeds_do_baseline = [np.random.randint(0, 100000) for _ in range(N_SPLITS)] 

print("--- PREPARAÇÃO PARA XGBOOST (SEM ONE_HOT_THRESH, DRIVER ONE-HOT) ---")

X_proc = X_raw.copy()
X_proc[cat_cols] = X_proc[cat_cols].fillna("Missing")
X_proc = pd.get_dummies(X_proc, columns=cat_cols, drop_first=True)
X_proc[num_cols_base] = X_proc[num_cols_base].fillna(X_proc[num_cols_base].median())

X = X_proc
y = y_raw.copy()

# Split sequencial por faixa de voltas: 80% para modelagem e 20% holdout final.
LAP_COL = "LapNumber"
HOLDOUT_RATIO = 0.20

if LAP_COL not in df_base.columns:
    raise KeyError(f"Coluna '{LAP_COL}' não encontrada para split sequencial.")

lap_series = df_base.loc[valid_indices, LAP_COL]

if lap_series.dropna().empty:
    raise ValueError("Não há valores válidos de LapNumber para aplicar split sequencial.")

lap_min = int(np.floor(lap_series.min()))
lap_max = int(np.floor(lap_series.max()))
total_laps_track = lap_max - lap_min + 1

if total_laps_track < 2:
    raise ValueError("Número de voltas insuficiente para separar treino e holdout.")

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
        "Split sequencial inválido: treino/modelagem ou holdout ficou vazio."
    )

X_model = X.loc[model_idx]
y_model = y.loc[model_idx]
X_holdout = X.loc[holdout_idx]
y_holdout = y.loc[holdout_idx]

print("\n--- SPLIT SEQUENCIAL DEFINIDO ---")
print(f"Pista: {target_gp_name}")
print(f"Total de voltas consideradas: {total_laps_track} (LapNumber {lap_min}-{lap_max})")
print(f"Holdout (20% final): voltas {holdout_start_lap}-{lap_max} | Total de voltas: {holdout_laps} | Registros: {len(X_holdout)}")
print(f"Modelagem (80% iniciais): voltas {lap_min}-{model_end_lap} | Total de voltas: {total_laps_track - holdout_laps} | Registros: {len(X_model)}")


print("\n--- TUNAGEM OPTUNA ---")
def objective(trial):
    param = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "eval_metric": "rmse",
        "seed": 42,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    rmse_scores = []
    
    for train_idx, val_idx in kf.split(X_model):
        dtr = xgb.DMatrix(X_model.iloc[train_idx], label=y_model.iloc[train_idx])
        dva = xgb.DMatrix(X_model.iloc[val_idx], label=y_model.iloc[val_idx])

        bst = xgb.train(params=param, dtrain=dtr, num_boost_round=2000, 
                        evals=[(dva, "validation")], early_stopping_rounds=50, verbose_eval=False)

        preds = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
        rmse_scores.append(np.sqrt(mean_squared_error(y_model.iloc[val_idx], preds)))

    return float(np.mean(rmse_scores))

study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=40, show_progress_bar=False)

best_params = study.best_params
best_params_train = {
    "objective": "reg:squarederror", "tree_method": "hist", "eval_metric": "rmse", "seed": 42,
    **best_params
}

print("\n--- CALIBRANDO N_ESTIMATORS (xgb.cv) COM BEST PARAMS ---")
dtrain_full = xgb.DMatrix(X_model, label=y_model)

cv_results = xgb.cv(
    params=best_params_train, dtrain=dtrain_full, num_boost_round=5000,
    nfold=3, metrics="rmse", early_stopping_rounds=100, seed=42, verbose_eval=False
)

best_n = cv_results.shape[0]

final_params = {
    **best_params,
    "objective": "reg:squarederror", "tree_method": "hist", "eval_metric": "rmse", "seed": 42,
    "n_estimators": best_n
}

print("\n--- TREINANDO MODELO FINAL (BASE DE MODELAGEM 80%) ---")
dmodel_full = xgb.DMatrix(X_model, label=y_model)
modelo_final = xgb.train(
    params=best_params_train,
    dtrain=dmodel_full,
    num_boost_round=best_n,
    verbose_eval=False
)

print("\nParâmetros Finais Otimizados:")
print(final_params)

# --------------- SALVANDO PARAMETROS EM Results/xgboost/cv/params ---------------
params_dir = os.path.join(parent_dir, 'Results', 'xgboost', 'cv', 'params')
os.makedirs(params_dir, exist_ok=True)
json_path = os.path.join(params_dir, f"{safe_gp_name}_xgb_params_cv.json")

with open(json_path, 'w') as f:
    json.dump(final_params, f, indent=4)
print(f"\n[SUCESSO] Parametros do Optuna salvos em:\n{json_path}")
# -----------------------------------------------

print("\n--- INICIANDO COMPARAÇÃO PAREADA (XGBOOST OPTUNA + EARLY STOPPING POR SPLIT) ---")
results_xgb = {"seed_usada": [], "rmse": [], "mae": [], "r2": []}

for i, seed in enumerate(seeds_do_baseline):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_model, y_model, test_size=0.20, random_state=seed, shuffle=True
    )
    print(
        f"Split {i+1} | Seed {seed} | Treino interno (80% de modelagem): {len(X_tr)} registros | "
        f"Teste interno (20% de modelagem): {len(X_te)} registros"
    )
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dte = xgb.DMatrix(X_te, label=y_te)

    bst = xgb.train(
        params=best_params_train, dtrain=dtr, num_boost_round=5000,
        evals=[(dte, "validation")], early_stopping_rounds=100, verbose_eval=False
    )

    preds = bst.predict(dte, iteration_range=(0, bst.best_iteration + 1))
    rmse_val = np.sqrt(mean_squared_error(y_te, preds))
    mae_val  = mean_absolute_error(y_te, preds)
    r2_val   = r2_score(y_te, preds)

    results_xgb["seed_usada"].append(seed)
    results_xgb["rmse"].append(rmse_val)
    results_xgb["mae"].append(mae_val)
    results_xgb["r2"].append(r2_val)
    print(f"Split {i+1} | Seed {seed} | RMSE={rmse_val:.4f} | R2={r2_val:.4f}")

rmse_m_xgb, rmse_l_xgb, rmse_u_xgb = calc_stats(results_xgb["rmse"])
mae_m_xgb, mae_l_xgb, mae_u_xgb = calc_stats(results_xgb["mae"])
r2_m_xgb, r2_l_xgb, r2_u_xgb = calc_stats(results_xgb["r2"])

print("\n--- RESULTADO FINAL XGBOOST ---")
print(f"RMSE Médio: {rmse_m_xgb:.4f} IC95%: [{rmse_l_xgb:.4f}, {rmse_u_xgb:.4f}]")
print(f"MAE Médio: {mae_m_xgb:.4f} IC95%: [{mae_l_xgb:.4f}, {mae_u_xgb:.4f}]")
print(f"R2 Médio: {r2_m_xgb:.4f} IC95%: [{r2_l_xgb:.4f}, {r2_u_xgb:.4f}]")

print("\n--- TESTE FINAL NO HOLDOUT SEQUENCIAL (20%) ---")
print(f"Teste final (holdout): voltas {holdout_start_lap}-{lap_max} | Registros: {len(X_holdout)}")
dholdout_final = xgb.DMatrix(X_holdout, label=y_holdout)

preds_holdout = modelo_final.predict(dholdout_final)
rmse_holdout = np.sqrt(mean_squared_error(y_holdout, preds_holdout))
mae_holdout = mean_absolute_error(y_holdout, preds_holdout)
r2_holdout = r2_score(y_holdout, preds_holdout)
holdout_ci = calc_holdout_ci(y_holdout.to_numpy(), preds_holdout)

print(f"Holdout RMSE: {rmse_holdout:.4f} IC95%: [{holdout_ci['rmse'][0]:.4f}, {holdout_ci['rmse'][1]:.4f}]")
print(f"Holdout MAE: {mae_holdout:.4f} IC95%: [{holdout_ci['mae'][0]:.4f}, {holdout_ci['mae'][1]:.4f}]")
print(f"Holdout R2: {r2_holdout:.4f} IC95%: [{holdout_ci['r2'][0]:.4f}, {holdout_ci['r2'][1]:.4f}]")
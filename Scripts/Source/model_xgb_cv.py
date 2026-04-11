import pandas as pd
import numpy as np
import os
import json
import scipy.stats as stats
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configurar o Grande Prêmio para carregar os dados
target_gp_name = 'Bahrain Grand Prix'
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

X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y, test_size=0.20, random_state=42
)

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_holdout, label=y_holdout)

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
    
    for train_idx, val_idx in kf.split(X_train):
        dtr = xgb.DMatrix(X_train.iloc[train_idx], label=y_train.iloc[train_idx])
        dva = xgb.DMatrix(X_train.iloc[val_idx], label=y_train.iloc[val_idx])

        bst = xgb.train(params=param, dtrain=dtr, num_boost_round=2000, 
                        evals=[(dva, "validation")], early_stopping_rounds=50, verbose_eval=False)

        preds = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
        rmse_scores.append(np.sqrt(mean_squared_error(y_train.iloc[val_idx], preds)))

    return float(np.mean(rmse_scores))

study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=40, show_progress_bar=False)

best_params = study.best_params
best_params_train = {
    "objective": "reg:squarederror", "tree_method": "hist", "eval_metric": "rmse", "seed": 42,
    **best_params
}

print("\n--- CALIBRANDO N_ESTIMATORS (xgb.cv) COM BEST PARAMS ---")
dtrain_full = xgb.DMatrix(X_train, label=y_train)

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

print("\nParâmetros Finais Otimizados:")
print(final_params)

# --------------- SALVANDO EM Utils ---------------
utils_dir = os.path.join(parent_dir, 'Utils')
os.makedirs(utils_dir, exist_ok=True)
json_path = os.path.join(utils_dir, f"{safe_gp_name}_xgb_params.json")

with open(json_path, 'w') as f:
    json.dump(final_params, f, indent=4)
print(f"\n[SUCESSO] Parâmetros do Optuna salvos em:\n{json_path}")
# -----------------------------------------------

print("\n--- INICIANDO COMPARAÇÃO PAREADA (XGBOOST OPTUNA + EARLY STOPPING POR SPLIT) ---")
results_xgb = {"seed_usada": [], "rmse": [], "mae": [], "r2": []}

for i, seed in enumerate(seeds_do_baseline):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=seed, shuffle=True)
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
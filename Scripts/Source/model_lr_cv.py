import pandas as pd
import numpy as np
import os
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

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

print("\n--- PREPARAÇÃO E VALIDAÇÃO ROBUSTA DO BASELINE ---")

target_col = 'LapTime_seconds'
df_base = laps_cleaned.copy() 

num_cols_base = [
    'TyreLife', 'LapNumber',
    'Humidity_RBF_Median','Pressure_RBF_Median', 'TrackTemp_RBF_Median', 
    'WindSpeed_RBF_Median',
    'TempDelta_RBF_Median', 'LapTime_prev'
]

cat_cols_base = ['Driver', 'Team', 'pirelliCompound', 'Year']

num_cols_base = [c for c in num_cols_base if c in df_base.columns]
cat_cols_base = [c for c in cat_cols_base if c in df_base.columns]

X_raw = df_base[num_cols_base + cat_cols_base].copy()
y_raw = df_base[target_col].copy()

valid_indices = y_raw.dropna().index
X_raw = X_raw.loc[valid_indices]
y_raw = y_raw.loc[valid_indices]

X_encoded = pd.get_dummies(X_raw, columns=cat_cols_base, drop_first=True)
y_final = y_raw

print(f"Dados codificados. Shape X: {X_encoded.shape}, Shape y: {y_final.shape}")

np.random.seed(42)

results_baseline = {
    'seed_usada': [], 
    'rmse': [],
    'mae': [],
    'r2': []
}

N_SPLITS = 5

for i in range(N_SPLITS):
    seed_da_rodada = np.random.randint(0, 100000)
    
    print(f"\nSplit {i+1}/{N_SPLITS} | Seed Sorteada: {seed_da_rodada}")
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_encoded, y_final, 
        test_size=0.20, 
        random_state=seed_da_rodada, 
        shuffle=True
    )
    
    imputer = SimpleImputer(strategy='mean')
    X_tr_imp = pd.DataFrame(imputer.fit_transform(X_tr_raw), columns=X_tr_raw.columns)
    X_te_imp = pd.DataFrame(imputer.transform(X_te_raw), columns=X_te_raw.columns)

    scaler = StandardScaler()
    X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr_imp), columns=X_tr_imp.columns)
    X_te_scaled = pd.DataFrame(scaler.transform(X_te_imp), columns=X_te_imp.columns)
    
    model = LinearRegression()
    model.fit(X_tr_scaled, y_tr)
    
    preds = model.predict(X_te_scaled)
    
    rmse_val = np.sqrt(mean_squared_error(y_te, preds))
    mae_val = mean_absolute_error(y_te, preds)
    r2_val = r2_score(y_te, preds)
    
    results_baseline['seed_usada'].append(seed_da_rodada)
    results_baseline['rmse'].append(rmse_val)
    results_baseline['mae'].append(mae_val)
    results_baseline['r2'].append(r2_val)
    
    print(f"   Resultado: RMSE={rmse_val:.4f} | MAE={mae_val:.4f} | R2={r2_val:.4f}")

def calc_stats(values):
    mean_v = np.mean(values)
    ci = stats.t.interval(0.95, len(values)-1, loc=mean_v, scale=stats.sem(values))
    return mean_v, ci[0], ci[1]

rmse_m, rmse_l, rmse_u = calc_stats(results_baseline['rmse'])
mae_m, mae_l, mae_u = calc_stats(results_baseline['mae'])
r2_m, r2_l, r2_u = calc_stats(results_baseline['r2'])

mean_rmse_lr, lower_rmse_lr, upper_rmse_lr = rmse_m, rmse_l, rmse_u
mean_mae_lr, lower_mae_lr, upper_mae_lr = mae_m, mae_l, mae_u
mean_r2_lr, lower_r2_lr, upper_r2_lr = r2_m, r2_l, r2_u

print("\n--- RESULTADOS FINAIS DO BASELINE ROBUSTO ---")
print(f"RMSE: {rmse_m:.4f}  IC95%: [{rmse_l:.4f}, {rmse_u:.4f}]")
print(f"MAE:  {mae_m:.4f}  IC95%: [{mae_l:.4f}, {mae_u:.4f}]")
print(f"R²:   {r2_m:.4f}  IC95%: [{r2_l:.4f}, {r2_u:.4f}]")

df_res_base = pd.DataFrame(results_baseline)
import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.stats as stats
import statsmodels.api as sm
import xgboost as xgb
import optuna
import shap
from sklearn.ensemble import RandomForestRegressor

start_year_analysis = 2022
end_year_analysis = 2026
target_gp_name = 'Saudi Arabian Grand Prix'
laps_dir = r'C:\Users\mpoli\Desktop\UFOP\9 PERIODO\TCC\Data\Saudi Arabia\Race\Laps'
weather_dir = r'C:\Users\mpoli\Desktop\UFOP\9 PERIODO\TCC\Data\Saudi Arabia\Race\Weather'
safe_gp_name = target_gp_name.lower().replace(' ', '_')

all_laps_data_by_year = {}
all_weather_data_by_year = {}
all_results_data_by_year = {}

print(f"--- Carregando dados locais de {target_gp_name} ---")

for year in range(start_year_analysis, end_year_analysis):
    laps_file = os.path.join(laps_dir, f"{safe_gp_name}_laps_{year}.csv")
    weather_file = os.path.join(weather_dir, f"{safe_gp_name}_weather_{year}.csv")
    
    if os.path.exists(laps_file) and os.path.exists(weather_file):
        print(f"  [CSV] Carregando e convertendo temporada {year}...")
        
        df_laps = pd.read_csv(laps_file)
        df_weather = pd.read_csv(weather_file)
        
        df_laps['Time'] = pd.to_timedelta(df_laps['Time'])
        df_weather['Time'] = pd.to_timedelta(df_weather['Time'])
        
        if 'LapTime' in df_laps.columns:
            df_laps['LapTime'] = pd.to_timedelta(df_laps['LapTime'])
        
        df_laps = df_laps.sort_values('Time')
        df_weather = df_weather.sort_values('Time')
        
        all_laps_data_by_year[year] = df_laps
        all_weather_data_by_year[year] = df_weather
    else:
        print(f"  [!] Arquivos de {year} não encontrados em '{laps_dir}'.")

print("\nIniciando combinação de múltiplos anos...")
lista_de_dataframes_anuais_laps = []
lista_de_dataframes_anuais_weather = []

for year in range(start_year_analysis, end_year_analysis):
    if year in all_laps_data_by_year:
        print(f"Carregando dados de {target_gp_name} {year}...")
        laps_df = all_laps_data_by_year[year].copy()
        weather_df = all_weather_data_by_year[year].copy()
        
        laps_df['Year'] = year
        weather_df['Year'] = year
        
        lista_de_dataframes_anuais_laps.append(laps_df)
        lista_de_dataframes_anuais_weather.append(weather_df)
    else:
        print(f"Sem dados para {target_gp_name} {year}.")

if not lista_de_dataframes_anuais_laps:
    print("Nenhum dado encontrado para processar. Encerrando a análise.")
else:
    combined_laps_df = pd.concat(lista_de_dataframes_anuais_laps, ignore_index=True)
    print(f"\nDados de {len(lista_de_dataframes_anuais_laps)} anos combinados com sucesso!")
    print(f"Total de {len(combined_laps_df)} voltas carregadas.")
    
    clean_laps_df = combined_laps_df[combined_laps_df['IsAccurate'] == True].copy()
    
    cols_to_check = ['LapTime_seconds', 'TyreLife']
    if 'pirelliCompound' in clean_laps_df.columns:
        cols_to_check.append('pirelliCompound')
        
    clean_laps_df.dropna(subset=cols_to_check, inplace=True)
    clean_laps_df['Year'] = clean_laps_df['Year'].astype('category')

    print(f"Analisando {len(clean_laps_df)} voltas 'limpas' de todos os anos.")

if not lista_de_dataframes_anuais_weather:
    print("Nenhum dado meteorológico encontrado para o intervalo de anos. Encerrando a análise.")
else:
    combined_weather_df = pd.concat(lista_de_dataframes_anuais_weather, ignore_index=True)
    print(f"Total de {len(combined_weather_df)} registros meteorológicos carregados.")
    print(f"\nDados de {len(lista_de_dataframes_anuais_weather)} anos combinados com sucesso!")

clean_laps_df = clean_laps_df.sort_values(['Year','Driver','Stint','LapNumber'])
clean_laps_df['LapTime_prev'] = clean_laps_df.groupby(['Year','Driver','Stint'])['LapTime_seconds'].shift(1)

combined_laps_df_filtered = clean_laps_df.sort_values('Time').reset_index(drop=True)

combined_weather_df = combined_weather_df.sort_values('Time').reset_index(drop=True)
combined_weather_df = combined_weather_df.sort_values('Year').reset_index(drop=True)

weather_df_filtered = combined_weather_df.copy()

combined_laps_df_filtered['Year'] = combined_laps_df_filtered['Year'].astype(int)
weather_df_filtered['Year'] = weather_df_filtered['Year'].astype(int)

combined_laps_df_filtered = combined_laps_df_filtered.sort_values(['Year', 'Time']).reset_index(drop=True)
weather_df_filtered = weather_df_filtered.sort_values(['Year', 'Time']).reset_index(drop=True)

combined_laps_df_filtered = combined_laps_df_filtered.sort_values('Time')
weather_df_filtered = weather_df_filtered.sort_values('Time')

seconds_margin = 60
laps_with_weather = pd.merge_asof(
    combined_laps_df_filtered,
    weather_df_filtered.drop_duplicates(subset=['Time', 'Year']),
    on='Time',
    by='Year',
    direction='backward',
    tolerance=pd.Timedelta(seconds=seconds_margin)
)

laps_with_weather['TempDelta'] = laps_with_weather['TrackTemp'] - laps_with_weather['AirTemp']

print("\nEquipes únicas antes do mapeamento:")
print(laps_with_weather['Team'].unique())

team_mapping = {
    'Alfa Romeo Racing': 'Kick Sauber',
    'Alfa Romeo': 'Kick Sauber',
    'Racing Point': 'Aston Martin',
    'Toro Rosso': 'Racing Bulls',
    'AlphaTauri': 'Racing Bulls',
    'RB': 'Racing Bulls',
    'Renault': 'Alpine'
}

laps_with_weather['Team'] = laps_with_weather['Team'].replace(team_mapping)
print("\nEquipes únicas após mapeamento:")
print(laps_with_weather['Team'].unique())

laps_with_weather['laps_diff'] = laps_with_weather['LapTime_seconds'] - laps_with_weather['LapTime_prev']
diff_data = laps_with_weather['laps_diff'].dropna()

p1 = np.percentile(diff_data, 1)
p99 = np.percentile(diff_data, 99)
Q1 = np.percentile(diff_data, 25)
Q3 = np.percentile(diff_data, 75)
IQR = Q3 - Q1
lower_iqr = Q1 - 1.5 * IQR
upper_iqr = Q3 + 1.5 * IQR
p5 = np.percentile(diff_data, 5)
p95 = np.percentile(diff_data, 95)

print("\n--- CANDIDATOS A CORTE PERFEITO ---")
print(f"1. Conservador (Top 1%):  [{p1:.2f}s, {p99:.2f}s]")
print(f"2. Padrão Estatístico (IQR): [{lower_iqr:.2f}s, {upper_iqr:.2f}s]")
print(f"3. Rigoroso (Top 5%):     [{p5:.2f}s, {p95:.2f}s]")

mask_clean = (
    (laps_with_weather['laps_diff'] >= p5) &
    (laps_with_weather['laps_diff'] <= p95)
)

laps_cleaned = laps_with_weather[mask_clean].copy()

print(f"\nTotal original: {len(laps_with_weather)}")
print(f"Total após filtro: {len(laps_cleaned)}")
print(f"Outliers removidos (Voltas lentas/Pit In): {len(laps_with_weather) - len(laps_cleaned)}")

weather_cols = ['TrackTemp', 'Humidity', 'Pressure', 'WindSpeed', 'WindDirection', 'TempDelta']
gamma_value = 0.1

print(f"\nAplicando Transformação RBF (Gamma={gamma_value}) usando a MEDIANA como referência...")

for col in weather_cols:
    if col in laps_cleaned.columns:
        median_val = laps_cleaned[col].median()
        print(f"Coluna '{col}': Mediana = {median_val:.2f}")
    
        col_name = f"{col}_RBF_Median"
        squared_dist = (laps_cleaned[col] - median_val) ** 2
        laps_cleaned[col_name] = np.exp(-gamma_value * squared_dist)
        laps_cleaned[col_name] = laps_cleaned[col_name].fillna(0)

new_features = [c for c in laps_cleaned.columns if '_RBF_Median' in c]
print(f"Novas Features Criadas: {new_features}")

# ---------------------------------------------------------
# SALVANDO OS DADOS LIMPOS
# ---------------------------------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
model_data_dir = os.path.join(parent_dir, 'ModelData', target_gp_name)

os.makedirs(model_data_dir, exist_ok=True)

output_csv_path = os.path.join(model_data_dir, f"{safe_gp_name}_cleaned_data.csv")

laps_cleaned.to_csv(output_csv_path, index=False)

print(f"\n[SUCESSO] Dados de {target_gp_name} salvos com sucesso em:\n{output_csv_path}")
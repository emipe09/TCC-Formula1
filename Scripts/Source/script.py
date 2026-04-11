import json
import fastf1
import os

CACHE_DIR = './fastf1_cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
print(f"FastF1 cache enabled at: {os.path.abspath(CACHE_DIR)}")

BASE_PLOT_DIR = './f1_plots'
if not os.path.exists(BASE_PLOT_DIR):
    os.makedirs(BASE_PLOT_DIR)
print(f"Base directory for saving plots: {os.path.abspath(BASE_PLOT_DIR)}")

start_year_analysis = 2022
end_year_analysis = 2026
target_gp_name = 'Hungarian Grand Prix'

all_results_data_by_year = {}
all_laps_data_by_year = {}
all_weather_data_by_year = {}
date = {}


print(f"\n--- Starting comprehensive data collection for {target_gp_name} from {start_year_analysis} to {end_year_analysis-1} ---")

for year in range(start_year_analysis, end_year_analysis):
    print(f"\n--- Collecting data for {target_gp_name} in Season: {year} ---")
    schedule = fastf1.get_event_schedule(year)
    target_event = schedule[schedule['EventName'].str.contains(target_gp_name, case=False, na=False)]

    if target_event.empty:
        print(f"  {target_gp_name} not found in {year} schedule, skipping.")
        continue

    target_race_round_row = target_event[target_event['EventFormat'].isin(['conventional', 'sprint', 'sprint_shootout', 'sprint_qualifying'])]

    if target_race_round_row.empty:
        print(f"  No 'conventional' or 'sprint' race event found for {target_gp_name} in {year}, skipping.")
        continue

    round_num = target_race_round_row['RoundNumber'].iloc[0]
    actual_event_name = target_race_round_row['EventName'].iloc[0]

    print(f"  Found {actual_event_name} as Round {round_num} in {year}. Loading session...")

    session = fastf1.get_session(year, round_num, 'R')
    session.load(laps=True, telemetry=True, weather=True, messages=True)

    laps_df = session.laps.copy()
    results_df = session.results.copy()

    if 'LapTime_seconds' not in laps_df.columns:
        laps_df['LapTime_seconds'] = laps_df['LapTime'].dt.total_seconds()

    race_name = session.event['EventName']
    json_path = r'c:\Users\mpoli\Desktop\UFOP\9 PERIODO\TCC\Scripts\compounds.json'
    with open(json_path, 'r') as file:
        compounds_data = json.load(file)

    year_str = str(year)

    if year_str in compounds_data['data']:
        compounds_for_year = compounds_data['data'][year_str]
        gp_compound_mapping = compounds_for_year.get(target_gp_name)

        if gp_compound_mapping:
            laps_df['pirelliCompound'] = laps_df['Compound'].map(gp_compound_mapping)

            laps_df['pirelliCompound'] = laps_df['pirelliCompound'].fillna(laps_df['Compound'])

            print(f" 'pirelliCompound' column created for {target_gp_name} {year}.")

        else:
            print(f" Warning: No compound mapping found in JSON for '{target_gp_name}' in year {year}. 'pirelliCompound' not created.")
    else:
        print(f" Warning: Year {year} not found in 'compounds.json'. 'pirelliCompound' not created.")
        laps_df['pirelliCompound'] = laps_df['Compound']

    all_results_data_by_year[year] = results_df
    all_laps_data_by_year[year] = laps_df
    all_weather_data_by_year[year] = session.weather_data
    date[year]= session.date

    print(f"Session data loaded for: {race_name} {year} Round {round_num}")
    print(f"Total laps: {len(laps_df)}, Total results: {len(results_df)}")

import os

# Definir o diretório de saída
output_dir = 'f1_data_export'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"\n--- Iniciando exportação dos dados para CSV ---")

for year in all_laps_data_by_year.keys():
    print(f"Processando e salvando dados de {year}...")
    
    # 1. Recuperar DataFrames
    laps_df = all_laps_data_by_year[year]
    results_df = all_results_data_by_year[year]
    weather_df = all_weather_data_by_year[year]
    
    # --- SANITY CHECK DE ÚLTIMA HORA (Tratamento de None/Inconsistências) ---
    # Propagar composto anterior por piloto para evitar os 'None' intermitentes
    if 'Compound' in laps_df.columns:
        laps_df['Compound'] = laps_df.groupby('Driver')['Compound'].ffill()
    
    # Se você criou a pirelliCompound, garanta que ela também não tenha nulos
    if 'pirelliCompound' in laps_df.columns:
        laps_df['pirelliCompound'] = laps_df.groupby('Driver')['pirelliCompound'].ffill()
    
    # 2. Gerar nomes de arquivos amigáveis
    safe_gp_name = target_gp_name.lower().replace(' ', '_')
    
    laps_file = os.path.join(output_dir, f"{safe_gp_name}_laps_{year}.csv")
    results_file = os.path.join(output_dir, f"{safe_gp_name}_results_{year}.csv")
    weather_file = os.path.join(output_dir, f"{safe_gp_name}_weather_{year}.csv")
    
    # 3. Salvar em CSV
    # index=False evita que o pandas crie uma coluna extra de números sem nome
    laps_df.to_csv(laps_file, index=False)
    results_df.to_csv(results_file, index=False)
    weather_df.to_csv(weather_file, index=False)
    
    print(f"  [OK] Arquivos salvos para o ano {year}.")

print(f"\nExportação concluída! Os arquivos estão na pasta: {output_dir}")
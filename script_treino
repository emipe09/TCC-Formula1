import json
import fastf1
import os
import pandas as pd

CACHE_DIR = "./fastf1_cache"
OUTPUT_DIR = "f1_data_export\Bahrain\Free Practice"

start_year_analysis = 2022
end_year_analysis = 2026  # exclusivo (vai até 2025)
target_gp_name = "Bahrain Grand Prix"

# Sessões de treino livre
PRACTICE_SESSIONS = ["FP1", "FP2", "FP3"]

# Caminho do mapeamento de compostos
json_path = r"c:\Users\mpoli\Desktop\UFOP\9 PERIODO\TCC\Scripts\compounds.json"

# =========================
# PREPARAÇÃO
# =========================
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

fastf1.Cache.enable_cache(CACHE_DIR)
print(f"FastF1 cache enabled at: {os.path.abspath(CACHE_DIR)}")
print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")

with open(json_path, "r", encoding="utf-8") as f:
    compounds_data = json.load(f)

all_practice_laps_by_year = {}
all_practice_weather_by_year = {}

print(
    f"\n--- Coletando TREINO LIVRE para {target_gp_name} "
    f"de {start_year_analysis} até {end_year_analysis - 1} ---"
)

# =========================
# COLETA
# =========================
for year in range(start_year_analysis, end_year_analysis):
    print(f"\n=== Ano {year} ===")

    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as e:
        print(f"  [ERRO] Falha ao carregar calendário de {year}: {e}")
        continue

    target_event = schedule[
        schedule["EventName"].str.contains(target_gp_name, case=False, na=False)
    ]

    if target_event.empty:
        print(f"  [SKIP] GP não encontrado no calendário de {year}.")
        continue

    # Pega round do evento
    round_num = int(target_event["RoundNumber"].iloc[0])
    actual_event_name = target_event["EventName"].iloc[0]
    print(f"  Evento encontrado: {actual_event_name} (Round {round_num})")

    year_laps = []
    year_weather = []

    for sess_code in PRACTICE_SESSIONS:
        print(f"  -> Tentando sessão {sess_code}...")

        try:
            session = fastf1.get_session(year, round_num, sess_code)
            session.load(laps=True, telemetry=False, weather=True, messages=False)
        except Exception as e:
            print(f"     [SKIP] {sess_code} indisponível/falhou: {e}")
            continue

        laps_df = session.laps.copy()
        weather_df = session.weather_data.copy()

        if laps_df.empty:
            print(f"     [SKIP] {sess_code} sem voltas.")
            continue

        # Colunas auxiliares
        if "LapTime_seconds" not in laps_df.columns:
            laps_df["LapTime_seconds"] = laps_df["LapTime"].dt.total_seconds()

        laps_df["SessionType"] = sess_code
        laps_df["Year"] = year
        laps_df["Race"] = actual_event_name

        weather_df["SessionType"] = sess_code
        weather_df["Year"] = year
        weather_df["Race"] = actual_event_name

        # Mapeamento de compostos (igual ao seu pipeline de corrida)
        year_str = str(year)
        gp_mapping = (
            compounds_data.get("data", {})
            .get(year_str, {})
            .get(target_gp_name, None)
        )

        if gp_mapping and "Compound" in laps_df.columns:
            laps_df["pirelliCompound"] = laps_df["Compound"].map(gp_mapping)
            laps_df["pirelliCompound"] = laps_df["pirelliCompound"].fillna(laps_df["Compound"])
        elif "Compound" in laps_df.columns:
            laps_df["pirelliCompound"] = laps_df["Compound"]

        # Sanity check de compostos por piloto
        if "Compound" in laps_df.columns:
            laps_df["Compound"] = laps_df.groupby("Driver")["Compound"].ffill()
        if "pirelliCompound" in laps_df.columns:
            laps_df["pirelliCompound"] = laps_df.groupby("Driver")["pirelliCompound"].ffill()

        year_laps.append(laps_df)
        year_weather.append(weather_df)

        print(f"     [OK] {sess_code}: {len(laps_df)} voltas")

    if not year_laps:
        print(f"  [SKIP] Nenhuma sessão de treino livre coletada em {year}.")
        continue

    all_practice_laps_by_year[year] = pd.concat(year_laps, ignore_index=True)
    all_practice_weather_by_year[year] = pd.concat(year_weather, ignore_index=True)

# =========================
# EXPORTAÇÃO CSV
# =========================
safe_gp_name = target_gp_name.lower().replace(" ", "_")

print("\n--- Exportando CSVs de treino livre ---")
for year in sorted(all_practice_laps_by_year.keys()):
    laps_df = all_practice_laps_by_year[year]
    weather_df = all_practice_weather_by_year[year]

    laps_file = os.path.join(OUTPUT_DIR, f"{safe_gp_name}_practice_laps_{year}.csv")
    weather_file = os.path.join(OUTPUT_DIR, f"{safe_gp_name}_practice_weather_{year}.csv")

    laps_df.to_csv(laps_file, index=False)
    weather_df.to_csv(weather_file, index=False)

    print(f"  [OK] {year}:")
    print(f"       - {laps_file}")
    print(f"       - {weather_file}")

print("\nConcluído.")
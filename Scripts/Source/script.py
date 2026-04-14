import json
import os
import fastf1


CACHE_DIR = "./fastf1_cache"
START_YEAR = 2022
END_YEAR_EXCLUSIVE = 2026
TARGET_GP_NAME = os.environ.get("TARGET_GP_NAME", "Hungarian Grand Prix")

GP_TO_DATA_DIR = {
    "Bahrain Grand Prix": "Bahrain",
    "Hungarian Grand Prix": "Hungary",
    "Italian Grand Prix": "Italy",
    "Saudi Arabian Grand Prix": "Saudi Arabia",
    "United States Grand Prix": "United States",
}


def resolve_paths(target_gp_name: str) -> tuple[str, str]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(scripts_dir)

    track_dir = GP_TO_DATA_DIR.get(target_gp_name)
    if not track_dir:
        valid = ", ".join(sorted(GP_TO_DATA_DIR.keys()))
        raise ValueError(f"Unsupported TARGET_GP_NAME: '{target_gp_name}'. Valid options: {valid}")

    output_root = os.path.join(project_root, "Data", track_dir)
    compounds_path = os.path.join(project_root, "Utils", "compounds.json")
    return output_root, compounds_path


def main() -> None:
    output_root, compounds_path = resolve_paths(TARGET_GP_NAME)

    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)
    print(f"FastF1 cache enabled at: {os.path.abspath(CACHE_DIR)}")
    print(f"Data output root: {os.path.abspath(output_root)}")

    with open(compounds_path, "r", encoding="utf-8") as file:
        compounds_data = json.load(file)

    all_results_data_by_year = {}
    all_laps_data_by_year = {}
    all_weather_data_by_year = {}

    print(
        f"\n--- Starting race data collection for {TARGET_GP_NAME} "
        f"from {START_YEAR} to {END_YEAR_EXCLUSIVE - 1} ---"
    )

    for year in range(START_YEAR, END_YEAR_EXCLUSIVE):
        print(f"\n--- Collecting race data for {TARGET_GP_NAME} in {year} ---")

        schedule = fastf1.get_event_schedule(year)
        target_event = schedule[schedule["EventName"].str.contains(TARGET_GP_NAME, case=False, na=False)]

        if target_event.empty:
            print(f"  {TARGET_GP_NAME} not found in {year} schedule, skipping.")
            continue

        race_round_row = target_event[
            target_event["EventFormat"].isin(["conventional", "sprint", "sprint_shootout", "sprint_qualifying"])
        ]

        if race_round_row.empty:
            print(f"  No supported race event found for {TARGET_GP_NAME} in {year}, skipping.")
            continue

        round_num = race_round_row["RoundNumber"].iloc[0]
        actual_event_name = race_round_row["EventName"].iloc[0]
        print(f"  Found {actual_event_name} as Round {round_num} in {year}. Loading session...")

        session = fastf1.get_session(year, round_num, "R")
        session.load(laps=True, telemetry=True, weather=True, messages=True)

        laps_df = session.laps.copy()
        results_df = session.results.copy()

        if "LapTime_seconds" not in laps_df.columns:
            laps_df["LapTime_seconds"] = laps_df["LapTime"].dt.total_seconds()

        year_str = str(year)
        gp_mapping = compounds_data.get("data", {}).get(year_str, {}).get(TARGET_GP_NAME)

        if gp_mapping and "Compound" in laps_df.columns:
            laps_df["pirelliCompound"] = laps_df["Compound"].map(gp_mapping)
            laps_df["pirelliCompound"] = laps_df["pirelliCompound"].fillna(laps_df["Compound"])
            print(f"  'pirelliCompound' column created for {TARGET_GP_NAME} {year}.")
        else:
            print(f"  Warning: no compound mapping found for '{TARGET_GP_NAME}' in {year}. Falling back to source compounds.")
            if "Compound" in laps_df.columns:
                laps_df["pirelliCompound"] = laps_df["Compound"]

        all_results_data_by_year[year] = results_df
        all_laps_data_by_year[year] = laps_df
        all_weather_data_by_year[year] = session.weather_data

        print(f"Session data loaded for {actual_event_name} {year} Round {round_num}")
        print(f"Total laps: {len(laps_df)}, Total results: {len(results_df)}")

    safe_gp_name = TARGET_GP_NAME.lower().replace(" ", "_")

    race_laps_dir = os.path.join(output_root, "Race", "Laps")
    race_results_dir = os.path.join(output_root, "Results")
    race_weather_dir = os.path.join(output_root, "Race", "Weather")

    os.makedirs(race_laps_dir, exist_ok=True)
    os.makedirs(race_results_dir, exist_ok=True)
    os.makedirs(race_weather_dir, exist_ok=True)

    print("\n--- Starting CSV export ---")

    for year in all_laps_data_by_year.keys():
        print(f"Processing and saving data for {year}...")

        laps_df = all_laps_data_by_year[year]
        results_df = all_results_data_by_year[year]
        weather_df = all_weather_data_by_year[year]

        if "Compound" in laps_df.columns:
            laps_df["Compound"] = laps_df.groupby("Driver")["Compound"].ffill()
        if "pirelliCompound" in laps_df.columns:
            laps_df["pirelliCompound"] = laps_df.groupby("Driver")["pirelliCompound"].ffill()

        laps_file = os.path.join(race_laps_dir, f"{safe_gp_name}_laps_{year}.csv")
        results_file = os.path.join(race_results_dir, f"{safe_gp_name}_results_{year}.csv")
        weather_file = os.path.join(race_weather_dir, f"{safe_gp_name}_weather_{year}.csv")

        laps_df.to_csv(laps_file, index=False)
        results_df.to_csv(results_file, index=False)
        weather_df.to_csv(weather_file, index=False)

        print(f"  [OK] Files saved for year {year}.")

    print(f"\nExport completed. Files are in: {output_root}")


if __name__ == "__main__":
    main()



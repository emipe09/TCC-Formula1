import json
import os
import fastf1
import pandas as pd


CACHE_DIR = "./fastf1_cache"
START_YEAR = 2022
END_YEAR_EXCLUSIVE = 2026
TARGET_GP_NAME = os.environ.get("TARGET_GP_NAME", "Bahrain Grand Prix")
PRACTICE_SESSIONS = ["FP1", "FP2", "FP3"]

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

    output_dir = os.path.join(project_root, "Data", track_dir, "Free Practice")
    compounds_path = os.path.join(project_root, "Utils", "compounds.json")
    return output_dir, compounds_path


def main() -> None:
    output_dir, compounds_path = resolve_paths(TARGET_GP_NAME)

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    fastf1.Cache.enable_cache(CACHE_DIR)
    print(f"FastF1 cache enabled at: {os.path.abspath(CACHE_DIR)}")
    print(f"Output directory: {os.path.abspath(output_dir)}")

    with open(compounds_path, "r", encoding="utf-8") as file:
        compounds_data = json.load(file)

    all_practice_laps_by_year = {}
    all_practice_weather_by_year = {}

    print(
        f"\n--- Collecting FREE PRACTICE for {TARGET_GP_NAME} "
        f"from {START_YEAR} to {END_YEAR_EXCLUSIVE - 1} ---"
    )

    for year in range(START_YEAR, END_YEAR_EXCLUSIVE):
        print(f"\n=== Year {year} ===")

        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as exc:
            print(f"  [ERROR] Failed to load schedule for {year}: {exc}")
            continue

        target_event = schedule[
            schedule["EventName"].str.contains(TARGET_GP_NAME, case=False, na=False)
        ]

        if target_event.empty:
            print(f"  [SKIP] Grand Prix not found in {year} schedule.")
            continue

        round_num = int(target_event["RoundNumber"].iloc[0])
        actual_event_name = target_event["EventName"].iloc[0]
        print(f"  Event found: {actual_event_name} (Round {round_num})")

        year_laps = []
        year_weather = []

        for session_code in PRACTICE_SESSIONS:
            print(f"  -> Trying session {session_code}...")

            try:
                session = fastf1.get_session(year, round_num, session_code)
                session.load(laps=True, telemetry=False, weather=True, messages=False)
            except Exception as exc:
                print(f"     [SKIP] {session_code} unavailable/failed: {exc}")
                continue

            laps_df = session.laps.copy()
            weather_df = session.weather_data.copy()

            if laps_df.empty:
                print(f"     [SKIP] {session_code} without laps.")
                continue

            if "LapTime_seconds" not in laps_df.columns:
                laps_df["LapTime_seconds"] = laps_df["LapTime"].dt.total_seconds()

            laps_df["SessionType"] = session_code
            laps_df["Year"] = year
            laps_df["Race"] = actual_event_name

            weather_df["SessionType"] = session_code
            weather_df["Year"] = year
            weather_df["Race"] = actual_event_name

            year_str = str(year)
            gp_mapping = (
                compounds_data.get("data", {})
                .get(year_str, {})
                .get(TARGET_GP_NAME, None)
            )

            if gp_mapping and "Compound" in laps_df.columns:
                laps_df["pirelliCompound"] = laps_df["Compound"].map(gp_mapping)
                laps_df["pirelliCompound"] = laps_df["pirelliCompound"].fillna(laps_df["Compound"])
            elif "Compound" in laps_df.columns:
                laps_df["pirelliCompound"] = laps_df["Compound"]

            if "Compound" in laps_df.columns:
                laps_df["Compound"] = laps_df.groupby("Driver")["Compound"].ffill()
            if "pirelliCompound" in laps_df.columns:
                laps_df["pirelliCompound"] = laps_df.groupby("Driver")["pirelliCompound"].ffill()

            year_laps.append(laps_df)
            year_weather.append(weather_df)
            print(f"     [OK] {session_code}: {len(laps_df)} laps")

        if not year_laps:
            print(f"  [SKIP] No free practice session collected in {year}.")
            continue

        all_practice_laps_by_year[year] = pd.concat(year_laps, ignore_index=True)
        all_practice_weather_by_year[year] = pd.concat(year_weather, ignore_index=True)

    safe_gp_name = TARGET_GP_NAME.lower().replace(" ", "_")

    print("\n--- Exporting free practice CSVs ---")
    for year in sorted(all_practice_laps_by_year.keys()):
        laps_df = all_practice_laps_by_year[year]
        weather_df = all_practice_weather_by_year[year]

        laps_file = os.path.join(output_dir, "Laps", f"{safe_gp_name}_practice_laps_{year}.csv")
        weather_file = os.path.join(output_dir, "Weather", f"{safe_gp_name}_practice_weather_{year}.csv")

        os.makedirs(os.path.dirname(laps_file), exist_ok=True)
        os.makedirs(os.path.dirname(weather_file), exist_ok=True)

        laps_df.to_csv(laps_file, index=False)
        weather_df.to_csv(weather_file, index=False)

        print(f"  [OK] {year}:")
        print(f"       - {laps_file}")
        print(f"       - {weather_file}")

    print("\nCompleted.")


if __name__ == "__main__":
    main()



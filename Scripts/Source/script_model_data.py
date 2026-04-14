import os
import numpy as np
import pandas as pd


START_YEAR = 2022
END_YEAR_EXCLUSIVE = 2026
TARGET_GP_NAME = os.environ.get("TARGET_GP_NAME", "Saudi Arabian Grand Prix")

GP_TO_DATA_DIR = {
    "Bahrain Grand Prix": "Bahrain",
    "Hungarian Grand Prix": "Hungary",
    "Italian Grand Prix": "Italy",
    "Saudi Arabian Grand Prix": "Saudi Arabia",
    "United States Grand Prix": "United States",
}


def resolve_directories(target_gp_name: str) -> tuple[str, str, str, str]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(scripts_dir)

    track_dir = GP_TO_DATA_DIR.get(target_gp_name)
    if not track_dir:
        valid = ", ".join(sorted(GP_TO_DATA_DIR.keys()))
        raise ValueError(f"Unsupported TARGET_GP_NAME: '{target_gp_name}'. Valid options: {valid}")

    laps_dir = os.path.join(project_root, "Data", track_dir, "Race", "Laps")
    weather_dir = os.path.join(project_root, "Data", track_dir, "Race", "Weather")
    model_data_dir = os.path.join(scripts_dir, "ModelData", target_gp_name)
    return laps_dir, weather_dir, model_data_dir, project_root


def main() -> None:
    safe_gp_name = TARGET_GP_NAME.lower().replace(" ", "_")
    laps_dir, weather_dir, model_data_dir, _ = resolve_directories(TARGET_GP_NAME)

    all_laps_data_by_year: dict[int, pd.DataFrame] = {}
    all_weather_data_by_year: dict[int, pd.DataFrame] = {}

    print(f"--- Loading local data for {TARGET_GP_NAME} ---")
    print(f"Laps directory: {laps_dir}")
    print(f"Weather directory: {weather_dir}")

    for year in range(START_YEAR, END_YEAR_EXCLUSIVE):
        laps_file = os.path.join(laps_dir, f"{safe_gp_name}_laps_{year}.csv")
        weather_file = os.path.join(weather_dir, f"{safe_gp_name}_weather_{year}.csv")

        if os.path.exists(laps_file) and os.path.exists(weather_file):
            print(f"  [CSV] Loading and converting season {year}...")
            laps_df = pd.read_csv(laps_file)
            weather_df = pd.read_csv(weather_file)

            laps_df["Time"] = pd.to_timedelta(laps_df["Time"])
            weather_df["Time"] = pd.to_timedelta(weather_df["Time"])

            if "LapTime" in laps_df.columns:
                laps_df["LapTime"] = pd.to_timedelta(laps_df["LapTime"])

            all_laps_data_by_year[year] = laps_df.sort_values("Time")
            all_weather_data_by_year[year] = weather_df.sort_values("Time")
        else:
            print(f"  [WARN] Missing files for year {year} in '{laps_dir}'.")

    print("\nStarting multi-year merge...")
    yearly_laps = []
    yearly_weather = []

    for year in range(START_YEAR, END_YEAR_EXCLUSIVE):
        if year in all_laps_data_by_year:
            print(f"Loading data for {TARGET_GP_NAME} {year}...")
            laps_df = all_laps_data_by_year[year].copy()
            weather_df = all_weather_data_by_year[year].copy()

            laps_df["Year"] = year
            weather_df["Year"] = year
            yearly_laps.append(laps_df)
            yearly_weather.append(weather_df)
        else:
            print(f"No data for {TARGET_GP_NAME} {year}.")

    if not yearly_laps:
        print("No lap data found to process. Exiting.")
        return

    combined_laps_df = pd.concat(yearly_laps, ignore_index=True)
    print(f"\nMerged {len(yearly_laps)} years successfully.")
    print(f"Total laps loaded: {len(combined_laps_df)}")

    clean_laps_df = combined_laps_df[combined_laps_df["IsAccurate"] == True].copy()

    required_cols = ["LapTime_seconds", "TyreLife"]
    if "pirelliCompound" in clean_laps_df.columns:
        required_cols.append("pirelliCompound")

    clean_laps_df.dropna(subset=required_cols, inplace=True)
    clean_laps_df["Year"] = clean_laps_df["Year"].astype("category")
    print(f"Analyzing {len(clean_laps_df)} clean laps from all years.")

    if not yearly_weather:
        print("No weather data found for the selected year range. Exiting.")
        return

    combined_weather_df = pd.concat(yearly_weather, ignore_index=True)
    print(f"Total weather records loaded: {len(combined_weather_df)}")

    clean_laps_df = clean_laps_df.sort_values(["Year", "Driver", "Stint", "LapNumber"])
    clean_laps_df["LapTime_prev"] = clean_laps_df.groupby(["Year", "Driver", "Stint"])["LapTime_seconds"].shift(1)

    laps_filtered = clean_laps_df.sort_values(["Year", "Time"]).reset_index(drop=True)
    weather_filtered = combined_weather_df.sort_values(["Year", "Time"]).reset_index(drop=True)

    laps_filtered["Year"] = laps_filtered["Year"].astype(int)
    weather_filtered["Year"] = weather_filtered["Year"].astype(int)

    seconds_margin = 60
    laps_with_weather = pd.merge_asof(
        laps_filtered,
        weather_filtered.drop_duplicates(subset=["Time", "Year"]),
        on="Time",
        by="Year",
        direction="backward",
        tolerance=pd.Timedelta(seconds=seconds_margin),
    )

    laps_with_weather["TempDelta"] = laps_with_weather["TrackTemp"] - laps_with_weather["AirTemp"]

    print("\nUnique teams before mapping:")
    print(laps_with_weather["Team"].unique())

    team_mapping = {
        "Alfa Romeo Racing": "Kick Sauber",
        "Alfa Romeo": "Kick Sauber",
        "Racing Point": "Aston Martin",
        "Toro Rosso": "Racing Bulls",
        "AlphaTauri": "Racing Bulls",
        "RB": "Racing Bulls",
        "Renault": "Alpine",
    }

    laps_with_weather["Team"] = laps_with_weather["Team"].replace(team_mapping)
    print("\nUnique teams after mapping:")
    print(laps_with_weather["Team"].unique())

    laps_with_weather["laps_diff"] = laps_with_weather["LapTime_seconds"] - laps_with_weather["LapTime_prev"]
    diff_data = laps_with_weather["laps_diff"].dropna()

    p1 = np.percentile(diff_data, 1)
    p99 = np.percentile(diff_data, 99)
    q1 = np.percentile(diff_data, 25)
    q3 = np.percentile(diff_data, 75)
    iqr = q3 - q1
    lower_iqr = q1 - 1.5 * iqr
    upper_iqr = q3 + 1.5 * iqr
    p5 = np.percentile(diff_data, 5)
    p95 = np.percentile(diff_data, 95)

    print("\n--- CANDIDATE THRESHOLDS ---")
    print(f"1. Conservative (Top 1%): [{p1:.2f}s, {p99:.2f}s]")
    print(f"2. Statistical Standard (IQR): [{lower_iqr:.2f}s, {upper_iqr:.2f}s]")
    print(f"3. Strict (Top 5%): [{p5:.2f}s, {p95:.2f}s]")

    mask_clean = (laps_with_weather["laps_diff"] >= p5) & (laps_with_weather["laps_diff"] <= p95)
    laps_cleaned = laps_with_weather[mask_clean].copy()

    print(f"\nOriginal total: {len(laps_with_weather)}")
    print(f"Total after filtering: {len(laps_cleaned)}")
    print(f"Outliers removed (slow laps/Pit In): {len(laps_with_weather) - len(laps_cleaned)}")

    weather_cols = ["TrackTemp", "Humidity", "Pressure", "WindSpeed", "WindDirection", "TempDelta"]
    gamma_value = 0.1

    print(f"\nApplying RBF transformation (Gamma={gamma_value}) using MEDIAN as reference...")
    for col in weather_cols:
        if col in laps_cleaned.columns:
            median_val = laps_cleaned[col].median()
            print(f"Column '{col}': Median = {median_val:.2f}")
            col_name = f"{col}_RBF_Median"
            squared_dist = (laps_cleaned[col] - median_val) ** 2
            laps_cleaned[col_name] = np.exp(-gamma_value * squared_dist).fillna(0)

    new_features = [col for col in laps_cleaned.columns if "_RBF_Median" in col]
    print(f"New features created: {new_features}")

    os.makedirs(model_data_dir, exist_ok=True)
    output_csv_path = os.path.join(model_data_dir, f"{safe_gp_name}_cleaned_data.csv")
    laps_cleaned.to_csv(output_csv_path, index=False)

    print(f"\n[SUCCESS] Data for {TARGET_GP_NAME} saved successfully at:\n{output_csv_path}")


if __name__ == "__main__":
    main()



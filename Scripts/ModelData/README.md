# Cleaned Modeling Data

This folder contains the cleaned, feature-engineered datasets consumed by the notebooks and by the two reproducible scripts in `Scripts/Source/`.

Each Grand Prix folder follows this naming pattern:

```text
<safe_gp_name>_cleaned_data.csv
```

The current study uses:

| Folder | Dataset | Circuit |
|---|---|---|
| `Bahrain Grand Prix/` | `bahrain_grand_prix_cleaned_data.csv` | Bahrain International Circuit |
| `Saudi Arabian Grand Prix/` | `saudi_arabian_grand_prix_cleaned_data.csv` | Jeddah Corniche Circuit |
| `United States Grand Prix/` | `united_states_grand_prix_cleaned_data.csv` | Circuit of the Americas |
| `Italian Grand Prix/` | `italian_grand_prix_cleaned_data.csv` | Autodromo Nazionale Monza |
| `Hungarian Grand Prix/` | `hungarian_grand_prix_cleaned_data.csv` | Hungaroring |

These files include the target `LapTime_seconds`, the autoregressive `LapTime_prev` feature, RBF-transformed weather variables, tyre life, lap number, and categorical identifiers.

# Formula 1 Race-Pace Prediction

This repository contains the current research code and notebooks for multi-circuit Formula 1 lap-time prediction. The project uses public FastF1-derived race data to model `LapTime_seconds` with a temporal protocol that mirrors a real race: sliding-window validation inside the modeling segment and a final sequential holdout on the last laps.

## Scope

The current version focuses on five Grand Prix events from the 2022-2025 technical-regulation period:

| Grand Prix | Circuit | Location |
|---|---|---|
| Bahrain Grand Prix | Bahrain International Circuit | Sakhir, Bahrain |
| Saudi Arabian Grand Prix | Jeddah Corniche Circuit | Jeddah, Saudi Arabia |
| United States Grand Prix | Circuit of the Americas | Austin, United States |
| Italian Grand Prix | Autodromo Nazionale Monza | Monza, Italy |
| Hungarian Grand Prix | Hungaroring | Mogyorod, Hungary |

## Repository Layout

```text
TCC/
|- Data/
|  |- Bahrain/
|  |- Hungary/
|  |- Italy/
|  |- Saudi Arabia/
|  |- United States/
|- Scripts/
|  |- ModelData/
|  |- Notebooks/
|  |- Source/
|     |- model_lr_sw.py
|     |- model_xgb_sw.py
|- Utils/
|  |- compounds.json
|  |- requirements.txt
|- README.md
```

Generated outputs, FastF1 caches, local PDFs, notebook plot folders, XGBoost parameter dumps, and historical run logs are intentionally ignored by Git.

## Data

`Data/` stores raw race-session CSV files by circuit:

- race laps
- race weather
- race results

The modeling scripts run from cleaned datasets in `Scripts/ModelData/`. Those files contain the article-facing engineered data used by the notebooks and by the two scripts in `Scripts/Source/`.

## Notebooks

The notebooks in `Scripts/Notebooks/` are the full circuit-specific analyses:

| Notebook | Circuit |
|---|---|
| `Notebook_Bahrain.ipynb` | Bahrain Grand Prix |
| `Notebook_Saudi.ipynb` | Saudi Arabian Grand Prix |
| `Notebook_USA.ipynb` | United States Grand Prix |
| `Notebook_Italia.ipynb` | Italian Grand Prix |
| `Notebook_Hungary.ipynb` | Hungarian Grand Prix |

Each notebook is written in English and follows the same structure: data preparation, exploratory analysis, feature engineering, Linear Regression, XGBoost, sliding-window validation, sequential holdout, and COS metrics.

## Modeling Scripts

Only the current sliding-window scripts are kept in `Scripts/Source/`:

- `model_lr_sw.py`: Linear Regression with median imputation, standard scaling, sliding-window validation, and sequential holdout.
- `model_xgb_sw.py`: XGBoost with Optuna hyperparameter tuning, sliding-window validation, and sequential holdout.

Both scripts report:

- sliding-window RMSE, MAE, R2, and residual standard deviation
- sequential-holdout RMSE, MAE, and R2 with bootstrap confidence intervals
- `COS_MAE` and `COS_RMSE` with indicative 95% confidence intervals

The COS metrics are computed as:

```text
COS_MAE  = 0.5 * (MAE_SW / MAE_final)  + 0.5 * (STD_SW / STD_final)
COS_RMSE = 0.5 * (RMSE_SW / RMSE_final) + 0.5 * (STD_SW / STD_final)
```

The COS confidence intervals are descriptive because the sliding windows overlap.

## Installation

```bash
pip install -r Utils/requirements.txt
```

## Running a Model

PowerShell:

```powershell
$env:TARGET_GP_NAME = "Bahrain Grand Prix"
python Scripts/Source/model_lr_sw.py
python Scripts/Source/model_xgb_sw.py
```

Bash:

```bash
TARGET_GP_NAME="Bahrain Grand Prix" python Scripts/Source/model_lr_sw.py
TARGET_GP_NAME="Bahrain Grand Prix" python Scripts/Source/model_xgb_sw.py
```

Supported `TARGET_GP_NAME` values are:

- `Bahrain Grand Prix`
- `Saudi Arabian Grand Prix`
- `United States Grand Prix`
- `Italian Grand Prix`
- `Hungarian Grand Prix`

## Key Features

Numerical predictors:

- `TyreLife`
- `LapNumber`
- `Humidity_RBF_Median`
- `Pressure_RBF_Median`
- `TrackTemp_RBF_Median`
- `WindSpeed_RBF_Median`
- `TempDelta_RBF_Median`
- `LapTime_prev`

Categorical predictors:

- `Driver`
- `Team`
- `pirelliCompound`
- `Year`

Target:

- `LapTime_seconds`

## Reproducibility Notes

- The final 20% of race laps is reserved as a sequential holdout.
- Sliding-window validation is performed only inside the first 80% modeling block.
- XGBoost parameter files are generated under `Scripts/Results/` when needed and are ignored by Git.
- The notebooks remain the narrative, circuit-specific record of the analysis; the scripts are the lean reproducible runners for GitHub.

## Authors

- Marcos Paulo de Oliveira Pereira
- Carlos Henrique Gomes Ferreira
- Alexandre Magno de Sousa

Universidade Federal de Ouro Preto (UFOP)

# TCC Formula 1 - Predictive Lap Time Analysis

## Title
Multi-Circuit Analysis of Formula 1 Lap Times: comparison of temporal modeling approaches for forecasting and strategy support.

## Objective
This project develops and compares Machine Learning models to predict lap time in Formula 1 races using historical data from 2022 to 2025.

Current repository focus:
- comparison between Linear Regression and XGBoost
- comparison between temporal validation strategies
- evaluation on a final sequential holdout set

## Main Implemented Updates
1. Automated multi-track execution with a single script.
2. Track parameterization through the `TARGET_GP_NAME` environment variable across main models.
3. Standardized temporal split by `LapNumber`:
   - initial 80% for modeling
   - final 20% for sequential holdout
4. Organized results by model family and approach.
5. Removed coupling with legacy `batch_runs` in the current orchestrator.
6. Added holdout 95% confidence intervals for `cv`, `ew`, and `sw` (bootstrap) for:
   - RMSE
   - MAE
   - R2

## Project Structure

```text
TCC/
|- Bibliography/
|- Data/
|  |- Bahrain/
|  |- Hungary/
|  |- Italy/
|  |- Saudi Arabia/
|  |- United States/
|- Scripts/
|  |- Notebooks/
|  |- Source/
|  |  |- script_model_data.py
|  |  |- run_all_models_tracks.py
|  |  |- model_lr_cv.py
|  |  |- model_lr_ew.py
|  |  |- model_lr_sw.py
|  |  |- model_lr_wf.py
|  |  |- model_xgb_cv.py
|  |  |- model_xgb_ew.py
|  |  |- model_xgb_sw.py
|  |  |- model_xgb_wf.py
|  |- ModelData/
|  |- Results/
|  |- fastf1_cache/
|- Utils/
|  |- compounds.json
|  |- requirements.txt
|- reproducibility_paths.json
|- README.md
```

Notes:
- Legacy run folders may still exist inside some result subdirectories.
- The active execution/output structure is the one described above.

## Execution Flow

### 1) Generate cleaned data by track

```bash
python Scripts/Source/script_model_data.py
```

Expected input:
- `Data/<Track>/...`

Expected output:
- `Scripts/ModelData/<Grand Prix>/<safe_gp_name>_cleaned_data.csv`

### 2) Run models individually

Examples:

```bash
python Scripts/Source/model_lr_cv.py
python Scripts/Source/model_lr_ew.py
python Scripts/Source/model_lr_sw.py
python Scripts/Source/model_xgb_cv.py
python Scripts/Source/model_xgb_ew.py
python Scripts/Source/model_xgb_sw.py
```

### 3) Run multi-track and multi-model batch

```bash
python Scripts/Source/run_all_models_tracks.py
```

The orchestrator allows selecting tracks and models and generates logs per run.

## Track Parameterization
Main scripts use:
- `TARGET_GP_NAME` (environment variable)

If the variable is not set, each script uses its internal default value.

PowerShell example:

```powershell
$env:TARGET_GP_NAME = "United States Grand Prix"
python Scripts/Source/model_xgb_sw.py
```

## Validation Approaches

### CV
- internal K-Fold validation on the modeling set (80%)

### EW
- Expanding Window on the modeling set (80%)

### SW
- Sliding Window on the modeling set (80%)

### WF
- Walk-Forward (WF-specific scripts)

## Metrics and Holdout CI
For `cv`, `ew`, and `sw`, final evaluation includes:
- Holdout RMSE
- Holdout MAE
- Holdout R2
- Holdout 95% confidence interval for RMSE, MAE, and R2 via bootstrap

Internal evaluation remains reported by approach (CV, EW, SW).

## Artifact Locations

### XGBoost Parameters
- `Scripts/Results/xgboost/cv/params/*_xgb_params_cv.json`
- `Scripts/Results/xgboost/ew/params/*_xgb_params_ew.json`
- `Scripts/Results/xgboost/sw/params/*_xgb_params_sw.json`

### Execution Logs
Current structure by family/approach:
- `Scripts/Results/<family>/<approach>/runs/<timestamp>/logs/*.log`

### Orchestrator Summaries
- `Scripts/Results/runs/<timestamp>/summary.json`
- `Scripts/Results/runs/<timestamp>/summary.csv`

## Dependencies
Dependencies are listed in:
- `Utils/requirements.txt`

Installation:

```bash
pip install -r Utils/requirements.txt
```

## Reproducibility
The repository includes a machine-readable directory map for reproducibility:
- `reproducibility_paths.json`

This file centralizes input/output directories and canonical script paths used to run the pipeline.

## Current Status
- Functional pipeline for 5 GPs: Bahrain, Hungary, Italy, Saudi Arabia, United States.
- Active comparison between model families and temporal approaches.
- Reorganized and standardized result structure.
- Holdout 95% confidence intervals implemented for `cv`, `ew`, and `sw`.

## Authors
- Marcos Paulo de Oliveira Pereira
- Alexandre Magno de Sousa

UFOP - Computer Engineering

## License
Academic use.

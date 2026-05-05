# Modeling Scripts

Only the current sliding-window modeling scripts are kept here.

## `model_lr_sw.py`

Runs Linear Regression for the selected Grand Prix using:

- median imputation for numerical predictors
- standard scaling
- one-hot encoding with `drop_first=True`
- sliding-window validation over the modeling block
- final sequential holdout evaluation

## `model_xgb_sw.py`

Runs XGBoost for the selected Grand Prix using:

- one-hot encoding with all categories retained
- median imputation for numerical predictors
- Optuna tuning when no saved parameters are available
- calibrated `n_estimators` from sliding-window early stopping
- final sequential holdout evaluation

## COS Metrics

Both scripts report:

```text
COS_MAE  = 0.5 * (MAE_SW / MAE_final)  + 0.5 * (STD_SW / STD_final)
COS_RMSE = 0.5 * (RMSE_SW / RMSE_final) + 0.5 * (STD_SW / STD_final)
```

The confidence intervals for COS are indicative because the sliding windows overlap.

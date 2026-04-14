import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Configure the target Grand Prix to load corresponding data
target_gp_name = os.environ.get('TARGET_GP_NAME', 'Bahrain Grand Prix')
safe_gp_name = target_gp_name.lower().replace(' ', '_')

# Define paths to read the ModelData file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
model_data_dir = os.path.join(parent_dir, 'ModelData', target_gp_name)
input_csv_path = os.path.join(model_data_dir, f"{safe_gp_name}_cleaned_data.csv")

print(f"Loading cleaned data from:\n{input_csv_path}")
if not os.path.exists(input_csv_path):
    raise FileNotFoundError(f"File not found: {input_csv_path}. Run script_model_data.py first.")

# Load the DataFrame
laps_cleaned = pd.read_csv(input_csv_path)

target_col = 'LapTime_seconds'
num_cols = [
    'TyreLife', 'LapNumber',
    'Humidity_RBF_Median','Pressure_RBF_Median', 'TrackTemp_RBF_Median', 
    'WindSpeed_RBF_Median', 'TempDelta_RBF_Median', 'WindDirection_RBF_Median', 'LapTime_prev'
]
cat_cols = ['Driver', 'Team', 'pirelliCompound', 'Year']

print("\nPreparing data for Linear Regression (Baseline)...")

num_cols = [c for c in num_cols if c in laps_cleaned.columns]
cat_cols = [c for c in cat_cols if c in laps_cleaned.columns]

X_base = laps_cleaned[num_cols + cat_cols].copy()
y = laps_cleaned[target_col].copy()

valid_indices = y.dropna().index
X_base = X_base.loc[valid_indices]
y = y.loc[valid_indices]

X_encoded = pd.get_dummies(X_base, columns=cat_cols, drop_first=True)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X_train_raw.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test_raw), columns=X_test_raw.columns)

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train_imputed.columns)
X_test = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test_imputed.columns)

print(f"Shape final de X_train: {X_train.shape}")
print(f"Shape final de X_test: {X_test.shape}")

print("\nTreinando Linear Regression...")
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

y_pred_train = model_lr.predict(X_train)
y_pred_test = model_lr.predict(X_test)

rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("\n--- BASELINE RESULTS (Linear Regression) ---")
print(f"RMSE (Root Mean Squared Error): {rmse_test:.4f} seconds")
print(f"MAE (Erro Mean Absoluto):    {mae_test:.4f} seconds")
print(f"R2 (Coefficient of Determination):   {r2_test:.4f}")

coefs = pd.DataFrame({'Feature': X_train.columns, 'Coef': model_lr.coef_})
coefs['Abs_Coef'] = coefs['Coef'].abs()

print("\nMost Impactful Features:")
print(coefs.sort_values(by='Abs_Coef', ascending=False).head(15).to_string(index=False))





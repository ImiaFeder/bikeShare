

# %% [markdown]
# # Import

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import GridSearchCV


# %% [markdown]
# # Day

# %% [markdown]
# ### Load Dataset

# %%
# Load daily dataset
day_url = "https://raw.githubusercontent.com/michaeldavidsinn/csvml/refs/heads/main/day.csv"
day_df = pd.read_csv(day_url)

# %%
# Show dataset
day_df.head()

# %% [markdown]
# ### Pre-Processing

# %%
# Show missing values
print(day_df.isnull().sum())

# %%
# Show info
day_df.info()

# %%
# Describe data
day_df.describe()

# %%
# Convert 'dteday' to datetime format and set index
day_df['dteday'] = pd.to_datetime(day_df['dteday'])
day_df.set_index('dteday', inplace=True)

# %%
# Selected features for daily modeling
selected_features = ['temp', 'hum', 'windspeed', 'season', 'weathersit', 'yr', 'mnth', 'holiday', 'weekday', 'workingday']

# %%
# Scale the features for XGBoost (daily data)
scaler = StandardScaler()
day_df[selected_features] = scaler.fit_transform(day_df[selected_features])

# %%
# Split day data 
X_day = day_df[selected_features]
y_day = day_df['cnt']
# %% [markdown]
# ### Visualization



# %% [markdown]
# ### Processing

# %%
# SARIMAX model fitting (daily)
sarimax_model = SARIMAX(day_df['cnt'],
                        exog=day_df[selected_features],
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, 12),  # Monthly seasonality
                        enforce_stationarity=False,
                        enforce_invertibility=False)
results = sarimax_model.fit()

sarimax_predictions = results.predict(start=day_df.index[0], end=day_df.index[-1], exog=day_df[selected_features])
residuals = day_df['cnt'] - sarimax_predictions


# %%
X_train_residuals = X_day
y_train_residuals = residuals.loc[X_day.index]

# %%
# Hyperparameter tuning for XGBoost
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
xgb_model = XGBRegressor(random_state=42)

# %%
# Perform GridSearch 
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train_residuals, y_train_residuals)
best_xgb_model = grid_search.best_estimator_
print(f"Best XGBoost Parameters (Daily): {grid_search.best_params_}")

# %%
# Predict residuals using tuned XGBoost model (daily)
xgb_residual_predictions = best_xgb_model.predict(X_train_residuals)

# %%
day_df[selected_features]

# %% [markdown]
# ### Evaluation

# %%
# Predict using SARIMAX and XGBoost for the entire dataset (daily)
sarimax_predictions_full = results.predict(start=day_df.index[0], end=day_df.index[-1], exog=day_df[selected_features])
xgb_residual_predictions_full = best_xgb_model.predict(day_df[selected_features])

# %%
# Combine SARIMAX and XGBoost predictions (daily)
final_predictions = sarimax_predictions_full + xgb_residual_predictions_full

# %%
# Evaluate combined model (daily)
mae_combined = mean_absolute_error(day_df['cnt'], final_predictions)
mse_combined = mean_squared_error(day_df['cnt'], final_predictions)
r2_combined = r2_score(day_df['cnt'], final_predictions)
print(f"Combined Model MAE (Daily): {mae_combined:.2f}")
print(f"Combined Model MSE (Daily): {mse_combined:.2f}")
print(f"Combined Model R2 (Daily): {r2_combined:.2f}")

# %% [markdown]
# ### Model Result


# %% [markdown]
# # Hour

# %% [markdown]
# ### Load Dataset

# %%
# Load hourly dataset
hour_url = "https://raw.githubusercontent.com/michaeldavidsinn/csvml/refs/heads/main/hour.csv"
hour_df = pd.read_csv(hour_url)

# %%
# Show dataset
hour_df.head()

# %% [markdown]
# ### Pre-Processing

# %%
# Show missing values
print(hour_df.isnull().sum())

# %%
# Show info
hour_df.info()

# %%
# Describe data
hour_df.describe()

# %%
# Convert 'dteday' to datetime format and set datetime index (hourly)
hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])
hour_df['datetime'] = hour_df['dteday'] + pd.to_timedelta(hour_df['hr'], unit='h')
hour_df.set_index('datetime', inplace=True)

# %%
# Ensure sample size is not larger than the dataset
sample_size = 100000  # Adjust as needed
total_rows = len(hour_df)
sample_size = min(sample_size, total_rows)

# %%
# Sample the data
hour_df_sampled = hour_df.sample(n=sample_size, random_state=42)

# %%
# Selected features for hourly modeling
selected_features_hour = ['temp', 'hum', 'windspeed', 'season', 'weathersit', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'hr']

# %%
# Scale the features for XGBoost (daily data)
scaler = StandardScaler()
hour_df[selected_features_hour] = scaler.fit_transform(hour_df[selected_features_hour])

# %%
# Split hour data 
X_hour = hour_df_sampled[selected_features_hour]
y_hour = hour_df_sampled['cnt']

# %% [markdown]
# ### Visualization


# %% [markdown]
# ### Processing

# %%
# XGBoost model fitting (hourly)
xgb_model = XGBRegressor(random_state=42)


# %%
# Hyperparameter tuning for XGBoost
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 5, 10]
}


# %%
# Perform Grid Search 
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1)

grid_search.fit(X_hour, y_hour)
xgb_model_hour = grid_search.best_estimator_

# %%
# Predict using XGBoost (hourly)
xgb_predictions_hour = xgb_model_hour.predict(X_hour)

# %% [markdown]
# ### Evaluation

# %%
# Evaluate XGBoost (hourly)
xgb_mae_hour = mean_absolute_error(y_hour, xgb_predictions_hour)
xgb_mse_hour = mean_squared_error(y_hour, xgb_predictions_hour)
xgb_r2_hour = r2_score(y_hour, xgb_predictions_hour)

print(f"XGBoost MAE (Hourly): {xgb_mae_hour:.2f}")
print(f"XGBoost MSE (Hourly): {xgb_mse_hour:.2f}")
print(f"XGBoost R2 (Hourly): {xgb_r2_hour:.2f}")

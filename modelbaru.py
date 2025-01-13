
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

file_path = "day.csv"
day_df = pd.read_csv(file_path)
day_df1 = pd.read_csv(file_path)

# Show dataset
day_df.head()


# Convert 'dteday' to datetime format and set index
day_df['dteday'] = pd.to_datetime(day_df['dteday'])
day_df.set_index('dteday', inplace=True)

# Selected features for daily modeling
selected_features = ['temp', 'hum', 'windspeed', 'season', 'weathersit', 'yr', 'mnth', 'holiday', 'weekday', 'workingday']

# # Scale the features for XGBoost (daily data)
scaler = StandardScaler()
day_df[selected_features] = scaler.fit_transform(day_df[selected_features])

# Split day data into train and test sets
X_day = day_df[selected_features]
y_day = day_df['cnt']

# SARIMAX model fitting (daily)
sarimax_model = SARIMAX(day_df['cnt'],
                        exog=day_df[selected_features],
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, 12),  # Monthly seasonality
                        enforce_stationarity=False,
                        enforce_invertibility=False)
results = sarimax_model.fit()

sarimax_predictions = results.forecast(steps=len(day_df), exog=day_df[selected_features])
residuals = day_df['cnt'] - sarimax_predictions


X_train_residuals = day_df[selected_features]
y_train_residuals = residuals.loc[day_df[selected_features].index]


param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
xgb_model = XGBRegressor(random_state=42)

# Perform GridSearch 
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train_residuals, y_train_residuals)
best_xgb_model = grid_search.best_estimator_


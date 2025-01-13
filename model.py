import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

def forecast_and_save(exog_df):
    

    # Load dataset
    day_df = pd.read_csv("day.csv")
    
    # Preprocess dataset
    day_df['dteday'] = pd.to_datetime(day_df['dteday'])
    day_df.set_index('dteday', inplace=True)
    
    selected_features = ['temp', 'hum', 'windspeed', 'season', 'weathersit', 'yr', 'mnth', 'holiday', 'weekday', 'workingday']
    
    # Scale the selected features
    scaler = StandardScaler()
    day_df[selected_features] = scaler.fit_transform(day_df[selected_features])
    exog_df[selected_features] = scaler.transform(exog_df[selected_features])
    
    # Prepare data for SARIMAX
    X_day = day_df[selected_features]
    y_day = day_df['cnt']
    
    # Fit SARIMAX model
    sarimax_model = SARIMAX(y_day,
                            exog=X_day,
                            order=(1, 1, 1),
                            seasonal_order=(1, 1, 1, 12),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
    results = sarimax_model.fit()
    
    # Generate SARIMAX predictions for the next step
    sarimax_prediction = results.forecast(steps=1, exog=exog_df[selected_features])
    
    # Calculate residuals for XGBoost
    residuals = day_df['cnt'] - results.fittedvalues
    X_residuals = day_df[selected_features]
    y_residuals = residuals.loc[X_residuals.index]
    
    # Train XGBoost model with GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    xgb_model = XGBRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_residuals, y_residuals)
    best_xgb_model = grid_search.best_estimator_
    
    # Predict residuals using XGBoost
    xgb_prediction = best_xgb_model.predict(exog_df[selected_features])
    
    # Final prediction (SARIMAX + XGBoost)
    final_prediction = sarimax_prediction + xgb_prediction
    
    # Create a new row for the prediction
    new_row = exog_df.copy()
    new_row['cnt'] = final_prediction[0]
    new_row['instant'] = len(day_df) + 1
    new_row = new_row.reset_index()
    new_row['dteday'] = new_row['dteday'].dt.strftime('%Y-%m-%d')  # Format date
    
    # Append the new row to the original DataFrame
    all_columns = ['instant', 'dteday', 'season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 
                   'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
    for col in all_columns:
        if col not in new_row.columns:
            new_row[col] = np.nan
    new_row = new_row[all_columns]  # Reorder columns
    
    day_df.reset_index(inplace=True)
    updated_df = pd.concat([day_df, new_row], ignore_index=True)
    
    # Save the updated DataFrame to the CSV file
    updated_df.to_csv("day1.csv", index=False)
    
    print(f"Predictions saved to day.csv")
    return final_prediction


dummy_data = {
    'dteday': ['2025-01-14'],  # Tanggal prediksi
    'temp': [0.45],            # Fitur suhu (dummy)
    'hum': [0.65],             # Fitur kelembapan (dummy)
    'windspeed': [0.2],        # Fitur kecepatan angin (dummy)
    'season': [1],             # Musim (dummy)
    'weathersit': [2],         # Situasi cuaca (dummy)
    'yr': [1],                 # Tahun (dummy, 0: 2011, 1: 2012)
    'mnth': [1],               # Bulan (dummy, 1: Januari, dst.)
    'holiday': [0],            # Hari libur (dummy, 0: bukan libur, 1: libur)
    'weekday': [3],            # Hari kerja (dummy, 0: Minggu, dst.)
    'workingday': [1],         # Hari kerja efektif (dummy, 0: tidak, 1: ya)
}

# Konversi ke DataFrame
exog_df = pd.DataFrame(dummy_data)

# Pastikan 'dteday' dikonversi ke datetime
exog_df['dteday'] = pd.to_datetime(exog_df['dteday'])

# Tampilkan DataFrame dummy
print(exog_df)

print(forecast_and_save(exog_df))


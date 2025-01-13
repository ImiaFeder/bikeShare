from datetime import timedelta
from flask import Flask, request, render_template, jsonify
import pandas as pd
from modelbaru import best_xgb_model, results, selected_features, day_df1
from model import forecast_and_save
import numpy as np

app = Flask(__name__)

# Definisikan t_min dan t_max untuk normalisasi temperatur
T_MIN = -8
T_MAX = 39
WINDSPEED_MAX = 67

def get_next_day_year_month(day_df):
    # Convert 'dteday' to datetime format
    day_df['dteday'] = pd.to_datetime(day_df['dteday'])
    
    # Get the last date in the dataframe (latest entry)
    last_date = day_df['dteday'].max()
    
    # Calculate the next day
    next_day = last_date + timedelta(days=1)
    
    # Return the year and month of the next day
    return next_day.year, next_day.month

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/hourly_predictions')
def hourly_prediction():
    return render_template('hourly_prediction.html') 

@app.route('/predict_day', methods=['POST'])
def predict_day():
    try:
        # Ambil data JSON dari permintaan pengguna
        data = request.get_json()

        # Cek apakah data ada dan tidak kosong
        if not data:
            return "No data provided.", 400  # Mengirimkan teks jika tidak ada data

        # Konversi data ke DataFrame
        if isinstance(data, dict):
            data = [data]  # Jika data berbentuk dictionary, ubah menjadi list of dicts

        year, month = get_next_day_year_month(day_df1)

        df = pd.DataFrame(data)

        df['yr'] = year-2012
        df['mnth'] = month
        

        # Validasi input untuk memastikan kolom yang dibutuhkan ada
        required_columns = ["temp", "hum", "windspeed", "season", "weathersit", "yr", "mnth", "holiday", "weekday", "workingday"]
        if not all(col in df.columns for col in required_columns):
            return "Missing required columns in input data.", 400  # Mengirimkan teks jika kolom hilang

        # Pastikan DataFrame tidak kosong
        if df.empty:
            return "Empty input data.", 400  # Mengirimkan teks jika data kosong

        # Normalisasi kolom sesuai permintaan
        df["temp"] = (df["temp"] - T_MIN) / (T_MAX - T_MIN)  # Normalisasi temperatur
        df["hum"] = df["hum"] / 100  # Normalisasi kelembaban
        df["windspeed"] = df["windspeed"] / WINDSPEED_MAX  # Normalisasi kecepatan angin

        # Prediksi menggunakan model SARIMAX (menggunakan exogenous variable yang sesuai)
        # sarimax_predictions_full = results.predict(
        #     start=df.index[0],  # Mulai prediksi dari data pertama
        #     end=day_df.index[0],  # Hingga data pertama
        #     exog=df[selected_features]  # Variabel eksternal untuk model SARIMAX
        # )

        print(df)
        # Prediksi XGBoost menggunakan residual dari model sebelumnya
     
        # Bulatkan hasil prediksi
        final_predictions_rounded = forecast_and_save(df)

        # Kirim hasil prediksi sebagai teks biasa
        return str(final_predictions_rounded.tolist()[0])  # Mengirimkan hasil prediksi sebagai angka bulat

    except Exception as e:
        # Menangani kesalahan dan mengirimkan pesan error
        return str(e), 500  # Mengirimkan teks error jika terjadi kesalahan
    

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, render_template, jsonify
import pandas as pd
from modelbaru import best_xgb_model, results, selected_features, day_df
import numpy as np

app = Flask(__name__)

# Definisikan t_min dan t_max untuk normalisasi temperatur
T_MIN = -8
T_MAX = 39
WINDSPEED_MAX = 67

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/daily_predictions')
def daily_prediction():
    return render_template('daily_prediction.html')  

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

        df = pd.DataFrame(data)

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
        sarimax_predictions_full = results.predict(
            start=df.index[0],  # Mulai prediksi dari data pertama
            end=day_df.index[0],  # Hingga data pertama
            exog=df[selected_features]  # Variabel eksternal untuk model SARIMAX
        )

        # Prediksi XGBoost menggunakan residual dari model sebelumnya
        xgb_residual_predictions_full = best_xgb_model.predict(df[selected_features])

        # Gabungkan hasil prediksi SARIMAX dan XGBoost
        final_predictions = sarimax_predictions_full + xgb_residual_predictions_full

        # Bulatkan hasil prediksi
        final_predictions_rounded = np.round(final_predictions)

        # Kirim hasil prediksi sebagai teks biasa
        return str(final_predictions_rounded.tolist()[0])  # Mengirimkan hasil prediksi sebagai angka bulat

    except Exception as e:
        # Menangani kesalahan dan mengirimkan pesan error
        return str(e), 500  # Mengirimkan teks error jika terjadi kesalahan
    
@app.route('/predict_hour', methods=['POST'])
def predict_hour():
    try:
        # Ambil data JSON dari permintaan pengguna
        data = request.get_json()

        # Cek apakah data ada dan tidak kosong
        if not data:
            return "No data provided.", 400  # Mengirimkan teks jika tidak ada data

        # Konversi data ke DataFrame
        if isinstance(data, dict):
            data = [data]  # Jika data berbentuk dictionary, ubah menjadi list of dicts

        df = pd.DataFrame(data)

        # Validasi input untuk memastikan kolom yang dibutuhkan ada
        required_columns = ["temp", "hum", "windspeed", "season", "weathersit", "yr", "mnth", "holiday", "weekday", "workingday", "hr"]
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
        sarimax_predictions_full = results.predict(
            start=df.index[0],  # Mulai prediksi dari data pertama
            end=day_df.index[0],  # Hingga data pertama
            exog=df[selected_features]  # Variabel eksternal untuk model SARIMAX
        )

        # Prediksi XGBoost menggunakan residual dari model sebelumnya
        xgb_residual_predictions_full = best_xgb_model.predict(df[selected_features])

        # Gabungkan hasil prediksi SARIMAX dan XGBoost
        final_predictions = sarimax_predictions_full + xgb_residual_predictions_full

        # Bulatkan hasil prediksi
        final_predictions_rounded = np.round(final_predictions)

        # Kirim hasil prediksi sebagai teks biasa
        return str(final_predictions_rounded.tolist()[0])  # Mengirimkan hasil prediksi sebagai angka bulat

    except Exception as e:
        # Menangani kesalahan dan mengirimkan pesan error
        return str(e), 500  # Mengirimkan teks error jika terjadi kesalahan


if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, render_template, jsonify
import pandas as pd
from model import best_xgb_model, results, selected_features, day_df

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Halaman input data

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data JSON dari permintaan pengguna
        data = request.get_json()

        # Cek apakah data ada dan tidak kosong
        if not data:
            return jsonify({"error": "No data provided."}), 400

        # Konversi data ke DataFrame
        # Pastikan data yang diterima berbentuk list (bukan scalar)
        if isinstance(data, dict):
            data = [data]  # Jika data berbentuk dictionary, ubah menjadi list of dicts

        df = pd.DataFrame(data)

        # Validasi input untuk memastikan kolom yang dibutuhkan ada
        required_columns = ["temp", "hum", "windspeed", "season", "weathersit", "yr", "mnth", "holiday", "weekday", "workingday"]
        if not all(col in df.columns for col in required_columns):
            return jsonify({"error": "Missing required columns in input data."}), 400

        # Pastikan DataFrame tidak kosong
        if df.empty:
            return jsonify({"error": "Empty input data."}), 400

        # Prediksi menggunakan model SARIMAX (menggunakan exogenous variable yang sesuai)
        # Pastikan Anda mengirimkan data yang sesuai untuk exogenous variable pada model SARIMAX
        
        # Prediksi SARIMAX
        sarimax_predictions_full = results.predict(
            start=df.index[0],  # Mulai prediksi dari data pertama
            end=day_df.index[0],    # Hingga data pertama
            exog=df[selected_features]     # Variabel eksternal untuk model SARIMAX
        )

        # Prediksi XGBoost menggunakan residual dari model sebelumnya
        xgb_residual_predictions_full = best_xgb_model.predict(df[selected_features])

        # Gabungkan hasil prediksi SARIMAX dan XGBoost
        final_predictions = sarimax_predictions_full + xgb_residual_predictions_full

        # Kirim hasil prediksi sebagai respon JSON
        return jsonify({"final_predictions": final_predictions.tolist()})

    except Exception as e:
        # Menangani kesalahan dan mengirimkan pesan error
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

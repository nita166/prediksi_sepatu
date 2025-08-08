from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load('model_sepatu.pkl')

@app.route('/')
def home():
    return "✅ Aplikasi Prediksi Harga Sepatu Aktif!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Pastikan input punya kolom yang sama seperti saat training
    df_input = pd.DataFrame([data], columns=['Brand_Name', 'How_Many_Sold', 'RATING'])

    prediction = model.predict(df_input)
    return jsonify({'prediksi_harga': float(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=81)
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Inisialisasi Flask app
app = Flask(__name__)

# Load model yang sudah dipickle
model = joblib.load('model_sepatu.pkl')

@app.route('/')
def home():
    return "Aplikasi Prediksi Harga Sepatu Aktif!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = np.array(data['input']).reshape(1, -1)
    prediction = model.predict(input_data)
    return jsonify({'prediksi_harga': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)

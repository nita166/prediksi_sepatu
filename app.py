import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('model_sepatu.pkl')

# Judul aplikasi
st.title("Prediksi Harga Sepatu")
st.write("Masukkan fitur sepatu untuk memprediksi harga.")

# Input dari pengguna
rating = st.number_input("Rating Produk", min_value=0.0, max_value=5.0, step=0.1)
jumlah_terjual = st.number_input("Jumlah Terjual", min_value=0)

# Prediksi
if st.button("Prediksi"):
    input_data = np.array([[rating, jumlah_terjual]])
    prediksi = model.predict(input_data)
    st.success(f"Prediksi Harga Sepatu: ₹{prediksi[0]:,.2f}")

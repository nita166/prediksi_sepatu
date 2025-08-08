# ============================================================
# 👟 SHOE PRICE PREDICTION APP
# ============================================================

import streamlit as st
import joblib
import pandas as pd
import os

# ============================================================
# 1️⃣ Load Best Model
# ============================================================
@st.cache_resource
def load_model():
    """
    Memuat pipeline model dari file .pkl.
    Fungsi ini menggunakan st.cache_resource agar model hanya dimuat sekali.
    """
    model_path = "model_sepatu.pkl"
    if not os.path.exists(model_path):
        st.error(f"❌ File model '{model_path}' tidak ditemukan.")
        st.stop()
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# ============================================================
# 2️⃣ App Configuration
# ============================================================
st.set_page_config(
    page_title="Prediksi Harga Sepatu",
    page_icon="👟",
    layout="centered"
)

st.title("👟 Aplikasi Prediksi Harga Sepatu")
st.markdown("Masukkan detail sepatu untuk memprediksi harga.")

# ============================================================
# 3️⃣ Input Form
# ============================================================
with st.form("shoe_price_form"):
    st.subheader("📝 Masukkan Detail Sepatu")
    
    # Daftar brand yang umum dari dataset
    brand_options = [
        "ASIAN", "Reebok", "Puma", "Generic", "Sparx", "BATA", 
        "Robbie jones", "Campus", "Bourge", "Adidas", "Nivia"
    ]
    
    # Input untuk fitur-fitur yang digunakan model
    brand_name = st.selectbox("Nama Merek", options=brand_options)
    how_many_sold = st.number_input("Jumlah Terjual (misal: 2242)", min_value=0)
    rating = st.slider("Rating Produk", min_value=0.0, max_value=5.0, value=4.0, step=0.1)
    product_details = st.text_area("Detail Produk", value="Oxygen-01 Sports Running,Walking & Gym Shoes with Oxygen Technology Lightweight Casual Sneaker Shoes for Men's & Boy's")
    
    submitted = st.form_submit_button("🔮 Prediksi Harga")
    
    if submitted:
        try:
            # Buat DataFrame dari input mentah pengguna
            input_data = pd.DataFrame([{
                'Brand_Name': brand_name,
                'How_Many_Sold': how_many_sold,
                'Product_details': product_details,
                'RATING': rating
            }])
            
            # Prediksi menggunakan pipeline
            prediction = model.predict(input_data)
            
            st.subheader("✅ Prediksi Berhasil!")
            st.success(f"Harga sepatu yang diprediksi adalah: ₹{prediction[0]:,.2f}")

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")
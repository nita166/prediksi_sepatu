# ============================================================
# 🌳 SHOE PRICE PREDICTION APP - DECISION TREE
# ============================================================

import streamlit as st
import joblib
import pandas as pd
import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# ============================================================
# 1️⃣ Definisi Pipeline Pra-pemrosesan
# ============================================================
def create_preprocessor():
    """
    Membangun ulang ColumnTransformer untuk pra-pemrosesan data numerik.
    Ini adalah cara yang lebih stabil daripada memuatnya dari file, 
    karena model Decision Tree Anda dilatih hanya dengan fitur numerik.
    """
    # Kolom numerik yang akan diskalakan
    numeric_features = ['How_Many_Sold', 'RATING']
    
    # Membuat transformer untuk kolom numerik
    numeric_transformer = StandardScaler()
    
    # Menggabungkan transformer ke dalam ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

# ============================================================
# 2️⃣ Load Model dan Gabungkan ke dalam Pipeline
# ============================================================
@st.cache_resource
def load_model_pipeline():
    """
    Memuat model Decision Tree yang sudah terlatih dari file .pkl
    dan menggabungkannya ke dalam pipeline pra-pemrosesan yang baru dibuat.
    """
    model_path = "model_sepatu_tree.pkl"
    if not os.path.exists(model_path):
        st.error(f"❌ File model '{model_path}' tidak ditemukan. Pastikan Anda sudah menjalankan script untuk membuat file ini.")
        st.stop()

    try:
        # Memuat hanya model DecisionTreeRegressor yang telah dilatih
        trained_model = joblib.load(model_path)
        
        # Membuat ulang preprocessor
        preprocessor = create_preprocessor()
        
        # Membuat pipeline akhir yang menggabungkan preprocessor dan model
        final_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', trained_model) # Menggunakan 'regressor' sebagai nama step
        ])
        
        return final_pipeline
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}. Pastikan versi pustaka Anda konsisten.")
        st.stop()

# ============================================================
# 3️⃣ App Configuration
# ============================================================
st.set_page_config(
    page_title="Prediksi Harga Sepatu",
    page_icon="👟",
    layout="centered"
)

st.title("👟 Aplikasi Prediksi Harga Sepatu (Decision Tree)")
st.markdown("Masukkan detail sepatu untuk memprediksi harga.")

# ============================================================
# 4️⃣ Input Form
# ============================================================
with st.form("shoe_price_form"):
    st.subheader("📝 Masukkan Detail Sepatu")
    
    # Karena model Decision Tree dilatih hanya dengan fitur numerik,
    # input yang diperlukan hanya 'How_Many_Sold' dan 'RATING'.
    how_many_sold = st.number_input("Jumlah Terjual (misal: 2242)", min_value=0)
    rating = st.slider("Rating Produk", min_value=0.0, max_value=5.0, value=4.0, step=0.1)
    
    submitted = st.form_submit_button("🔮 Prediksi Harga")
    
    if submitted:
        try:
            # Buat DataFrame dari input pengguna
            input_data = pd.DataFrame([{
                'How_Many_Sold': how_many_sold,
                'RATING': rating
            }])
            
            model = load_model_pipeline()
            prediction = model.predict(input_data)
            
            st.subheader("✅ Prediksi Berhasil!")
            st.success(f"Harga sepatu yang diprediksi adalah: ₹{prediction[0]:,.2f}")

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")
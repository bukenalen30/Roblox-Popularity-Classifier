import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from PIL import Image

st.set_page_config(page_title="Roblox Popularity Classifier 🌈", layout="wide")

# ==============================================
# CUSTOM CSS - BACKGROUND BIRU CERIA
# ==============================================
st.markdown(
    """
    <style>
    /* Background biru ceria untuk seluruh app */
    .stApp {
        background: linear-gradient(135deg, #cceeff, #99ddff);
    }

    /* Hapus padding default Streamlit */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    </style>
    """, unsafe_allow_html=True
)

# ==============================================
# CERIA HEADER
# ==============================================
# Load logo Roblox
logo = Image.open("logo_roblox.jpeg")  # pastikan file ada di folder yang sama

# Layout 3 kolom: kiri kosong, tengah judul+emoji, kanan logo
col1, col2, col3 = st.columns([1, 4, 1])

with col1:
    st.write("")

with col2:
    st.markdown(
        """
        <div style="
            text-align:center; 
            background: linear-gradient(90deg, #ffeb3b, #ff5722, #2196f3, #4caf50, #e91e63);
            padding:25px 20px; 
            border-radius:12px; 
            border:2px solid #ffc107;">
            <h1 style="margin:0; line-height:1.2; color:white;">Roblox Popularity Classifier 📊</h1>
            <p style="margin:0; font-size:16px; line-height:1.2; color:white;">Prediksi tingkat popularitas game Roblox menggunakan model SVM & KNN</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    # container untuk vertical center logo
    st.markdown(
        """
        <div style="
            display:flex;
            align-items:center;
            height:100%;
        "></div>
        """,
        unsafe_allow_html=True
    )
    st.image(logo, width=100)  # logo di kanan, vertikal center


with col3:
    # Gunakan container flex agar logo vertikal center
    st.markdown(
        f"""
        <div style="
            display:flex; 
            align-items:center;      /* center vertical */
            height:100%;             /* isi tinggi kolom sesuai bar header */
        ">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.image(logo, width=100)  # logo di kanan, pasti sejajar vertikal

# ==============================================
# LOAD MODEL DAN RESOURCE
# ==============================================
@st.cache_resource
def load_all():
    try:
        svm = joblib.load("svm_model.pkl")
        knn = joblib.load("knn_model.pkl")
        scaler = joblib.load("scaler.pkl")
        features = joblib.load("features.pkl")
        
        # Load confusion matrix jika ada
        svm_matrix = joblib.load("svm_confusion_matrix.pkl") if "svm_confusion_matrix.pkl" in os.listdir() else None
        knn_matrix = joblib.load("knn_confusion_matrix.pkl") if "knn_confusion_matrix.pkl" in os.listdir() else None
        
        return svm, knn, scaler, features, svm_matrix, knn_matrix
    except Exception as e:
        st.error(f"❌ Gagal load model atau resource: {e}")
        return None, None, None, None, None, None

svm_model, knn_model, scaler, feature_cols, svm_matrix, knn_matrix = load_all()

# ==============================================
# CEK VALIDITAS MODEL
# ==============================================
def model_invalid(model, name):
    if model is None:
        return True
    if isinstance(model, np.ndarray):
        st.error(f"❌ ERROR: {name} adalah numpy.ndarray — file PKL salah.")
        return True
    if not hasattr(model, "predict"):
        st.error(f"❌ ERROR: {name} tidak punya method .predict() — file PKL rusak.")
        return True
    return False

invalid_svm = model_invalid(svm_model, "svm_model.pkl")
invalid_knn = model_invalid(knn_model, "knn_model.pkl")

if scaler is None:
    st.error("❌ ERROR: scaler.pkl gagal dimuat.")
    invalid_svm = invalid_knn = True

if feature_cols is None:
    st.error("❌ ERROR: features.pkl gagal dimuat.")
    invalid_svm = invalid_knn = True

# ==============================================
# SIDEBAR INPUT
# ==============================================
st.sidebar.write("### Masukkan fitur game Roblox")
inputs = {}
for col in feature_cols:
    inputs[col] = st.sidebar.number_input(col, min_value=0, value=0)

# ==============================================
# PREDIKSI
# ==============================================
if st.sidebar.button("🌟 Prediksi"):

    if invalid_svm or invalid_knn:
        st.error("❌ Tidak dapat melakukan prediksi karena model tidak valid.")
    else:

        x_df = pd.DataFrame([list(inputs.values())], columns=feature_cols)
        st.write("### 🔍 Input DataFrame:")
        st.write(x_df)

        x_scaled = scaler.transform(x_df)

        svm_pred = svm_model.predict(x_scaled)[0]
        knn_pred = knn_model.predict(x_scaled)[0]

        # ==============================
        # Mapping label aman
        # ==============================
        label_map = {0: "Low", 1: "Medium", 2: "High"}
        svm_label = label_map.get(svm_pred, f"Unknown ({svm_pred})")
        knn_label = label_map.get(knn_pred, f"Unknown ({knn_pred})")

        st.subheader("🔮 Hasil Prediksi Popularitas Game")
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**SVM:** {svm_label}")
        with col2:
            st.info(f"**KNN:** {knn_label}")

# ==============================================
# VISUALISASI CONFUSION MATRIX DENGAN WARNA SOFT & BERBEDA
# ==============================================
if svm_matrix is not None or knn_matrix is not None:
    st.header("📊 Confusion Matrix Model")

    def plot_matrix(matrix, title, cmap_color):
        fig, ax = plt.subplots()
        # Warna soft sesuai cmap_color
        cax = ax.imshow(matrix, cmap=cmap_color, alpha=0.8)
        ax.set_title(title, color="#333333")
        ax.set_xlabel("Predicted", color="#333333")
        ax.set_ylabel("Actual", color="#333333")
        
        # Tampilkan angka di tengah kotak
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, matrix[i, j], ha="center", va="center", color="black", fontsize=12)
        
        # Tambahkan colorbar
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

    colA, colB = st.columns(2)
    with colA:
        if svm_matrix is not None:
            # SVM = soft biru
            plot_matrix(svm_matrix, "Confusion Matrix - SVM", cmap_color="Blues")
    with colB:
        if knn_matrix is not None:
            # KNN = soft hijau
            plot_matrix(knn_matrix, "Confusion Matrix - KNN", cmap_color="Greens")

st.write("---")
st.caption("🌈 © 2025 — Roblox Popularity ML Deployment | Ceria Theme 🌈")

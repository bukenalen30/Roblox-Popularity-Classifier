import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Roblox Popularity Classifier 🌈", layout="wide")

# ==============================================
# CERIA HEADER
# ==============================================
st.markdown("""
    <div style="background: linear-gradient(90deg, #ffeb3b, #ff5722, #2196f3, #4caf50, #e91e63);
                padding:20px; border-radius:12px; border:2px solid #ffc107; margin-bottom:20px;">
        <h1 style="color:white; text-align:center;">🌈 Roblox Game Popularity Classifier 🌈</h1>
        <p style="color:white; text-align:center;">Prediksi tingkat popularitas game Roblox menggunakan model SVM & KNN.</p>
    </div>
""", unsafe_allow_html=True)

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
        label_encoder = joblib.load("label_encoder.pkl")
        evaluation = joblib.load("evaluation.pkl") if "evaluation.pkl" in os.listdir() else None
        return svm, knn, scaler, features, label_encoder, evaluation
    except Exception as e:
        st.error(f"❌ Gagal load model atau resource: {e}")
        return None, None, None, None, None, None

import os
svm_model, knn_model, scaler, feature_cols, label_encoder, evaluation = load_all()

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

if feature_cols is None or label_encoder is None:
    st.error("❌ ERROR: features.pkl atau label_encoder.pkl gagal dimuat.")
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

        # Jika label encoder ada
        if label_encoder is not None:
            svm_label = label_encoder.inverse_transform([svm_pred])[0]
            knn_label = label_encoder.inverse_transform([knn_pred])[0]
        else:
            svm_label = str(svm_pred)
            knn_label = str(knn_pred)

        st.subheader("🔮 Hasil Prediksi")
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**SVM:** {svm_label}")
        with col2:
            st.info(f"**KNN:** {knn_label}")

# ==============================================
# VISUALISASI EVALUASI
# ==============================================
if evaluation:
    st.header("📊 Visualisasi Evaluasi Model")

    svm_matrix = evaluation.get("svm_matrix")
    knn_matrix = evaluation.get("knn_matrix")

    def plot_matrix(matrix, title):
        fig, ax = plt.subplots()
        ax.imshow(matrix, cmap="coolwarm")
        ax.set_title(title, color="#ff5722")
        ax.set_xlabel("Predicted", color="#4caf50")
        ax.set_ylabel("Actual", color="#4caf50")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, matrix[i, j], ha="center", va="center", color="black")
        st.pyplot(fig)

    colA, colB = st.columns(2)
    with colA:
        if svm_matrix is not None:
            plot_matrix(svm_matrix, "Confusion Matrix - SVM")
    with colB:
        if knn_matrix is not None:
            plot_matrix(knn_matrix, "Confusion Matrix - KNN")

    st.header("📈 Perbandingan Metrik Evaluasi")
    st.subheader("SVM Classification Report")
    st.code(evaluation.get("svm_report", "Tidak ada."))

    st.subheader("KNN Classification Report")
    st.code(evaluation.get("knn_report", "Tidak ada."))

st.write("---")
st.caption("🌈 © 2025 — Roblox Popularity ML Deployment | Ceria Theme 🌈")

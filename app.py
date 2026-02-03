import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from PIL import Image

# ==============================================
# CUSTOM CSS - BACKGROUND BIRU CERIA
# ==============================================
st.set_page_config(page_title="Roblox Popularity Classifier 🌈", layout="wide")

# ==============================================
# CUSTOM CSS - SIDEBAR BIRU TUA CERAH
# ==============================================
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d47a1, #1565c0, #1e88e5);
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    [data-testid="stSidebar"] input {
        background-color: white;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================================
# CUSTOM CSS - BACKGROUND MAIN PAGE
# ==============================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #cceeff, #99ddff);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================================
# CERIA HEADER
# ==============================================
col1, col2, col3 = st.columns([1, 6, 1])

with col1:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.image("ysalen.png", width=180)

with col2:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="
            text-align:center; 
            background: linear-gradient(90deg, #f06292, #f48fb1, #f8bbd0, #fce4ec);
            padding:20px; 
            border-radius:20px; 
            border:8px solid #ad1457;
            margin-bottom:20px;
        ">
            <h1 style="margin:5px 0; line-height:1.2; color:white; font-size:38px;">
                Roblox Popularity Classifier 📊
            </h1>
            <p style="margin:5px 0; font-size:18px; line-height:1.2; color:white;">
                Prediksi tingkat popularitas game Roblox menggunakan model SVM & KNN
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.image("salen.png", width=200)


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
        
        # Confusion matrix
        svm_matrix = joblib.load("svm_confusion_matrix.pkl") if "svm_confusion_matrix.pkl" in os.listdir() else None
        knn_matrix = joblib.load("knn_confusion_matrix.pkl") if "knn_confusion_matrix.pkl" in os.listdir() else None
        
        # Classification report
        svm_report = joblib.load("svm_classification_report.pkl") if "svm_classification_report.pkl" in os.listdir() else None
        knn_report = joblib.load("knn_classification_report.pkl") if "knn_classification_report.pkl" in os.listdir() else None

        return svm, knn, scaler, features, svm_matrix, knn_matrix, svm_report, knn_report
    except Exception as e:
        st.error(f"❌ Gagal load model atau resource: {e}")
        return None, None, None, None, None, None, None, None

svm_model, knn_model, scaler, feature_cols, svm_matrix, knn_matrix, svm_report, knn_report = load_all()

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
# CUSTOM CSS - SIDEBAR WARNA KUNING MUDA
# ==============================================
st.markdown(
    """
    <style>
    /* Sidebar berwarna kuning muda */
    [data-testid="stSidebar"] {
        background-color: #fff9c4;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================================
# SIDEBAR LOGO
# ==============================================
import base64

logo = None
try:
    logo = Image.open("logo_roblox.jpeg")
except:
    pass

if logo is not None:
    with open("logo_roblox.jpeg", "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.sidebar.markdown(
        f"""
        <div style="display:flex; justify-content:center; margin-bottom:15px;">
            <img src="data:image/jpeg;base64,{encoded}" width="170">
        </div>
        """,
        unsafe_allow_html=True
    )

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

        # Mapping label aman
        label_map = {0: "Low", 1: "Medium", 2: "High"}
        svm_label = label_map.get(svm_pred, f" ({svm_pred})")
        knn_label = label_map.get(knn_pred, f" ({knn_pred})")

        st.subheader("🔮 Hasil Prediksi Popularitas Game")
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**SVM:** {svm_label}")
        with col2:
            st.info(f"**KNN:** {knn_label}")

# ==============================================
# VISUALISASI CONFUSION MATRIX
# ==============================================
if svm_matrix is not None or knn_matrix is not None:
    st.header("📊 Confusion Matrix Model")

    def plot_matrix(matrix, title, cmap_color):
        fig, ax = plt.subplots()
        cax = ax.imshow(matrix, cmap=cmap_color, alpha=0.8)
        ax.set_title(title, color="#333333")
        ax.set_xlabel("Predicted", color="#333333")
        ax.set_ylabel("Actual", color="#333333")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, matrix[i, j], ha="center", va="center", color="black", fontsize=12)
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

    colA, colB = st.columns(2)
    with colA:
        if svm_matrix is not None:
            plot_matrix(svm_matrix, "Confusion Matrix - SVM", cmap_color="Greens")
    with colB:
        if knn_matrix is not None:
            plot_matrix(knn_matrix, "Confusion Matrix - KNN", cmap_color="Blues")

# ==============================================
# TAMPILKAN CLASSIFICATION REPORT LENGKAP + Accuracy & Weighted F1
# ==============================================
def display_classification_report_full(report_str, model_name):
    if report_str is None:
        st.warning(f"{model_name} classification report tidak tersedia.")
        return

    st.subheader(f"📈 {model_name} Classification Report")
    
    # Tampilkan seluruh string report
    st.code(report_str, language=None)

    # ==============================
    # Ekstrak Accuracy & Weighted F1
    # ==============================
    acc = None
    weighted_f1 = None

    lines = report_str.split('\n')
    for line in lines:
        if "accuracy" in line.lower():
            # Accuracy biasanya di line 
            parts = line.strip().split()
            if len(parts) >= 2:
                acc = parts[1]
        if "weighted avg" in line.lower():
            # Weighted avg biasanya di line 
            parts = line.strip().split()
            if len(parts) >= 5:
                weighted_f1 = parts[3]  # kolom ke-4 = f1-score

    if acc is not None:
        st.info(f"**Accuracy:** {acc}")
    if weighted_f1 is not None:
        st.info(f"**Weighted F1-Score:** {weighted_f1}")

# Panggil function untuk SVM dan KNN
colA, colB = st.columns(2)
with colA:
    display_classification_report_full(svm_report, "SVM")
with colB:
    display_classification_report_full(knn_report, "KNN")

st.write("---")
st.caption(" © 2025 — Roblox Popularity Deployment ")

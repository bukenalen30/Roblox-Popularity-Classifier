import streamlit as st
import pandas as pd
import joblib

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Roblox Popularity Classifier",
    layout="wide"
)

# ======================================================
# HEADER
# ======================================================
st.markdown("""
<div style="background:linear-gradient(90deg,#a1c4fd,#c2e9fb);
            padding:25px;
            border-radius:15px;
            margin-bottom:25px;">
    <h1 style="text-align:center;color:#0f172a;">
        🎮 Roblox Game Popularity Classifier
    </h1>
    <p style="text-align:center;color:#1e293b;">
        Prediksi tingkat popularitas game Roblox menggunakan SVM dan KNN
    </p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODEL & SCALER
# ======================================================
@st.cache_resource
def load_artifacts():
    svm_model = joblib.load("svm_model.pkl")
    knn_model = joblib.load("knn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return svm_model, knn_model, scaler

svm_model, knn_model, scaler = load_artifacts()

# ======================================================
# FEATURE (HARUS SAMA DENGAN TRAINING)
# ======================================================
activity_features = [
    "Active",
    "Visits",
    "Favourites",
    "Likes",
    "Dislikes"
]

# ======================================================
# SIDEBAR INPUT
# ======================================================
st.sidebar.header("🧩 Input Activity Feature")

active = st.sidebar.number_input("Active Players", min_value=0)
visits = st.sidebar.number_input("Visits", min_value=0)
favourites = st.sidebar.number_input("Favourites", min_value=0)
likes = st.sidebar.number_input("Likes", min_value=0)
dislikes = st.sidebar.number_input("Dislikes", min_value=0)

# ======================================================
# PREDIKSI
# ======================================================
if st.sidebar.button("🚀 Prediksi Popularitas"):

    X_input = pd.DataFrame([[
        active, visits, favourites, likes, dislikes
    ]], columns=activity_features)

    st.subheader("📥 Data Input")
    st.dataframe(X_input, use_container_width=True)

    X_scaled = scaler.transform(X_input)

    svm_pred = svm_model.predict(X_scaled)[0]
    knn_pred = knn_model.predict(X_scaled)[0]

    label_map = {
        0: "Low 🔴",
        1: "Medium 🟡",
        2: "High 🟢"
    }

    st.subheader("🔮 Hasil Prediksi")

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"**SVM Prediction**  
        {label_map[svm_pred]}")

    with col2:
        st.info(f"**KNN Prediction**  
        {label_map[knn_pred]}")

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("✨ Deployment ML — Streamlit | Roblox Popularity Classification")

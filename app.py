import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ============================================================
# CONFIGURATION GLOBALE
# ============================================================
st.set_page_config(
    page_title="Expresso Churn – App ML",
    page_icon="📱",
    layout="wide"
)

MODEL_PATH = "expresso_churn_model.joblib"
DATA_PATH = "Expresso_churn_sample.csv"   # IMPORTANT : fichier présent sur GitHub !


# ============================================================
# CHARGEMENT DU MODÈLE ET DES DONNÉES
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    if "CHURN" in df.columns:
        df["CHURN"] = df["CHURN"].astype(int)
    return df


model = load_model()


# ============================================================
# NAVIGATION
# ============================================================
st.sidebar.title("📱 Expresso Churn App")
st.sidebar.markdown("**RandomForest – 2M de clients (modèle réduit)**")

page = st.sidebar.radio(
    "Navigation",
    ["🔮 Prédiction du churn", "👁️ Vue du dataset", "📊 Dashboard"]
)


# ============================================================
# PAGE 1 — PREDICTION
# ============================================================
if page == "🔮 Prédiction du churn":
    st.title("🔮 Prédiction de churn client")

    st.markdown("""
        Remplis les informations ci-dessous pour estimer la probabilité de **churn** (désabonnement).  
        Le modèle utilisé est un **RandomForest** entraîné sur plus de **2 millions de clients**.
    """)

    col1, col2, col3 = st.columns(3)

    # --- Catégorielles ---
    with col1:
        REGION = st.selectbox("Région", sorted([
            "DAKAR", "SAINT-LOUIS", "THIES", "FATICK", "KAOLACK", "LOUGA",
            "DIOURBEL", "ZIGUINCHOR", "AUTRE"
        ]))

        TENURE = st.selectbox("Ancienneté", [
            "A 1-3 month", "B 3-6 month", "C 6-9 month", "D 9-12 month",
            "E 12-15 month", "F 15-18 month", "G 18-21 month",
            "H 21-24 month", "K > 24 month"
        ])

        MRG = st.selectbox("MRG", ["NO", "YES"])

        TOP_PACK = st.text_input("Pack principal (TOP_PACK)", "On-net 1000F=10MilF,10d")


    # --- Numériques ---
    with col2:
        MONTANT = st.number_input("Montant rechargé", 0.0, value=5000.0)
        FREQUENCE_RECH = st.number_input("Nb recharges", 0.0, value=5.0)
        REVENUE = st.number_input("Revenu généré", 0.0, value=10000.0)
        ARPU_SEGMENT = st.number_input("ARPU Segment", 0.0, value=1500.0)
        FREQUENCE = st.number_input("Fréquence globale", 0.0, value=10.0)

    with col3:
        DATA_VOLUME = st.number_input("Volume data", 0.0, value=2000.0)
        ON_NET = st.number_input("Minutes On-net", 0.0, value=100.0)
        ORANGE = st.number_input("Minutes vers Orange", 0.0, value=20.0)
        TIGO = st.number_input("Minutes vers Tigo", 0.0, value=10.0)
        REGULARITY = st.number_input("Jours actifs", 0, value=15)
        FREQ_TOP_PACK = st.number_input("Usage TOP_PACK", 0.0, value=2.0)

    # Préparation DataFrame
    input_data = pd.DataFrame({
        "REGION": [REGION],
        "TENURE": [TENURE],
        "MRG": [MRG],
        "TOP_PACK": [TOP_PACK],
        "MONTANT": [MONTANT],
        "FREQUENCE_RECH": [FREQUENCE_RECH],
        "REVENUE": [REVENUE],
        "ARPU_SEGMENT": [ARPU_SEGMENT],
        "FREQUENCE": [FREQUENCE],
        "DATA_VOLUME": [DATA_VOLUME],
        "ON_NET": [ON_NET],
        "ORANGE": [ORANGE],
        "TIGO": [TIGO],
        "REGULARITY": [REGULARITY],
        "FREQ_TOP_PACK": [FREQ_TOP_PACK],
    })

    st.markdown("---")

    # PREDICTION
    if st.button("🔍 Lancer la prédiction"):
        proba = float(model.predict_proba(input_data)[0][1])
        pred = int(model.predict(input_data)[0])

        if proba < 0.25:
            couleur = "🟢"
            niveau = "Faible"
        elif proba < 0.55:
            couleur = "🟡"
            niveau = "Modéré"
        else:
            couleur = "🔴"
            niveau = "Élevé"

        if pred == 1:
            st.error(f"{couleur} **Risque de churn ÉLEVÉ : {proba:.2f}**")
        else:
            st.success(f"{couleur} **Client fidèle : {proba:.2f}**")

        st.metric("Probabilité (%)", f"{proba*100:.1f}%")
        st.progress(proba)


# ============================================================
# PAGE 2 — TABLEAU DES DONNÉES
# ============================================================
elif page == "👁️ Vue du dataset":
    st.title("👁️ Exploration du dataset (échantillon 100 000 lignes)")

    df = load_data()
    st.write(df.head(500))


# ============================================================
# PAGE 3 — DASHBOARD
# ============================================================
elif page == "📊 Dashboard":
    st.title("📊 Dashboard analytique – Churn")

    df = load_data()

    col1, col2, col3 = st.columns(3)
    col1.metric("Clients", f"{len(df):,}")
    col2.metric("Taux de churn", f"{df['CHURN'].mean()*100:.1f}%")
    col3.metric(
        "Revenu moyen (churn/non-churn)",
        f"{df[df.CHURN==1].REVENUE.mean():.0f} / {df[df.CHURN==0].REVENUE.mean():.0f}"
    )

    st.markdown("---")

    # Churn par région
    churn_reg = df.groupby("REGION")["CHURN"].mean().sort_values(ascending=False)
    fig1 = px.bar(churn_reg, title="🌍 Churn par région")
    st.plotly_chart(fig1, use_container_width=True)

    # Churn par ancienneté
    churn_ten = df.groupby("TENURE")["CHURN"].mean()
    fig2 = px.bar(churn_ten, title="📆 Churn par ancienneté")
    st.plotly_chart(fig2, use_container_width=True)
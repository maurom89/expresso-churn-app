import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# CONFIGURATION DE L'APP
# ============================================================
st.set_page_config(
    page_title="Expresso Churn – Dashboard & Prédiction",
    page_icon="📱",
    layout="wide"
)

MODEL_PATH = "expresso_churn_model.joblib"
DATA_PATH = "Expresso_churn_sample.csv"  # échantillon allégé

# ============================================================
# FONCTIONS CACHÉES
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_sample():
    df = pd.read_csv(DATA_PATH)
    df["CHURN"] = df["CHURN"].astype(int)
    return df

# Charger modèle + données
model = load_model()
df_sample = load_sample()

# ============================================================
# BARRE LATERALE
# ============================================================
st.sidebar.title("📱 Expresso Churn App")
st.sidebar.markdown("**RandomForest – entraîné sur 2,1 millions de clients**")

page = st.sidebar.radio(
    "Navigation",
    ["🔮 Prédiction du churn", "👀 Vue du dataset", "📊 Dashboard"]
)

# ============================================================
# PAGE 1 – PRÉDICTION
# ============================================================

if page == "🔮 Prédiction du churn":
    st.title("🔮 Prédiction du churn client")
    st.markdown("""
    Remplissez les informations du client ci-dessous pour obtenir  
    la probabilité de désabonnement (**churn**).
    """)

    col1, col2, col3 = st.columns(3)

    # Variables catégorielles
    REGION = col1.selectbox("Région", sorted(df_sample["REGION"].dropna().unique()))
    TENURE = col1.selectbox("Ancienneté (TENURE)", sorted(df_sample["TENURE"].dropna().unique()))
    MRG = col1.selectbox("MRG", ["NO", "YES"])
    TOP_PACK = col1.text_input("Pack principal", "On-net 1000F=10MilF,10d")

    # Numériques
    MONTANT = col2.number_input("Montant rechargé", min_value=0.0, value=5000.0)
    FREQUENCE_RECH = col2.number_input("Nombre de recharges", min_value=0.0, value=5.0)
    DATA_VOLUME = col2.number_input("Volume Data", min_value=0.0, value=2000.0)
    ON_NET = col2.number_input("Minutes On-Net", min_value=0.0, value=100.0)

    REVENUE = col3.number_input("Revenu généré", min_value=0.0, value=10000.0)
    ARPU_SEGMENT = col3.number_input("ARPU Segment", min_value=0.0, value=1500.0)
    FREQUENCE = col3.number_input("Fréquence globale", min_value=0.0, value=10.0)
    ORANGE = col3.number_input("Minutes vers Orange", min_value=0.0, value=20.0)
    TIGO = col3.number_input("Minutes vers Tigo", min_value=0.0, value=10.0)
    REGULARITY = col3.number_input("Jours actifs", min_value=0, value=10)
    FREQ_TOP_PACK = col3.number_input("Utilisation pack", min_value=0.0, value=2.0)

    # DataFrame pour prédiction
    X = pd.DataFrame([{
        "REGION": REGION,
        "TENURE": TENURE,
        "MRG": MRG,
        "TOP_PACK": TOP_PACK,
        "MONTANT": MONTANT,
        "FREQUENCE_RECH": FREQUENCE_RECH,
        "REVENUE": REVENUE,
        "ARPU_SEGMENT": ARPU_SEGMENT,
        "FREQUENCE": FREQUENCE,
        "DATA_VOLUME": DATA_VOLUME,
        "ON_NET": ON_NET,
        "ORANGE": ORANGE,
        "TIGO": TIGO,
        "REGULARITY": REGULARITY,
        "FREQ_TOP_PACK": FREQ_TOP_PACK
    }])

    st.markdown("---")

    if st.button("🔍 Lancer la prédiction"):
        proba = model.predict_proba(X)[0][1]
        pred = model.predict(X)[0]

        if pred == 1:
            st.error(f"🔴 Probabilité de churn : **{proba*100:.2f}%**")
        else:
            st.success(f"🟢 Client fidèle – Probabilité de churn : **{proba*100:.2f}%**")

        st.progress(float(proba))

# ============================================================
# PAGE 2 – VUE DU DATASET
# ============================================================

elif page == "👀 Vue du dataset":
    st.title("👀 Exploration du dataset (échantillon 100 000 lignes)")
    st.dataframe(df_sample.head(500))

    with st.expander("Informations générales"):
        st.write("Nombre total :", len(df_sample))
        st.write("Colonnes :", list(df_sample.columns))
        st.write("Valeurs manquantes :", df_sample.isna().sum())

# ============================================================
# PAGE 3 – DASHBOARD PRO
# ============================================================

elif page == "📊 Dashboard":
    st.title("📊 Dashboard analytique – Churn Expresso")

    # KPIs
    colA, colB, colC = st.columns(3)
    colA.metric("Clients total (sample)", f"{len(df_sample):,}")
    colB.metric("Taux de churn", f"{df_sample['CHURN'].mean()*100:.1f}%")
    colC.metric(
        "Revenu moyen churn / non-churn",
        f"{df_sample[df_sample['CHURN']==1]['REVENUE'].mean():.0f}  /  {df_sample[df_sample['CHURN']==0]['REVENUE'].mean():.0f}"
    )

    st.markdown("---")

    # Churn par région
    churn_reg = df_sample.groupby("REGION")["CHURN"].mean().sort_values(ascending=False)*100
    fig1 = px.bar(churn_reg, title="🌍 Taux de churn par région")
    st.plotly_chart(fig1, use_container_width=True)

    # Churn par ancienneté
    churn_ten = df_sample.groupby("TENURE")["CHURN"].mean()*100
    fig2 = px.bar(churn_ten, title="📆 Churn par ancienneté")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Boxplot Montant
    fig3 = px.box(
        df_sample,
        x="CHURN",
        y="MONTANT",
        points="all",
        title="💰 Montant rechargé selon churn"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Scatter Data vs Revenue
    df_scatter = df_sample.sample(20000, random_state=42)
    fig4 = px.scatter(
        df_scatter,
        x="DATA_VOLUME",
        y="REVENUE",
        color="CHURN",
        opacity=0.5,
        title="📈 Relation Data vs Revenu"
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Corrélations
    numeric_cols = df_sample.select_dtypes(include=[np.number]).columns
    corr = df_sample[numeric_cols].corr()["CHURN"].sort_values(ascending=False)

    fig5 = px.bar(
        corr.head(10),
        title="🔗 Variables les plus corrélées au churn"
    )
    st.plotly_chart(fig5, use_container_width=True)
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
DATA_PATH = "Expresso_churn_sample.csv"


# ============================================================
# CHARGEMENT DES RESSOURCES
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
# SIDEBAR
# ============================================================
st.sidebar.title("📱 Expresso Churn App")
st.sidebar.markdown("**RandomForest – entraîné sur 2,1 millions de clients**")

page = st.sidebar.radio(
    "Navigation",
    ["🔮 Prédiction du churn", "👁️ Vue du dataset", "📊 Dashboard"]
)


# ============================================================
# PAGE 1 – PREDICTION
# ============================================================
if page == "🔮 Prédiction du churn":
    st.title("🔮 Prédiction du churn client")

    st.markdown("Remplissez les informations du client pour estimer la probabilité de **churn**.")

    col1, col2, col3 = st.columns(3)

    # ----------------- variables catégorielles -----------------
    with col1:
        REGION = st.selectbox("RÉGION", sorted([
            "DAKAR", "SAINT-LOUIS", "THIES", "FATICK", "KAOLACK",
            "LOUGA", "DIOURBEL", "ZIGUINCHOR", "AUTRE"
        ]))

        TENURE = st.selectbox("Ancienneté (TENURE)", [
            "A 1-3 month", "B 3-6 month", "C 6-9 month", "D 9-12 month",
            "E 12-15 month", "F 15-18 month", "G 18-21 month",
            "H 21-24 month", "K > 24 month"
        ])

        MRG = st.selectbox("MRG", ["NO", "YES"])

        TOP_PACK = st.text_input("TOP_PACK", "On-net 200F=Unlimited_call24H")

    # ----------------- variables numériques -----------------
    with col2:
        MONTANT = st.number_input("MONTANT", 0.0, step=500.0)
        FREQUENCE_RECH = st.number_input("FREQUENCE_RECH", 0.0, step=1.0)
        REVENUE = st.number_input("REVENUE", 0.0, step=500.0)
        ARPU_SEGMENT = st.number_input("ARPU_SEGMENT", 0.0, step=100.0)
        FREQUENCE = st.number_input("FREQUENCE", 0.0, step=1.0)

    with col3:
        DATA_VOLUME = st.number_input("DATA_VOLUME", 0.0, step=100.0)
        ON_NET = st.number_input("ON_NET", 0.0, step=10.0)
        ORANGE = st.number_input("ORANGE", 0.0, step=10.0)
        TIGO = st.number_input("TIGO", 0.0, step=10.0)
        REGULARITY = st.number_input("REGULARITY", 0, step=1)
        FREQ_TOP_PACK = st.number_input("FREQ_TOP_PACK", 0.0, step=1.0)

    # --- construction du dataframe
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

    if st.button("🔍 Lancer la prédiction"):
        proba = float(model.predict_proba(input_data)[0][1])
        pred = int(model.predict(input_data)[0])

        if proba < 0.25:
            niveau = "Faible"
            couleur = "🟢"
        elif proba < 0.55:
            niveau = "Modéré"
            couleur = "🟡"
        else:
            niveau = "Élevé"
            couleur = "🔴"

        st.subheader("Résultat")
        if pred == 1:
            st.error(f"{couleur} **Risque de churn ÉLEVÉ : {proba:.2f}**")
        else:
            st.success(f"{couleur} **Client fidèle : {proba:.2f}**")

        st.progress(proba)


# ============================================================
# PAGE 2 – VUE DU DATASET
# ============================================================
elif page == "👁️ Vue du dataset":
    st.title("👁️ Exploration du dataset")

    df = load_data()

    st.markdown(f"Échantillon chargé : **{len(df):,} lignes**")

    st.dataframe(df.head(500))

    st.markdown("### Statistiques générales")
    st.write(df.describe(include='all'))


# ============================================================
# PAGE 3 – DASHBOARD PREMIUM
# ============================================================
elif page == "📊 Dashboard":

    st.markdown("""
        <h1 style='text-align:center; color:#1f4e79;'>
            📊 Dashboard Premium – Churn Expresso
        </h1>
    """, unsafe_allow_html=True)

    df = load_data()

    # ================= KPI =================
    k1, k2, k3 = st.columns(3)
    k1.metric("Clients total", f"{len(df):,}")
    k2.metric("Taux de churn", f"{df['CHURN'].mean()*100:.1f} %")
    k3.metric("Revenu moyen churn", f"{df[df['CHURN']==1]['REVENUE'].mean():.0f}")

    st.markdown("---")

    # =============== Churn par région ===============
    st.subheader("🌍 Churn par région")

    churn_region = (
        df.groupby("REGION")["CHURN"].mean().reset_index()
    )
    churn_region["CHURN_PCT"] = churn_region["CHURN"] * 100

    fig1 = px.bar(
        churn_region,
        x="REGION", y="CHURN_PCT",
        color="CHURN_PCT",
        color_continuous_scale="Reds",
        title="Taux de churn par région"
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # =============== Churn par ancienneté (TENURE) ===============
    st.subheader("⏳ Churn par ancienneté")

    ordre = [
        "A 1-3 month", "B 3-6 month", "C 6-9 month",
        "D 9-12 month", "E 12-15 month", "F 15-18 month",
        "G 18-21 month", "H 21-24 month", "K > 24 month"
    ]
    df["TENURE"] = pd.Categorical(df["TENURE"], categories=ordre, ordered=True)

    churn_ten = df.groupby("TENURE")["CHURN"].mean().reset_index()
    churn_ten["CHURN_PCT"] = churn_ten["CHURN"] * 100

    fig2 = px.line(
        churn_ten, x="TENURE", y="CHURN_PCT",
        markers=True,
        title="Churn selon l’ancienneté",
    )
    fig2.update_traces(line=dict(color="#1f4e79", width=4))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # =============== Variables importantes ===============
    st.subheader("📈 Variables les plus liées au churn")

    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()["CHURN"].sort_values(ascending=False).drop("CHURN").head(10)

    fig3 = px.bar(
        x=corr.values, y=corr.index,
        orientation="h",
        color=corr.values,
        color_continuous_scale="Blues",
        title="Top 10 variables corrélées au churn"
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.success("🎉 Dashboard premium affiché avec succès !")
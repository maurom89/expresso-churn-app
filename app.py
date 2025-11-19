import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ----------------------------------------------
# 1. CONFIG GLOBALE DE LA PAGE
# ----------------------------------------------
st.set_page_config(
    page_title="Expresso Churn – App ML",
    page_icon="📱",
    layout="wide"
)

st.sidebar.title("📱 Expresso Churn App")
st.sidebar.markdown("**RandomForest – 2M de clients**")
page = st.sidebar.radio(
    "Navigation",
    ["🧠 Prédiction churn", "👁️ Vue du dataset", "📊 Dashboard"]
)

MODEL_PATH = "expresso_churn_model.joblib"
DATA_PATH = "Expresso_churn_dataset.csv"


# ----------------------------------------------
# 2. FONCTIONS UTILITAIRES
# ----------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data_sample(n_rows: int = 200_000):
    """Charge un échantillon pour l'affichage / dashboard (plus rapide)."""
    df = pd.read_csv(DATA_PATH)
    if len(df) > n_rows:
        df = df.sample(n=n_rows, random_state=42)
    return df

# Charger le modèle une fois
model = load_model()

# ----------------------------------------------
# 3. PAGE : PRÉDICTION CHURN
# ----------------------------------------------
if page == "🧠 Prédiction churn":
    st.title("🧠 Prédiction de churn client")

    st.markdown(
        """
        Remplis les informations du client ci-dessous pour estimer la probabilité de **churn** (désabonnement).
        Le modèle utilisé est un **RandomForest** entraîné sur plus de **2 millions de clients Expresso**.
        """
    )

    col1, col2, col3 = st.columns(3)

    # --- Variables catégorielles ---
    with col1:
        REGION = st.selectbox(
            "Région (REGION)",
            ["DAKAR", "SAINT-LOUIS", "THIES", "FATICK", "KAOLACK", "LOUGA",
             "DIOURBEL", "ZIGUINCHOR", "AUTRE"]
        )

        TENURE = st.selectbox(
            "Ancienneté (TENURE)",
            [
                "A 1-3 month", "B 3-6 month", "C 6-9 month", "D 9-12 month",
                "E 12-15 month", "F 15-18 month", "G 18-21 month",
                "H 21-24 month", "K > 24 month"
            ]
        )

        MRG = st.selectbox("MRG", ["NO", "YES"])

        TOP_PACK = st.text_input("TOP_PACK (nom du pack)", value="On-net 1000F=10MilF,10d")

    # --- Variables numériques (col2 & col3) ---
    with col2:
        MONTANT = st.number_input("MONTANT (montant rechargé)", min_value=0.0, step=500.0, value=5000.0)
        FREQUENCE_RECH = st.number_input("FREQUENCE_RECH (nb de recharges)", min_value=0.0, step=1.0, value=5.0)
        REVENUE = st.number_input("REVENUE (revenu généré)", min_value=0.0, step=500.0, value=10000.0)
        ARPU_SEGMENT = st.number_input("ARPU_SEGMENT", min_value=0.0, step=100.0, value=1500.0)
        FREQUENCE = st.number_input("FREQUENCE (activité globale)", min_value=0.0, step=1.0, value=10.0)

    with col3:
        DATA_VOLUME = st.number_input("DATA_VOLUME", min_value=0.0, step=100.0, value=2000.0)
        ON_NET = st.number_input("ON_NET (minutes on-net)", min_value=0.0, step=10.0, value=100.0)
        ORANGE = st.number_input("ORANGE (minutes vers Orange)", min_value=0.0, step=10.0, value=20.0)
        TIGO = st.number_input("TIGO (minutes vers Tigo)", min_value=0.0, step=10.0, value=10.0)
        REGULARITY = st.number_input("REGULARITY (jours actifs)", min_value=0, step=1, value=15)
        FREQ_TOP_PACK = st.number_input("FREQ_TOP_PACK", min_value=0.0, step=1.0, value=2.0)

    # Construction du DataFrame dans le même format que le training
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
    if st.button("🔮 Prédire le churn"):
        proba = model.predict_proba(input_data)[0][1]
        pred = model.predict(input_data)[0]

        col_res1, col_res2 = st.columns([2, 1])

        with col_res1:
            if pred == 1:
                st.error(f"⚠️ Ce client est **à risque de churn**.\n\nProbabilité estimée : **{proba:.2f}**")
            else:
                st.success(f"✅ Ce client est **probablement loyal**.\n\nProbabilité de churn : **{proba:.2f}**")

        with col_res2:
            st.write("Probabilité de churn")
            st.progress(float(proba))

        st.markdown("ℹ️ **Interprétation rapide** :")
        st.markdown(
            "- Proba proche de **1.0** ⇒ fort risque de désabonnement.\n"
            "- Proba proche de **0.0** ⇒ client plutôt fidèle.\n"
            "- Tu peux comparer plusieurs profils en changeant les paramètres."
        )


# ----------------------------------------------
# 4. PAGE : VUE DU DATASET
# ----------------------------------------------
elif page == "👁️ Vue du dataset":
    st.title("👁️ Vue du dataset Expresso Churn")

    st.markdown(
        """
        Voici un aperçu du dataset utilisé pour entraîner le modèle.
        Pour des raisons de performance, on affiche un **échantillon**.
        """
    )

    df = load_data_sample(n_rows=100_000)

    st.write(f"Nombre de lignes affichées : `{len(df)}` (sur 2 154 848 au total)")
    st.dataframe(df.head(200))

    with st.expander("Voir informations sur le dataset"):
        st.write("**Colonnes :**")
        st.write(df.columns.tolist())
        st.write("**Valeurs manquantes (top 15) :**")
        st.write(df.isna().sum().sort_values(ascending=False).head(15))


# ----------------------------------------------
# 5. PAGE : DASHBOARD
# ----------------------------------------------
elif page == "📊 Dashboard":
    st.title("📊 Dashboard churn – Expresso")

    df = load_data_sample(n_rows=200_000)

    col_top, col_filter = st.columns([3, 1])
    with col_filter:
        st.markdown("### Filtres")

        region_filter = st.selectbox(
            "Région",
            ["Toutes"] + sorted(df["REGION"].dropna().unique().tolist())
        )

        tenure_filter = st.selectbox(
            "Ancienneté (TENURE)",
            ["Toutes"] + sorted(df["TENURE"].dropna().unique().tolist())
        )

        tmp = df.copy()
        if region_filter != "Toutes":
            tmp = tmp[tmp["REGION"] == region_filter]
        if tenure_filter != "Toutes":
            tmp = tmp[tmp["TENURE"] == tenure_filter]

    # KPI en haut
    with col_top:
        total_clients = len(tmp)
        churn_rate = tmp["CHURN"].mean() if total_clients > 0 else 0
        nb_churn = tmp["CHURN"].sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Clients (échantillon)", f"{total_clients:,}")
        c2.metric("Taux de churn", f"{churn_rate*100:,.1f}%")
        c3.metric("Nombre de churn", f"{int(nb_churn):,}")

    st.markdown("---")

    # Graphiques
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("Répartition churn vs non churn")
        churn_counts = tmp["CHURN"].value_counts().rename(index={0: "Non churn", 1: "Churn"})
        fig = px.pie(
            values=churn_counts.values,
            names=churn_counts.index,
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_g2:
        st.subheader("Churn par région")
        churn_by_region = tmp.groupby("REGION")["CHURN"].mean().reset_index()
        churn_by_region["CHURN"] = churn_by_region["CHURN"] * 100
        fig2 = px.bar(
            churn_by_region.sort_values("CHURN", ascending=False),
            x="REGION",
            y="CHURN",
            labels={"CHURN": "Taux de churn (%)"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.subheader("ARPU_SEGMENT vs churn")
    fig3 = px.box(
        tmp,
        x="CHURN",
        y="ARPU_SEGMENT",
        points="suspectedoutliers",
        labels={"CHURN": "Churn (0=non, 1=oui)", "ARPU_SEGMENT": "ARPU_SEGMENT"}
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        """
        ℹ️ **Idées d'analyse :**
        - Comparer les régions avec fort churn.
        - Regarder l'impact de l'ancienneté (TENURE) sur le churn.
        - Croiser ARPU_SEGMENT / DATA_VOLUME / FREQUENCE avec le churn.
        """
    )
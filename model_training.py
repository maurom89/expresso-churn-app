import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# ----------------------------------------------
# 1. CHARGEMENT DU DATASET (2,1 millions)
# ----------------------------------------------

df = pd.read_csv("Expresso_churn_dataset.csv")

print("Shape du dataset complet :", df.shape)
print(df.head())

# ----------------------------------------------
# 2. SUPPRESSION DES COLONNES INUTILES
# ----------------------------------------------

cols_to_drop = ["user_id", "ZONE1", "ZONE2"]
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# ----------------------------------------------
# 3. SÉPARATION FEATURES / TARGET
# ----------------------------------------------

target = "CHURN"
y = df[target]
X = df.drop(columns=[target])

# ----------------------------------------------
# 4. IDENTIFICATION DES COLONNES
# ----------------------------------------------

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

print("Numériques :", numeric_features)
print("Catégorielles :", categorical_features)

# ----------------------------------------------
# 5. PREPROCESSING
# ----------------------------------------------

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ----------------------------------------------
# 6. TRAIN/TEST SPLIT
# ----------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------------------------
# 7. RANDOM FOREST OPTIMISÉ (pour 2M de lignes)
# ----------------------------------------------

clf = RandomForestClassifier(
    n_estimators=30,   # ← réduit de 200 à 30 (beaucoup plus rapide)
    max_depth=10,     # ← contrôle de la complexité
    n_jobs=-1,        # ← utiliser tous les coeurs du CPU
    random_state=42
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", clf)
])

print("\nEntraînement du modèle sur 2 millions de lignes...")
model.fit(X_train, y_train)

# ----------------------------------------------
# 8. EVALUATION
# ----------------------------------------------

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nROC AUC :", roc_auc_score(y_test, y_proba))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------------------------
# 9. SAUVEGARDE DU MODELE
# ----------------------------------------------

joblib.dump(model, "expresso_churn_model.joblib")
print("\nModèle enregistré sous : expresso_churn_model.joblib")
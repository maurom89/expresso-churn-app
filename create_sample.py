import pandas as pd

# 1) Charger le gros fichier (2 millions de lignes)
df = pd.read_csv("Expresso_churn_dataset.csv")

print("Shape complet :", df.shape)

# 2) Prendre un échantillon de 100 000 lignes
sample = df.sample(n=100_000, random_state=42)

print("Shape échantillon :", sample.shape)

# 3) Sauvegarder l'échantillon dans un nouveau fichier
sample.to_csv("Expresso_churn_sample.csv", index=False)

print("Fichier 'Expresso_churn_sample.csv' créé avec succès !")
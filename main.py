import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('raw_data.csv')
# Supprime les 3 premières colonnes ["Horodateur", "Nom d'utilisateur", "Mentions légales"]
data = data.drop(columns=data.columns[:3])
# Supprime les dernières colonnes qui concerne les avis sur le futur developpement de l'application
data = data.drop(columns=data.columns[20:])

def data_clean(data):
    text_cols = data.select_dtypes(include=['object']).columns
    for col in text_cols:
        # Remplace "mot / mot" par "mot;mot" dans toutes les colonnes texte
        data[col] = data[col].str.replace(r'\s*/\s*', ';', regex=True)
        data[col] = data[col].str.strip()

    data = data.fillna("null")
    data = data.replace("blank", "null")

    mapping = {
        1: 60,
        2: 90,
        3: 120,
        4: 150,
        5: 180
    }

    data.iloc[:, 8] = data.iloc[:, 8].replace(mapping)

    data.to_csv('main.csv', index=False)
    print("main.csv successfully created ✅")
    return data

data_clean(data)
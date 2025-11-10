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
    femme, homme = 0, 0
    # Renommage des colonnes
    new_cols = ["musique intrumentale ou avec parole", "support écouté", "station radio", "genre musicale", "période musicale", "langue écouté", "type de parole", "frequence ecouté artiste émergent", "tempo", "fréquence en travaillant", "fréquence en sport", "fréquence en cuisine", "fréquence en transport", "fréquence en passant le temps", "fréquence écoute mensuel", "fréquence écoute journalier", "genre", "age", "environnement", "situation professionnel"]
    data.columns = new_cols
    text_cols = data.select_dtypes(include=['object']).columns
    
    for col in text_cols:
        # Remplace "mot / mot" par "mot;mot" dans toutes les colonnes texte
        data[col] = data[col].str.replace(r'\s*/\s*', ';', regex=True)
        data[col] = data[col].str.replace(r'\s*,\s*', ';', regex=True)
        data[col] = data[col].str.strip()
        data[col] = data[col].str.replace(r'\s* ; \s*', ';', regex=True)
        data[col] = data[col].str.replace(r'\s* \( \s*', '(', regex=True)
        # data[col] = data[col].str.replace(r'\s* \s*', '_', regex=True)
        # data[col] = data[col].str.replace("aucune", "null")
        
        # Vérification des valeurs dans la colonne "genre"
        if col == "genre":
            for entry in data[col]:
                if (entry != "Femme") and (entry != "Homme") and (entry.lower() != "non binaire") and (entry!= "Je préfère ne pas répondre"):
                    new_entry = "Je préfère ne pas répondre"
                    data[col] = data[col].replace(entry, new_entry)

            data[col].str.lower()

    # data = data.fillna("null")
    # data = data.replace("blank", "null")

    data = data.map(lambda x: x.lower() if isinstance(x, str) else x)

    mapping_bpm = {
        1: 60,
        2: 90,
        3: 120,
        4: 150,
        5: 180
    }

    data.iloc[:, 8] = data.iloc[:, 8].replace(mapping_bpm)

    mapping_freq_mensuelle = {
        "plus d'une fois par jour": 1,
        "plus d'une fois par semaine": 2,
        "plus d'une fois par mois": 3,
        "moins d'une fois par mois": 4,
    }

    data.iloc[:, 14] = data.iloc[:, 14].replace(mapping_freq_mensuelle)

    mapping_freq_jour = {
        "plus de trois heures par jour": 1,
        "plus d'une heure par jour": 2,
        "moins d'une heure par jour": 3,
    }

    data.iloc[:, 15] = data.iloc[:, 15].replace(mapping_freq_jour)

    data.to_csv('cleaned_data.csv', index=False)
    print("cleaned_data.csv successfully created ✅")
    return data

data_clean(data)
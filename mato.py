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
    # Renommage des colonnes
    new_cols = ["musique intrumentale ou avec parole", "support écouté", "station radio", "genre musicale", "période musicale", "langue écouté", "type de parole", "frequence ecouté artiste émergent", "tempo", "fréquence en travaillant", "fréquence en sport", "fréquence en cuisine", "fréquence en transport", "fréquence en passant le temps", "fréquence écoute mensuel", "fréquence écoute journalier", "genre", "age", "environnement", "situation professionnel"]
    data.columns = new_cols
    text_cols = data.select_dtypes(include=['object']).columns
    
    for col in text_cols:
        # Remplace "mot / mot" par "mot;mot" dans toutes les colonnes texte
        data[col] = data[col].str.replace(r'\s*/\s*', ';', regex=True)
        data[col] = data[col].str.replace(r'\s*,\s*', ',', regex=True)
        data[col] = data[col].str.strip() 
        
    data.to_csv('cleaned_data.csv', index=False)
    return data

data_clean(data)
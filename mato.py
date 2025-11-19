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
    # Renommage des colonnes en anglais et simplification des noms
    new_cols = ["instrumental or vocal music", "platform listening", "radio station", "music style", "musical period", "language listening", "type of singing", "frequency listened to emerging artist", "tempo", "frequency when working", "frequency during exercise", "frequency while cooking", "frequency while driving", "frequency when passing the time", "monthly listening frequency", "daily listening frequency", "gender", "age", "environment", "professional situation"]
    data.columns = new_cols
    text_cols = data.select_dtypes(include=['object']).columns
    
    for col in text_cols:
        # Remplace "mot / mot" par "mot;mot" dans toutes les colonnes texte
        data[col] = data[col].str.replace(r'\s*/\s*', ';', regex=True)
        data[col] = data[col].str.replace(r'\s*,\s*', ';', regex=True)
        data[col] = data[col].str.strip()
        data[col] = data[col].str.replace(r'\s* ; \s*', ';', regex=True)
        data[col] = data[col].str.replace(r'\s* \( \s*', '(', regex=True)
        data[col] = data[col].str.replace(r'\s* \) \s*', ')', regex=True)
        
        # traduction en anglais
        series = data[col].astype(str).str.strip().str.lower()

        if col == "instrumental or vocal music":
            mapping = {
                "musique instrumentale": "instrumental music",
                "musique vocale": "vocal music",
                "les deux": "both"
            }
            data[col] = series.replace(mapping)

        elif col == "platform listening":
            mapping = {
                "plateforme de streaming": "streaming platform",
                "morceaux locaux": "local music",
                "vinyle": "vinyl"
            }
            data[col] = series.replace(mapping)

        if col == "type of singing":
            mapping = {
                "peu importe": "doesn't matter",
                "engagées": "engaged",
                "poétiques": "poetic",
                "humoristiques": "humorous",
                "peut importe": "doesn't matter"
                
            }
            data[col] = series.map(lambda x: mapping.get(x, "doesn't matter"))

        elif col == "musical period":
            mapping = {
                "années 70 - 90": "1970s-1990s",
                "années 90": "1990s",
                "années 2000": "2000s",
                "années 2010": "2010s",
                "années 2020": "2020s",
                "je ne sais pas": "i don't know",
                "pas de préférence": "no preference"
            }
            data[col] = series.replace(mapping)

        if col == "gender":
            mapping = {
                "femme": "woman",
                "homme": "man",
                "non binaire": "non-binary"
            }
            data[col] = series.map(lambda x: mapping.get(x, "prefer not to answer"))

        if col == "environment":
            mapping = {
                "banlieue": "suburb",
                "ville": "city",
                "campagne": "countryside"
            }
            data[col] = series.map(lambda x: mapping.get(x, "prefer not to answer"))
        
        if col == "professional situation":
            mapping = {
                "sans emploi": "unemployed",
                "étudiant": "student",
                "salarié": "employee",
                "indépendant": "self-employed",
                "retraité": "retired",
                "Autre / Je ne souhaite pas répondre": "prefer not to answer"
            }
            data[col] = series.map(lambda x: mapping.get(x, "prefer not to answer"))
            
    # lowercase pour toutes les colonnes texte (après les mappings)
    text_cols = data.select_dtypes(include=['object']).columns
    data[text_cols] = data[text_cols].apply(lambda s: s.str.lower())

    # AGE -> numérique
    data["age"] = pd.to_numeric(data["age"], errors="coerce")


    # Mapping tempo : valeurs 1–5 → BPM
    mapping_bpm = {
        1: 60,
        2: 90,
        3: 120,
        4: 150,
        5: 180
    }

    data.iloc[:, 8] = data.iloc[:, 8].replace(mapping_bpm)
    
    mapping_freq_mensuelle = {
        "plus d'une fois par jour": 4,
        "plus d'une fois par semaine": 3,
        "plus d'une fois par mois": 2,
        "moins d'une fois par mois": 1,
    }

    data.iloc[:, 14] = data.iloc[:, 14].replace(mapping_freq_mensuelle)

    mapping_freq_jour = {
        "plus de trois heures par jour": 3,
        "plus d'une heure par jour": 2,
        "moins d'une heure par jour": 1,
    }

    data.iloc[:, 15] = data.iloc[:, 15].replace(mapping_freq_jour)

    data.to_csv('cleaned_data.csv', index=False)
    print("cleaned_data.csv successfully created ✅")
    print(f"Mean age = {data['age'].mean()}")
    print(f"Mean tempo = {data['tempo'].mean()}")

    return data

data_clean(data)

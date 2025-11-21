import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Chargement des données brutes
data = pd.read_csv('raw_data.csv')

# Supprime les 3 premières colonnes ["Horodateur", "Nom d'utilisateur", "Mentions légales"]
data = data.drop(columns=data.columns[:3])

# Supprime les colonnes concernant les avis (après la 20ᵉ)
data = data.drop(columns=data.columns[20:])

def data_clean(data):

    # ======================================================
    # 1) RENOMMAGE DES COLONNES
    # ======================================================
    new_cols = [
        "musique intrumentale ou avec parole",
        "support écouté",
        "station radio",
        "genre musicale",
        "période musicale",
        "langue écouté",
        "type de parole",
        "frequence ecouté artiste émergent",
        "tempo",
        "fréquence en travaillant",
        "fréquence en sport",
        "fréquence en cuisine",
        "fréquence en transport",
        "fréquence en passant le temps",
        "fréquence écoute mensuel",
        "fréquence écoute journalier",
        "genre",
        "age",
        "environnement",
        "situation professionnel"
    ]
    data.columns = new_cols

    # ======================================================
    # 2) NETTOYAGE GLOBAL DES COLONNES TEXTE
    # ======================================================
    text_cols = data.select_dtypes(include=['object']).columns
    for col in text_cols:

        data[col] = data[col].str.replace(r'\s*/\s*', ';', regex=True)
        data[col] = data[col].str.replace(r'\s*,\s*', ';', regex=True)
        data[col] = data[col].str.strip()
        data[col] = data[col].str.replace(r'\s*;\s*', ';', regex=True)
        data[col] = data[col].str.replace(r'\s*\(\s*', '(', regex=True)

        # Correction du genre (F/H/NB)
        if col == "genre":
            valid = {"femme", "homme", "non binaire", "je préfère ne pas répondre"}
            data[col] = data[col].str.strip().str.lower()
            mask_invalid = ~data[col].isin(valid) & data[col].notna()
            data.loc[mask_invalid, col] = "je préfère ne pas répondre"

    # Mettre toutes les colonnes texte en minuscules
    for col in text_cols:
        data[col] = data[col].str.lower()

    # ======================================================
    # 3) NORMALISATION DES GENRES “TECHNO”
    # ======================================================
    techno_variants = [
        "techno", "tekno", "teknò", "tecno", "hardtechno", "hardtech",
        "hardteck", "uptempo", "hardstyle", "hard style", "rawstyle",
        "industrial techno", "des gros kicks sa mere", "hxc"
    ]

    def simplify_genre(genre):
        if not isinstance(genre, str):
            return np.nan
        genre = genre.strip().replace(",", ";").replace("/", ";")
        parts = [g.strip() for g in genre.split(";")]
        new_parts = []
        for g in parts:
            if any(t in g for t in techno_variants):
                new_parts.append("techno")
            else:
                new_parts.append(g)
        # nettoyer doublons + vides
        new_parts = [g for g in new_parts if g not in ["", None]]
        new_parts = list(dict.fromkeys(new_parts))
        return ";".join(new_parts) if new_parts else np.nan

    data["genre musicale"] = data["genre musicale"].apply(simplify_genre)

    # ======================================================
    # 4) SUPPRESSION PARTIELLE DES GENRES ABSURDES
    # ======================================================
    motifs_a_supprimer = [
        "je ne peux pas me satisfaire",
        "je n'écoute que rarement",
        "rap anglais des années 90",
        "musique de dépression",
        "sokuuu",
        "années",
        "60", "70", "80", "90", "2000", 
        "etc)",
        "2015",
        "avec des textes"
    ]

    escaped = [re.escape(m) for m in motifs_a_supprimer]
    pattern = "(" + "|".join(escaped) + ")"

    def clean_absurd_genres(value):
        if not isinstance(value, str):
            return np.nan
        genres = [g.strip() for g in value.split(";")]
        cleaned = []
        for g in genres:
            if re.search(pattern, g, flags=re.IGNORECASE):
                continue
            cleaned.append(g)
        if len(cleaned) == 0:
            return np.nan
        return ";".join(cleaned)

    data["genre musicale"] = data["genre musicale"].apply(clean_absurd_genres)
    data["genre musicale"] = data["genre musicale"].replace(r'^\s*$', np.nan, regex=True)

    # ======================================================
    # 5) FUSION / NORMALISATION DES GENRES
    # ======================================================
    def normalize_genre(g):
        if not isinstance(g, str):
            return None
        g = g.lower().strip()

        # Pop regroupée
        if g in ["k-pop", "j-pop", "dream pop"]:
            return "pop"

        # Variété française
        if g in ["variété française", "variete française", "chanson française à texte", "chanson francaise"]:
            return "french variety"

        # Métal / Metal
        if "metal" in g or "métal" in g:
            return "metal"

        # Rock & dérivés
        if g in [
            "hard rock", "rock prog", "indie rock", "alt rock",
            "rock progressif", "rock'n'roll", "rock n roll", "rock and roll", "alternative", "gothique", "indie", "musique alternative"
        ] or g.startswith("rock"):
            return "rock"

        # Electro variations
        if g in ["electro chill et populaires", "electro populaire", "electrique", "drum and bass", "breakcore", "dubstep", "chiptune", "dance", "vocaloid", "house"]:
            return "electro"
        
        # rap
        if g in ["r&b", "hip-hop"]:
            return "rap"
        
        # jazz
        if g in ["soul", "blues"]:
            return "jazz"

        # folk
        if g in ["musique du monde", "reggae", "celtique", "shatta"]:
            return "folk"
        
        # ost
        if g in ["musique de jeux"]:
            return "ost"
        
        #eclectique
        if g in ["indépendant divers", "éclectique"]:
            return "eclectic"

        # Aucun → pas de préférences
        if g in ["aucun préféré", "aucun", "aucun preference", "aucune idée", "pas de préférences"]:
            return "no preferences"

        # Musique classique en anglais        
        if g in ["musique classique"]:
            return "classical music"

        return g

    def apply_genre_normalization(value):
        if not isinstance(value, str):
            return np.nan

        genres = [g.strip() for g in value.split(";")]

        normalized = []
        for g in genres:
            ng = normalize_genre(g)
            if ng:
                normalized.append(ng)

        normalized = list(dict.fromkeys(normalized))
        return ";".join(normalized) if normalized else np.nan

    data["genre musicale"] = data["genre musicale"].apply(apply_genre_normalization)

    # ======================================================
    # 6) MAPPINGS & CONVERSIONS
    # ======================================================
    mapping_bpm = {1: 60, 2: 90, 3: 120, 4: 150, 5: 180}
    data["tempo"] = data["tempo"].replace(mapping_bpm)

    mapping_freq_mensuelle = {
        "plus d'une fois par jour": 1,
        "plus d'une fois par semaine": 2,
        "plus d'une fois par mois": 3,
        "moins d'une fois par mois": 4,
    }
    data["fréquence écoute mensuel"] = data["fréquence écoute mensuel"].replace(mapping_freq_mensuelle)

    mapping_freq_jour = {
        "plus de trois heures par jour": 1,
        "plus d'une heure par jour": 2,
        "moins d'une heure par jour": 3,
    }
    data["fréquence écoute journalier"] = data["fréquence écoute journalier"].replace(mapping_freq_jour)

    data["age"] = pd.to_numeric(data["age"], errors="coerce")
    data["age"] = data["age"].fillna(data["age"].mean())

    # ======================================================
    # 7) EXPORT
    # ======================================================
    data.to_csv('cleaned_data.csv', index=False)
    print("cleaned_data.csv successfully created with improvements ✅")

    return data


# Lancement
cleaned = data_clean(data)

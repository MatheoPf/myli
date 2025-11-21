import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------------
# CHARGEMENT DES DONNÉES
# -------------------------------------------------------------------------
data = pd.read_csv('raw_data.csv')

# Supprime les 3 premières colonnes ["Horodateur", "Nom d'utilisateur", "Mentions légales"]
data = data.drop(columns=data.columns[:3])

# Supprime les colonnes concernant les avis (après la 20ᵉ)
data = data.drop(columns=data.columns[20:])


# -------------------------------------------------------------------------
# FONCTION PRINCIPALE DE NETTOYAGE COMPLET
# -------------------------------------------------------------------------
def data_clean(data):

    # -------------------------------------------------------------
    # Renommage des colonnes (base = version la plus complète)
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # Nettoyage global des colonnes texte
    # -------------------------------------------------------------
    text_cols = data.select_dtypes(include=['object']).columns

    for col in text_cols:
        data[col] = (
            data[col]
            .astype(str)
            .str.replace(r'\s*/\s*', ';', regex=True)
            .str.replace(r'\s*,\s*', ';', regex=True)
            .str.replace(r'\s*;\s*', ';', regex=True)
            .str.strip()
            .str.lower()
        )

        # ---------------------------------------------------------
        # Traductions utiles depuis mato.py
        # ---------------------------------------------------------
        # Instrumental / vocal
        if col == "musique intrumentale ou avec parole":
            mapping = {
                "musique instrumentale": "instrumentale",
                "musique vocale": "vocale",
                "les deux": "les deux"
            }
            data[col] = data[col].replace(mapping)

        # Support écouté
        if col == "support écouté":
            mapping = {
                "plateforme de streaming": "streaming",
                "morceaux locaux": "local",
                "vinyle": "vinyle"
            }
            data[col] = data[col].replace(mapping)

        # Type de parole
        if col == "type de parole":
            mapping = {
                "peu importe": "peu importe",
                "engagées": "engagées",
                "poétiques": "poétiques",
                "humoristiques": "humoristiques",
                "peut importe": "peu importe"
            }
            data[col] = data[col].replace(mapping)

        # Période musicale
        if col == "période musicale":
            mapping = {
                "années 70 - 90": "1970-1990",
                "années 90": "1990",
                "années 2000": "2000",
                "années 2010": "2010",
                "années 2020": "2020",
                "je ne sais pas": "ne sait pas",
                "pas de préférence": "aucune préférence"
            }
            data[col] = data[col].replace(mapping)

        # Genre (F/H/NB)
        if col == "genre":
            mapping = {
                "femme": "femme",
                "homme": "homme",
                "non binaire": "non binaire"
            }
            data[col] = data[col].map(lambda x: mapping.get(x, "je préfère ne pas répondre"))

        # Environnement
        if col == "environnement":
            mapping = {
                "banlieue": "banlieue",
                "ville": "ville",
                "campagne": "campagne"
            }
            data[col] = data[col].map(lambda x: mapping.get(x, "je préfère ne pas répondre"))

        # Situation professionnelle
        if col == "situation professionnel":
            mapping = {
                "sans emploi": "sans emploi",
                "étudiant": "étudiant",
                "salarié": "salarié",
                "indépendant": "indépendant",
                "retraité": "retraité",
                "autre / je ne souhaite pas répondre": "je préfère ne pas répondre"
            }
            data[col] = data[col].map(lambda x: mapping.get(x, "je préfère ne pas répondre"))

    # ======================================================
    # SUPPRESSION PARTIELLE DES GENRES ABSURDES
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
    # FUSION / NORMALISATION DES GENRES
    # ======================================================
    def normalize_genre(g):
        if not isinstance(g, str):
            return None
        g = g.lower().strip()

        # pop
        if g in ["k-pop", "j-pop", "dream pop"]:
            return "pop"

        # variété française
        if g in ["variété française", "variete française", "chanson française à texte", "chanson francaise"]:
            return "french variety"

        # metal
        if "metal" in g or "métal" in g:
            return "metal"

        # rock
        if g in [
            "hard rock", "rock prog", "indie rock", "alt rock",
            "rock progressif", "rock'n'roll", "rock n roll", "rock and roll",
            "alternative", "gothique", "indie", "musique alternative"
        ] or g.startswith("rock"):
            return "rock"

        # electro
        if g in [
            "electro chill et populaires", "electro populaire", "electrique",
            "drum and bass", "breakcore", "dubstep", "chiptune", "dance",
            "vocaloid", "house"
        ]:
            return "electro"
        
        # techno
        if g in [
            "techno", "tekno", "teknò", "tecno", "hardtechno", "hardtech",
            "hardteck", "uptempo", "hardstyle", "hard style", "rawstyle",
            "industrial techno", "des gros kicks sa mere", "hxc"
        ]:
            return "techno"
        
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
        
        # éclectique
        if g in ["indépendant divers", "éclectique"]:
            return "eclectic"

        # pas de préférences
        if g in ["aucun préféré", "aucun", "aucun preference", "aucune idée", "pas de préférences"]:
            return "no preferences"

        # musique classique
        if g == "musique classique":
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


    # -------------------------------------------------------------
    # Mappings numériques
    # -------------------------------------------------------------
    tempo_map = {1: 60, 2: 90, 3: 120, 4: 150, 5: 180}
    data["tempo"] = data["tempo"].replace(tempo_map)

    freq_mensuelle = {
        "plus d'une fois par jour": 1,
        "plus d'une fois par semaine": 2,
        "plus d'une fois par mois": 3,
        "moins d'une fois par mois": 4,
    }
    data["fréquence écoute mensuel"] = data["fréquence écoute mensuel"].replace(freq_mensuelle)

    freq_jour = {
        "plus de trois heures par jour": 1,
        "plus d'une heure par jour": 2,
        "moins d'une heure par jour": 3,
    }
    data["fréquence écoute journalier"] = data["fréquence écoute journalier"].replace(freq_jour)

    # -------------------------------------------------------------
    # AGE → numérique + imputation moyenne
    # -------------------------------------------------------------
    data["age"] = pd.to_numeric(data["age"], errors="coerce")
    data["age"] = data["age"].fillna(data["age"].mean())

    # -------------------------------------------------------------
    # EXPORT
    # -------------------------------------------------------------
    data.to_csv('cleaned_data.csv', index=False)
    print("cleaned_data.csv created successfully ✅")

    return data

cleaned = data_clean(data)

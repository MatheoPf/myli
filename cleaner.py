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
# Supprime les colonnes concernant les avis (après la 20ᵉ)
data = data.drop(columns=data.columns[20:])


def data_clean(data):
    # -------------------------------------------------------------
    # Renommage des colonnes en anglais et simplification des noms
    # -------------------------------------------------------------
    new_cols = ["instrumental or vocal music", "platform listening", "radio station", "music style", "musical period", "language listening", "type of singing", "frequency listening of emerging artist", "tempo", "frequency during working", "frequency during exercising", "frequency during cooking", "frequency during driving", "frequency for passing the time", "monthly listening frequency", "daily listening frequency", "gender", "age", "environment", "professional situation"]
    data.columns = new_cols

    # -------------------------------------------------------------
    # Nettoyage global des colonnes
    # -------------------------------------------------------------
    text_cols = data.select_dtypes(include=['object']).columns

    for col in text_cols:
        # ---------------------------------------------------------
        # Remplace "mot / mot" par "mot;mot" dans toutes les colonnes texte
        # ---------------------------------------------------------
        data[col] = data[col].str.replace(r'\s*/\s*', ';', regex=True)
        data[col] = data[col].str.replace(r'\s*,\s*', ';', regex=True)
        data[col] = data[col].str.replace(r'\s* ; \s*', ';', regex=True)
        data[col] = data[col].str.replace(r'\s* \( \s*', '(', regex=True)
        data[col] = data[col].str.replace(r'\s* \) \s*', ')', regex=True)
        data[col] = data[col].str.strip()
        data[col] = data[col].str.lower()
        
        # ---------------------------------------------------------
        # Vérification des valeurs dans la colonne "genre"
        # ---------------------------------------------------------
        if col == "gender":
            for entry in data[col]:
                if (entry != "femme") and (entry != "homme") and (entry.lower() != "non binaire") and (entry!= "Je préfère ne pas répondre"):
                    new_entry = "Je préfère ne pas répondre"
                    data[col] = data[col].replace(entry, new_entry)

            data[col].str.lower()
        
        # ---------------------------------------------------------
        # Traductions des réponses en anglais
        # ---------------------------------------------------------
        series = data[col].astype(str).str.strip().str.lower()
                
        if col == "instrumental or vocal music":
            mapping = {
                "instrumentaux": "instrumental music",
                "avec des paroles": "vocal music",
                "les deux": "both"
            }
            data[col] = series.replace(mapping)

        # Type de parole
        if col == "type of singing":
            mapping = {
                "peu importe": "doesn't matter",
                "engagées": "engaged",
                "poétiques": "poetic",
                "humoristiques": "humorous",
                "humoristiques": "humorous",
                "peut importe": "doesn't matter"
            }
            data[col] = series.map(lambda x: mapping.get(x, "prefer not to answer"))

        # Genre
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

        # Situation professionnelle
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

    # -------------------------------------------------------------
    # Normalisation des plateformes d'écoute
    # -------------------------------------------------------------
    def normalize_platform(g):
        if not isinstance(g, str):
            return None
        g = g.lower().strip()

        # Pop regroupée
        if g in ["plateforme de streaming", "avec alexa de chez amazon", "spotify", "deezer", "youtube", "youtube music", "apple music", "amazon music", "tidal", "soundcloud", "napster", "youtube et tv", "pc youtube", "et téléchargement", "youtube et réseaux sociaux", "téléchargement des musiques", "you tube"]:
            return "streaming platform"
        
        if g in ["morceaux locaux"]:
            return "local music"
        
        if g in ["radio"]:
            return "radio"
        
        if g in ["cd", "dvd", "vinyle", "vinyles", "cd;dvd", "dvd;cd", "cd;dvd;vinyle", "clé usb", "support usb", "chaînes de clips", "bibliothèque de musique", "mp3", "mon lecteur cd gulli", "cassette dans ma magnifique voiture", "mobile"]:
            return "cd;dvd;vinyl"

        if g in ["concert", "concerts", "atelier de musique", "live", "festival"]:
            return "concert"

        else:
            return g
    def apply_platform_normalization(value):
        if not isinstance(value, str):
            return np.nan

        genres = [g.strip() for g in value.split(";")]

        normalized = []
        for g in genres:
            ng = normalize_platform(g)
            if ng:
                normalized.append(ng)

        # Enlever doublons
        normalized = list(dict.fromkeys(normalized))
        return ";".join(normalized) if normalized else np.nan

    data["platform listening"] = data["platform listening"].apply(apply_platform_normalization)
    
        # -------------------------------------------------------------
    # Normalisation des langues d'écoutes
    # -------------------------------------------------------------
    def normalize_language(g):
        if not isinstance(g, str):
            return None
        g = g.lower().strip()

        # Pop regroupée
        if g in ["francophone"]:
            return "french"
        
        if g in ["anglophone"]:
            return "english"
        
        if g in ["asiatique"]:
            return "asian"
        if g in ["japonais", "japonaise"]:
            return "japanese"
        
        if g in ["coréen", "coréenne"]:
            return "korean"
        
        if g in ["hispanique"]:
            return "spanish"

        if g in ["brésilienne"]:
            return "brazilian"
        
        if g in ["allemande"]:
            return "german"
        
        if g in ["cyrillique", "russe"]:
            return "cyrillic"
        
        if g in ["italienne", "italien"]:
            return "italian"
        
        if g in ["scandinave"]:
            return "scandinavian"
        
        if g in ["bulgare"]:
            return "bulgarian"
        
        if g in ["bretonne"]:
            return "breton"

        if g in ["il y a pas trop de paroles", "tout"]:
            return ""

        else:
            return  g
    def apply_language_normalization(value):
        if not isinstance(value, str):
            return np.nan

        genres = [g.strip() for g in value.split(";")]

        normalized = []
        for g in genres:
            ng = normalize_language(g)
            if ng:
                normalized.append(ng)

        # Enlever doublons
        normalized = list(dict.fromkeys(normalized))
        return ";".join(normalized) if normalized else np.nan

    data["language listening"] = data["language listening"].apply(apply_language_normalization)

    # -------------------------------------------------------------
    # Normalisation de la période musicale
    # -------------------------------------------------------------
    def normalize_musical_period(g):
        if not isinstance(g, str):
            return None
        g = g.lower().strip()

        # Pop regroupée
        if g in ["années 50 - 70"]:
            return "1950s-1970s"
        
        if g in ["années 70 - 90"]:
            return "1970s-1990s"
        
        if g in ["années 90"]:
            return "1990s"
        
        if g in ["années 2000"]:
            return "2000s"
        
        if g in ["années 2010"]:
            return "2010s"
        
        if g in ["années 2020"]:
            return "2020s"
        
        if g in ["je ne sais pas"]:
            return "i don't know"
        
        if g in ["pas de préférence"]:
            return "no preference"

    def apply_musical_period_normalization(value):
        if not isinstance(value, str):
            return np.nan

        genres = [g.strip() for g in value.split(";")]

        normalized = []
        for g in genres:
            ng = normalize_musical_period(g)
            if ng:
                normalized.append(ng)

        # Enlever doublons
        normalized = list(dict.fromkeys(normalized))
        return ";".join(normalized) if normalized else np.nan

    data["musical period"] = data["musical period"].apply(apply_musical_period_normalization)

    # ======================================================
    # Suppresion des genres absurdes
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

    data["music style"] = data["music style"].apply(clean_absurd_genres)
    data["music style"] = data["music style"].replace(r'^\s*$', np.nan, regex=True)


    # ======================================================
    # Normalisation des genres musicaux
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
            "rock progressif", "rock'n'roll", "rock n roll", "rock and roll",
            "alternative", "gothique", "indie", "musique alternative"
        ] or g.startswith("rock"):
            return "rock"

        # Electro variations
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
            "industrial techno", "des gros kicks sa mere", "hxc", "tech(uptempo"
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

        # Aucun → pas de préférences
        if g in ["aucun préféré", "aucun", "aucun preference", "aucune idée", "pas de préférences"]:
            return "no preferences"

        # Musique classique
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

        # Enlever doublons
        normalized = list(dict.fromkeys(normalized))
        return ";".join(normalized) if normalized else np.nan

    data["music style"] = data["music style"].apply(apply_genre_normalization)


    # -------------------------------------------------------------
    # Mapping numérique
    # -------------------------------------------------------------
    tempo_map = {1: 60, 2: 90, 3: 120, 4: 150, 5: 180}
    data["tempo"] = data["tempo"].replace(tempo_map)

    freq_mensuelle = {
        "plus d'une fois par jour": 4,
        "plus d'une fois par semaine": 3,
        "plus d'une fois par mois": 2,
        "moins d'une fois par mois": 1,
    }
    data["monthly listening frequency"] = data["monthly listening frequency"].replace(freq_mensuelle)

    freq_jour = {
        "plus de trois heures par jour": 3,
        "plus d'une heure par jour": 2,
        "moins d'une heure par jour": 1,
    }
    data["daily listening frequency"] = data["monthly listening frequency"].replace(freq_jour)

    # -------------------------------------------------------------
    # Finalisation et export
    # -------------------------------------------------------------
    data.to_csv('cleaned_data.csv', index=False)
    print("cleaned_data.csv created successfully ✅")

    return data


# Lancement
cleaned = data_clean(data)

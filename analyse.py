import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from biplot import biplot
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1) Chargement du fichier (corrigé)
# ============================================================

data = pd.read_csv("cleaned_data.csv")  # <<< CORRIGÉ ICI

# ============================================================
# 2) Sélection des variables quantitatives
# ============================================================

data_quant = data[[
    "frequence ecouté artiste émergent", 
    "tempo", 
    "fréquence en travaillant", 
    "fréquence en sport", 
    "fréquence en cuisine", 
    "fréquence en transport", 
    "fréquence en passant le temps", 
    "age"
]].dropna()

print(data_quant.columns)

# ============================================================
# 3) Visualisation : matrice scatterplot
# ============================================================

sns.pairplot(data_quant)
plt.show()

# ============================================================
# 4) Standardisation
# ============================================================

temp = data_quant.sub(data_quant.mean())
x_scaled = temp.div(data_quant.std())
print(x_scaled.shape)

# ============================================================
# 5) PCA
# ============================================================

n_compo = 5
pca = PCA(n_components=n_compo)
pca_res = pca.fit_transform(x_scaled)

eig = pd.DataFrame({
    "Dimension": ["Dim" + str(x + 1) for x in range(n_compo)],
    "Valeur Propre": pca.explained_variance_,
    "% valeur propre": np.round(pca.explained_variance_ratio_ * 100),
    "% cum. val. prop.": np.round(np.cumsum(pca.explained_variance_ratio_) * 100)
})

print(eig)

# ============================================================
# 6) Histogramme des variances
# ============================================================

y1 = list(pca.explained_variance_ratio_)
x1 = range(len(y1))

plt.bar(x1, y1)
plt.xticks(list(x1), [f"Dim{i+1}" for i in x1])
plt.xlabel("Principal components")
plt.ylabel("Explained variance ratio")
plt.show()

# ============================================================
# 7) Biplot PCA
# ============================================================

biplot(
    score=pca_res[:220, 0:2],
    coeff=np.transpose(pca.components_[0:2, :]),
    cat=data['genre'].iloc[:220],
    coeff_labels=list(data_quant.columns),
    density=False
)
plt.show()

# ============================================================
# 8) Scatter PCA Dim1 vs Dim2
# ============================================================

pcadf = pd.DataFrame({
    "Dim1": pca_res[:221, 0],
    "Dim2": pca_res[:221, 1],
    "genre": data["genre"][:221]
})

listVariance = np.round(pca.explained_variance_ratio_ * 100)

pcadf.plot.scatter("Dim1", "Dim2")
plt.title(f"Variance expliquée : {listVariance[0]} % - {listVariance[1]} %")
plt.xlabel(f"Dimension 1 ({listVariance[0]}%)")
plt.ylabel(f"Dimension 2 ({listVariance[1]}%)")
plt.show()

# ============================================================
# 9) Analyse des genres musicaux (corrigée)
# ============================================================

df = pd.read_csv("cleaned_data.csv")  # <<< CORRIGÉ ICI AUSSI

# Découpage + explosion des genres
df_genres = df.copy()
df_genres['genre_list'] = df_genres['genre musicale'].str.split(';')

exploded = df_genres.explode('genre_list')

# Comptage
genre_counts = exploded['genre_list'].value_counts().reset_index()
genre_counts.columns = ['genre', 'count']
print(genre_counts)

# ============================================================
# 10) Graphique des genres ordonnés
# ============================================================

sorted_counts = genre_counts.sort_values('count')

plt.figure(figsize=(12, 10))
plt.barh(sorted_counts['genre'], sorted_counts['count'])
plt.xlabel('Number of mentions')
plt.ylabel('Music styles')
plt.title("Interest in music styles (number of mentions)")
plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('raw_data.csv')
data = data.drop(columns=data.columns[:3])

def data_clean(data):
    text_cols = data.select_dtypes(include=['object']).columns
    for col in text_cols:
        # Remplace "mot / mot" par "mot;mot" dans toutes les colonnes texte
        data[col] = data[col].str.replace(r'\s*/\s*', ';', regex=True)
        data[col] = data[col].str.strip()
        
    data.to_csv('cleaned_data.csv', index=False)
    return data

data_clean(data)
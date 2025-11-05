import csv
import pandas as pd

df = pd.read_csv("raw_data.csv")
df = df.fillna("null")
df = df.replace("blank", "null")
print(df.head())
df.to_csv("blank_removed.csv", index=False)

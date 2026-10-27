import pandas as pd

df = pd.read_csv("data/churn.csv")
print("Shape:", df.shape)
print(list(df.columns))
print(df["Churn"].value_counts())
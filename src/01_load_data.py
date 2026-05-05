import pandas as pd

df = pd.read_csv("data/DisneylandReviews.csv", encoding="latin1")

print(df.head())
print("\nSpaltennamen:")
print(df.columns)

print("\nForm des Datensatzes:")
print(df.shape)

print("\nFehlende Werte in Review_Text:")
print(df["Review_Text"].isna().sum())

import re
import pandas as pd

df = pd.read_csv("data/DisneylandReviews.csv", encoding="latin1")
df = df[["Review_Text"]].dropna()

print("Anzahl Zeilen nach Entfernen von NaN:", len(df))
print(df.head())

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = re.findall(r"\b[a-z]+\b", text)
    return " ".join(tokens)

df["clean_text"] = df["Review_Text"].astype(str).apply(clean_text)

print(df[["Review_Text", "clean_text"]].head())

df.to_csv("output/cleaned_reviews.csv", index=False)
print("Bereinigte Texte gespeichert unter output/cleaned_reviews.csv")


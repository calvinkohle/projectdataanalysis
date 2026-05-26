import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Bereinigte Daten laden
df = pd.read_csv("output/cleaned_reviews.csv")
texts = df["clean_text"].dropna()

# Parameter
max_features = 1000

# Bag of Words mit Stopwörtern entfernt
count_vectorizer = CountVectorizer(
    max_features=max_features,
    stop_words="english",   # klassische englische Stopwörter entfernen
    max_df=0.8              # sehr häufige Wörter (>80% der Dokumente) filtern
)
X_bow = count_vectorizer.fit_transform(texts)

# TF-IDF mit denselben Einstellungen
tfidf_vectorizer = TfidfVectorizer(
    max_features=max_features,
    stop_words="english",
    max_df=0.8
)
X_tfidf = tfidf_vectorizer.fit_transform(texts)

print("BoW (optimiert) Matrix-Form:", X_bow.shape)
print("TF-IDF (optimiert) Matrix-Form:", X_tfidf.shape)

bow_terms = count_vectorizer.get_feature_names_out()
tfidf_terms = tfidf_vectorizer.get_feature_names_out()

print("\nBeispiel-BoW-Begriffe (optimiert):")
print(bow_terms[:20])

print("\nBeispiel-TF-IDF-Begriffe (optimiert):")
print(tfidf_terms[:20])

# Optional: Ergebnisse für Dokumentation speichern
pd.DataFrame(X_bow.toarray(), columns=bow_terms).to_csv(
    "output/bow_matrix_optimized.csv", index=False
)
pd.DataFrame(X_tfidf.toarray(), columns=tfidf_terms).to_csv(
    "output/tfidf_matrix_optimized.csv", index=False
)

print("\nOptimierte Vektormatrizen wurden gespeichert.")
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Bereinigte Daten laden
df = pd.read_csv("output/cleaned_reviews.csv")

# Texte auswählen und leere Einträge entfernen
texts = df["clean_text"].dropna()

# Bag of Words
count_vectorizer = CountVectorizer(max_features=1000)
X_bow = count_vectorizer.fit_transform(texts)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(texts)

print("BoW Matrix-Form:", X_bow.shape)
print("TF-IDF Matrix-Form:", X_tfidf.shape)

# Beispielbegriffe
bow_terms = count_vectorizer.get_feature_names_out()
tfidf_terms = tfidf_vectorizer.get_feature_names_out()

print("\nErste 20 BoW-Begriffe:")
print(bow_terms[:20])

print("\nErste 20 TF-IDF-Begriffe:")
print(tfidf_terms[:20])

# Häufigste Begriffe in BoW
bow_sum = X_bow.sum(axis=0).A1
bow_top = sorted(zip(bow_terms, bow_sum), key=lambda x: x[1], reverse=True)[:15]

print("\nTop 15 Begriffe nach BoW:")
for word, value in bow_top:
    print(f"{word}: {value}")

# Höchste Gesamtwerte in TF-IDF
tfidf_sum = X_tfidf.sum(axis=0).A1
tfidf_top = sorted(zip(tfidf_terms, tfidf_sum), key=lambda x: x[1], reverse=True)[:15]

print("\nTop 15 Begriffe nach TF-IDF:")
for word, value in tfidf_top:
    print(f"{word}: {value:.2f}")
    
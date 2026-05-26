import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation

# Bereinigte Daten laden
df = pd.read_csv("output/cleaned_reviews.csv")
texts = df["clean_text"].dropna()

n_topics = 5
n_top_words = 10
max_features = 1000

results = []

# ------------------------
# LSA (optimiert) mit TF-IDF
# ------------------------
tfidf_vectorizer = TfidfVectorizer(
    max_features=max_features,
    stop_words="english",
    max_df=0.8
)
X_tfidf = tfidf_vectorizer.fit_transform(texts)

lsa_model = TruncatedSVD(n_components=n_topics, random_state=42)
lsa_model.fit(X_tfidf)

tfidf_terms = tfidf_vectorizer.get_feature_names_out()

print("LSA Topics (optimiert):")
for idx, topic in enumerate(lsa_model.components_):
    top_indices = topic.argsort()[-n_top_words:][::-1]
    top_words = [tfidf_terms[i] for i in top_indices]
    topic_words = ", ".join(top_words)
    print(f"Topic {idx + 1}: {topic_words}")
    results.append({
        "model": "LSA_optimized",
        "topic_number": idx + 1,
        "top_words": topic_words
    })

# ------------------------
# LDA (optimiert) mit BoW
# ------------------------
count_vectorizer = CountVectorizer(
    max_features=max_features,
    stop_words="english",
    max_df=0.8
)
X_bow = count_vectorizer.fit_transform(texts)

lda_model = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    learning_method="batch",
    max_iter=10
)
lda_model.fit(X_bow)

bow_terms = count_vectorizer.get_feature_names_out()

print("\nLDA Topics (optimiert):")
for idx, topic in enumerate(lda_model.components_):
    top_indices = topic.argsort()[-n_top_words:][::-1]
    top_words = [bow_terms[i] for i in top_indices]
    topic_words = ", ".join(top_words)
    print(f"Topic {idx + 1}: {topic_words}")
    results.append({
        "model": "LDA_optimized",
        "topic_number": idx + 1,
        "top_words": topic_words
    })

# Ergebnisse speichern
results_df = pd.DataFrame(results)
results_df.to_csv("output/topic_results_optimized.csv", index=False)

print("\nOptimierte Themen wurden gespeichert unter: output/topic_results_optimized.csv")
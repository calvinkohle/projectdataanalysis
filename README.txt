# Disneyland Reviews Text Analysis

Analysiert wird ausschließlich die Textvariable `Review_Text`, um häufige Themen und sprachliche Muster in den Bewertungen zu untersuchen.

## Verwendete Methoden
- Datenimport mit pandas
- Textvorverarbeitung mit Python und regulären Ausdrücken
- Vektorisierung mit Bag of Words (CountVectorizer)
- Vektorisierung mit TF-IDF (TfidfVectorizer)
- Themenextraktion mit LSA (TruncatedSVD)
- Themenextraktion mit LDA (LatentDirichletAllocation)

## Projektstruktur
- `data/` enthält den Originaldatensatz
- `src/` enthält die Python-Skripte
- `output/` enthält bereinigte Daten und Ergebnisse
- `requirements.txt` enthält die benötigten Python-Bibliotheken

## Ausführung
1. Virtuelle Umgebung aktivieren
2. Abhängigkeiten installieren:
   `pip install -r requirements.txt`
3. Skripte nacheinander ausführen:
   - `python3 src/01_load_data.py`
   - `python3 src/02_preprocess.py`
   - `python3 src/03_vectorize.py`
   - `python3 src/04_topic_models.py`

## Ergebnisse
Die bereinigten Texte werden in `output/cleaned_reviews.csv` gespeichert.  
Die extrahierten Themen werden in `output/topic_results.csv` gespeichert.
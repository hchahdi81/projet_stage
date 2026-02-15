from transformers import pipeline

# Charger un modèle pré-entraîné pour l'analyse textuelle
nlp_model = pipeline("text-classification", model="distilbert-base-uncased")

def analyze_text(text):
    return nlp_model(text)

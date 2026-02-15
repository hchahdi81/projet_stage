import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

# Chemins des fichiers
features_file = "combined_features.npy"
labels_file = "combined_labels.npy"
model_path = "models/classifier_all_cancer.pkl"
scaler_path = "models/scaler_all_cancer.pkl"

# Charger les données
features = np.load(features_file, allow_pickle=True)
labels = np.load(labels_file, allow_pickle=True)

# Vérifier les données et convertir si nécessaire
print("Cleaning and checking data")
try:
    features = features.astype(np.float64)  # Convertir en type float64
except ValueError:
    print("Error: Features contain non-numeric values. Cleaning up...")
    # Convertir en float en ignorant les colonnes non numériques
    numeric_features = []
    for row in features:
        try:
            numeric_row = np.array(row, dtype=np.float64)
            numeric_features.append(numeric_row)
        except ValueError:
            continue
    features = np.array(numeric_features)

# Nettoyer les valeurs infinies et aberrantes
features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
print(f"Features cleaned. Shape: {features.shape}")

# Normaliser les caractéristiques
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Entraîner le modèle
print("Training the Random Forest model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Évaluer le modèle
accuracy = clf.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Sauvegarder le modèle et le scaler
os.makedirs("models", exist_ok=True)
with open(model_path, "wb") as model_file:
    pickle.dump(clf, model_file)
with open(scaler_path, "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print(f"Model and scaler saved in '{model_path}' and '{scaler_path}'.")

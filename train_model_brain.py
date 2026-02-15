import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Charger les fichiers .npy
tumor_features = np.load("brain_tumor_features_with_labels.npy", allow_pickle=True)
no_tumor_features = np.load("no_tumor_brain_features_with_labels.npy", allow_pickle=True)
test_features = np.load("test_features.npy", allow_pickle=True)

# Combiner les données tumorales et non tumorales
train_features = np.concatenate([tumor_features, no_tumor_features], axis=0)

# Normaliser les labels (minuscule, suppression des espaces)
def normalize_labels(labels):
    return [str(label).strip().lower() for label in labels]

# Séparer les caractéristiques et les labels pour l'entraînement
X_train = np.array([x[:-1] for x in train_features], dtype=np.float32)  # Caractéristiques
y_train = normalize_labels([x[-1] for x in train_features])  # Labels

# Séparer les caractéristiques et les labels pour le test
X_test = np.array([x[:-1] for x in test_features], dtype=np.float32)  # Caractéristiques
y_test = normalize_labels([x[-1] for x in test_features])  # Labels

# Vérifier les classes uniques après normalisation
print("Classes uniques dans les données d'entraînement :", np.unique(y_train))
print("Classes uniques dans les données de test :", np.unique(y_test))

# Vérifier et nettoyer les données
def clean_data(X, y):
    is_finite = np.all(np.isfinite(X), axis=1)  # Vérifie si toutes les valeurs sont finies
    cleaned_X = X[is_finite]
    cleaned_y = np.array(y)[is_finite]
    print(f"Cleaned data: {cleaned_X.shape[0]} samples remain after removing invalid rows.")
    return cleaned_X, cleaned_y

X_train, y_train = clean_data(X_train, y_train)
X_test, y_test = clean_data(X_test, y_test)

# Vérifier les dimensions
print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")

# Normaliser les données avec StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entraîner le modèle RandomForestClassifier
clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
print("Training the Random Forest model...")
clf.fit(X_train, y_train)

# Évaluer le modèle sur les données de test
y_test_pred = clf.predict(X_test)
print("\nTest Results:")
print(classification_report(y_test, y_test_pred))
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred) * 100:.2f}%")

# Sauvegarder le modèle et le scaler
model_file = "models/classifier_brain_cancer.pkl"
scaler_file = "models/scaler_brain_cancer.pkl"

with open(model_file, "wb") as model_out:
    pickle.dump(clf, model_out)
with open(scaler_file, "wb") as scaler_out:
    pickle.dump(scaler, scaler_out)

print(f"\nModel and scaler saved to '{model_file}' and '{scaler_file}'")

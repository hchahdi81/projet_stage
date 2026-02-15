from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil
import os
import pickle
import numpy as np
import cv2
from app.descriptor import bit_glcm_haralick_beta  # Importez votre fonction d'extraction

app = FastAPI(title="Multimodal Analysis Service")

# Configuration des templates et fichiers statiques
BASE_DIR = Path(__file__).resolve().parent
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Charger le modèle et le scaler
model_path = "C:/Users/hatim/Desktop/2025/medical_project/multimodal_analysis/app/models/classifier_all_cancer.pkl"
scaler_path = "C:/Users/hatim/Desktop/2025/medical_project/multimodal_analysis/app/models/scaler_all_cancer.pkl"

with open(model_path, "rb") as model_file:
    clf = pickle.load(model_file)
with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)


# Routes principales
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})

@app.post("/predict")
async def predict_post(request: Request, file: UploadFile):
    try:
        # Enregistrer temporairement l'image téléchargée
        upload_folder = "static/uploads/"
        os.makedirs(upload_folder, exist_ok=True)  # Créer le dossier s'il n'existe pas
        temp_file = os.path.join(upload_folder, file.filename)
        
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Vérification si le fichier a été bien sauvegardé
        if not os.path.exists(temp_file):
            return {"error": f"Failed to save the uploaded file: {temp_file}"}

        # Charger l'image et extraire les caractéristiques
        image = cv2.imread(temp_file, 0)
        if image is None:
            return {"error": f"Failed to read the uploaded image: {temp_file}"}
        
        features = bit_glcm_haralick_beta(temp_file)
        print(f"Extracted features: {features}")  # Debug

        features = scaler.transform([features])  # Normaliser les caractéristiques
        print(f"Normalized features: {features}")  # Debug

        # Obtenir les probabilités de prédiction
        probabilities = clf.predict_proba(features)[0]
        max_probability = max(probabilities)
        predicted_class = clf.classes_[np.argmax(probabilities)]  # La classe avec le pourcentage max

        # Afficher toutes les probabilités avec les noms des classes
        probabilities_with_labels = [
            f"{label}: {probability:.2f}" 
            for label, probability in zip(clf.classes_, probabilities)
        ]
        print(f"Probabilities with labels: {probabilities_with_labels}")  # Debug

        # Formater le résultat en fonction de la classe avec le pourcentage maximum
        result = f"Predicted Cancer Type: {predicted_class} (Confidence: {max_probability:.2f})"

        # Supprimer le fichier temporaire après utilisation
        os.remove(temp_file)

        # Générer des questions dynamiques
        generic_questions = [
            f"What are the symptoms of {predicted_class}?",
            f"What are the treatment options for {predicted_class}?",
            f"What are the preventions of the {predicted_class} before I see the doctor?",
            f"How does {predicted_class} affect the patient?",
            f"Can {predicted_class} be treated without surgery?",
            f"What are the other treatment options available to deal with this {predicted_class} disease?",
        ]

        # Retourner le résultat avec les questions
        return templates.TemplateResponse(
            "predict.html",
            {
                "request": request,
                "result": result,
                "predicted_class": predicted_class,
                "probabilities": probabilities_with_labels,
                "questions": generic_questions,
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "predict.html",
            {
                "request": request,
                "result": f"Prediction failed: {str(e)}",
            }
        )


model_brain_path = "C:/Users/hatim/Desktop/2025/medical_project/multimodal_analysis/app/models/classifier_brain_cancer.pkl"
scaler_brain_path = "C:/Users/hatim/Desktop/2025/medical_project/multimodal_analysis/app/models/scaler_brain_cancer.pkl"

import subprocess

@app.get("/predict_brain", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("predict_brain.html", {"request": request})


# Charger le modèle et le scaler
with open(model_brain_path, "rb") as f:
    model = pickle.load(f)
with open(scaler_brain_path, "rb") as f:
    scaler = pickle.load(f)

# Fonction pour interroger MedLLaMA2
def query_medllama2(question):
    try:
        result = subprocess.run(
            ["ollama", "run", "medllama2"],
            input=question,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip() if result.returncode == 0 else f"Error: {result.stderr.strip()}"
    except Exception as e:
        return f"Failed to query medllama2: {e}"
    
@app.post("/ask_question")
async def ask_question(request: Request):
    try:
        form_data = await request.form()
        question = form_data.get("question")

        if not question:
            return {"response": "No question provided."}

        # Appeler MedLLaMA2 avec la question sélectionnée
        response = query_medllama2(question)
        return {"response": response}
    except Exception as e:
        return {"response": f"Error occurred: {str(e)}"}
@app.post("/predict_brain")
async def predict_brain(request: Request, file: UploadFile):
    try:
        upload_folder = "static/uploads/"
        os.makedirs(upload_folder, exist_ok=True)
        temp_file = os.path.join(upload_folder, file.filename)

        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if not os.path.exists(temp_file):
            return {"error": f"Failed to save the uploaded file: {temp_file}"}

        image = cv2.imread(temp_file, 0)
        if image is None:
            return {"error": f"Failed to read the uploaded image: {temp_file}"}

        features = bit_glcm_haralick_beta(temp_file)
        features = scaler.transform([features])

        probabilities = model.predict_proba(features)[0]
        max_probability = max(probabilities)
        predicted_class = model.classes_[np.argmax(probabilities)]

        probabilities_with_labels = [
            f"{label}: {probability:.2f}"
            for label, probability in zip(model.classes_, probabilities)
        ]

        confidence_threshold = 0.5
        if max_probability < confidence_threshold:
            result = "No cancer type detected (low confidence)."
            predicted_class = "no_tumor"
        else:
            result = f"Predicted Cancer Type: {predicted_class} (Confidence: {max_probability:.2f})"

        os.remove(temp_file)

        # Générer des questions dynamiques
        generic_questions = [
            f"What are the symptoms of brain {predicted_class}?",
            f"What are the treatment options for brain {predicted_class}?",
            f"What are the preventions of the brain {predicted_class} before I see the doctor?",
            f"How does brain {predicted_class} affect the patient?",
            f"Can brain {predicted_class} be treated without surgery?",
            f"What are the other treatment options available to deal with this brain {predicted_class} disease?",
        ]

        return templates.TemplateResponse(
            "predict_brain.html",
            {
                "request": request,
                "result": result,
                "predicted_class": predicted_class,
                "probabilities": probabilities_with_labels,
                "questions": generic_questions,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "predict_brain.html",
            {
                "request": request,
                "result": f"Prediction failed: {str(e)}",
            },
        )

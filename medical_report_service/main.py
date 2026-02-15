from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
import pdfplumber
import subprocess
from fastapi.staticfiles import StaticFiles


# Montre les fichiers statiques depuis le dossier "static"

# Initialisation de l'application FastAPI et du moteur de templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Dossier pour stocker temporairement les fichiers téléchargés
UPLOAD_FOLDER = "uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

history = []

# Route pour afficher la page principale
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

# Fonction pour interroger le modèle avec un contexte
def query_model(question, context=""):
    try:
        # Fournir un contexte clair pour Ollama
        input_text = (
            f"Voici un extrait d'un rapport médical. Vous devez répondre uniquement aux questions basées sur ce contenu.\n\n"
            f"--- Début du Rapport ---\n{context}\n--- Fin du Rapport ---\n\n"
            f"Question : {question}\n\n"
            "Répondez uniquement en français, s'il vous plaît.\n"
            "Si la réponse ne peut pas être dérivée du contenu ci-dessus, répondez strictement par : 'Je ne sais pas'."
        )
        
        # Pass input_text as a string
        result = subprocess.run(
            ["ollama", "run", "llama3.2"],
            input=input_text,  # Pass as a string
            text=True,         # Enable text mode for subprocess
            capture_output=True,
            encoding="utf-8"  # Ajoutez explicitement l'encodage

        )
        
        # Decode the output correctly and handle errors
        return result.stdout.strip().encode('utf-8', 'replace').decode('utf-8') if result.returncode == 0 else f"Erreur : {result.stderr.strip()}"
    except Exception as e:
        return f"Échec lors de la requête au modèle : {e}"


# Route pour téléverser un fichier PDF
@app.post("/upload")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    try:
        # Sauvegarde temporaire du fichier
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Extraction du texte du PDF
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

        # Suppression du fichier temporaire
        os.remove(file_path)

        # Afficher la page avec le texte extrait
        return templates.TemplateResponse(
            "pdf_viewer.html",
            {"request": request, "text": text, "file_name": file.filename},
        )
    except Exception as e:
        return templates.TemplateResponse(
            "pdf_viewer.html",
            {"request": request, "text": f"Error: {str(e)}", "file_name": None},
        )


# Route pour poser une question
@app.post("/ask_question", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...), text: str = Form(...)):
    try:
        response = query_model(question, text)
        history.append((question, response))  # Ajout de la question et de la réponse dans l'historique

        return templates.TemplateResponse(
            "pdf_viewer.html",
            {"request": request, "text": text, "response": response, "question": question, "history": history}
        )
    except Exception as e:
        return templates.TemplateResponse(
            "pdf_viewer.html",
            {"request": request, "text": text, "response": f"Erreur : {str(e)}", "question": question, "history": history}
        )
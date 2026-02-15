from fastapi import FastAPI, Request, Form, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import subprocess
import random
import json

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

# Stocker l'historique du chat pour chaque session
clients = []

def query_ollama(question, context=""):
    """
    Interroge Ollama avec un contexte médical et génère aussi des questions de suivi et juste en francais.
    Si une réponse est en anglais, elle est rejetée et reformulée.

    """
    try:
        input_text = (
            "Vous êtes un assistant médical qui aide les patients à diagnostiquer leurs symptômes.\n"
            "Si un patient pose une question, répondez et posez-lui ensuite une autre question pour approfondir son état.\n"
            "Vous devez poser des questions pertinentes pour diagnostiquer au mieux le patient.\n"
            "Si votre réponse contient de l'anglais, reformulez-la immédiatement en français.\n\n"
            "Répondez toujours en français.\n"
            f"Question du patient : {question}\n\n"
            "Réponse et nouvelle question :"
        )

        result = subprocess.run(
            ["ollama", "run", "medllama2"],
            input=input_text,
            text=True,
            capture_output=True,
            encoding="utf-8"
        )

        if result.returncode != 0:
            return f"Erreur du modèle : {result.stderr.strip()}"

        return result.stdout.strip()

    except Exception as e:
        return f"Erreur lors de la requête au modèle : {e}"

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    try:
        while True:
            question = await websocket.receive_text()
            response = query_ollama(question)
            await websocket.send_text(response)
    except WebSocketDisconnect:
        clients.remove(websocket)

@app.get("/", response_class=HTMLResponse)
async def chat_interface(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

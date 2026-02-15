from fastapi import APIRouter, Form, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from pathlib import Path
from models.text_model import analyze_text
from models.image_model import analyze_image

router = APIRouter()

# Dossier pour stocker temporairement les fichiers uploadés
UPLOAD_DIR = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory="templates")

@router.get("/", response_class="HTMLResponse")
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@router.post("/multimodal", response_class="HTMLResponse")
async def multimodal_analysis(
    request: Request,
    text: str = Form(...),
    image: UploadFile = File(...)
):
    # Analyser le texte
    text_result = analyze_text(text)

    # Stocker et analyser l'image
    image_path = UPLOAD_DIR / image.filename
    with open(image_path, "wb") as f:
        f.write(image.file.read())
    image_result = analyze_image(image_path)

    # Renvoyer les résultats
    return templates.TemplateResponse("upload.html", {
        "request": request,
        "text_analysis": text_result,
        "image_analysis": f"Predicted class: {image_result}"
    })

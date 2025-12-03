# app/main.py
import uuid
from pathlib import Path
from typing import Dict, Optional
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .services.ocr import extract_text_from_file
from .services.qa import process_document_text, answer_question

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"

UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

app = FastAPI(title="DocVis - OCR-Based Document Q&A System")

# Static files (CSS) and templates
app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "app" / "static")),
    name="static",
)
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))

# In-memory "database" for MVP
DOCUMENTS: Dict[str, dict] = {}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Home page: file upload + list of documents.
    """
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "documents": DOCUMENTS},
    )


@app.post("/upload", response_class=RedirectResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Handle file upload, run OCR, prepare for Q&A, then redirect to doc page.
    """
    # Generate document ID
    doc_id = str(uuid.uuid4())

    # Determine original extension and save file
    file_ext = Path(file.filename).suffix or ".bin"
    saved_path = UPLOAD_DIR / f"{doc_id}{file_ext}"

    with saved_path.open("wb") as f:
        f.write(await file.read())

    # Run OCR + preprocessing
    text = extract_text_from_file(str(saved_path))

    # Prepare for Q&A (chunking, embeddings later, summary)
    summary, meta = process_document_text(doc_id, text)

    # Store in memory (for now)
    DOCUMENTS[doc_id] = {
        "id": doc_id,
        "filename": file.filename,
        "saved_path": str(saved_path),
        "summary": summary,
        "meta": meta,  # meta might include num_chunks, doc_type, etc.
    }

    # Redirect to document view
    return RedirectResponse(url=f"/doc/{doc_id}", status_code=303)


@app.get("/doc/{doc_id}", response_class=HTMLResponse)
async def view_document(doc_id: str, request: Request):
    """
    Show document summary + QA interface (no answer yet).
    """
    doc = DOCUMENTS.get(doc_id)
    if not doc:
        return templates.TemplateResponse(
            "document.html",
            {
                "request": request,
                "error": "Document not found",
                "doc": None,
                "answer": None,
                "references": [],
            },
        )

    return templates.TemplateResponse(
        "document.html",
        {
            "request": request,
            "doc": doc,
            "answer": None,
            "references": [],
        },
    )


@app.post("/doc/{doc_id}", response_class=HTMLResponse)
async def view_document_post(doc_id: str, request: Request, question: str = Form(...)):
    """
    Handle Q&A form submit and re-render page with answer.
    """
    doc = DOCUMENTS.get(doc_id)
    if not doc:
        return templates.TemplateResponse(
            "document.html",
            {
                "request": request,
                "error": "Document not found",
                "doc": None,
                "answer": None,
                "references": [],
            },
        )
    from .services.qa import answer_question

    answer, references = answer_question(doc_id, question, doc["meta"])

    return templates.TemplateResponse(
        "document.html",
        {
            "request": request,
            "doc": doc,
            "answer": answer,
            "references": references,
        },
    )

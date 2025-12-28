import os, json
import numpy as np
import cv2
import faiss

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from insightface.app import FaceAnalysis

# =========================
# PATH ASSOLUTI SICURI
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PHOTOS_DIR = os.path.join(BASE_DIR, "photos")
DATA_DIR   = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# CREA LE CARTELLE SE NON ESISTONO (FONDAMENTALE)
os.makedirs(PHOTOS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

INDEX_PATH = os.path.join(DATA_DIR, "faces.index")
META_PATH  = os.path.join(DATA_DIR, "faces.meta.jsonl")

# =========================
# FASTAPI
# =========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/photos", StaticFiles(directory=PHOTOS_DIR), name="photos")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# =========================
# FACE ENGINE
# =========================

def norm(v):
    v = v.astype("float32")
    return v / (np.linalg.norm(v) + 1e-8)

print("Carico motore facciale...")
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(1024, 1024))

print("Carico indice...")

if not os.path.exists(INDEX_PATH):
    print("Indice non trovato, avvio senza matching")
    index = None
    meta = []
else:
    index = faiss.read_index(INDEX_PATH)
    meta = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))

def get_embedding(image_bgr):
    faces = face_app.get(image_bgr)
    if not faces:
        return None
    faces.sort(key=lambda f: f.det_score, reverse=True)
    return norm(faces[0].embedding)

# =========================
# API
# =========================

@app.post("/match_selfie")
async def match_selfie(
    file: UploadFile = File(...),
    top_k_faces: int = 80,
    min_score: float = 0.30
):
    if index is None:
        return {"ok": False, "error": "Indice non inizializzato"}

    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "error": "Immagine non valida"}

    emb = get_embedding(img)
    if emb is None:
        return {"ok": False, "error": "Nessun volto rilevato"}

    D, I = index.search(emb.reshape(1, -1), top_k_faces)

    photo_best = {}
    for score, idx in zip(D[0], I[0]):
        if idx == -1 or score < min_score:
            continue
        pid = meta[idx]["photo_id"]
        if pid not in photo_best or score > photo_best[pid]:
            photo_best[pid] = float(score)

    results = [{"photo_id": k, "score": v} for k, v in photo_best.items()]
    results.sort(key=lambda x: x["score"], reverse=True)

    return {"ok": True, "count": len(results), "results": results}

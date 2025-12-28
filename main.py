import os
import json

import numpy as np
import cv2
import faiss

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from insightface.app import FaceAnalysis


# --- PATHS ASSOLUTI (sempre ok su Render) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PHOTOS_DIR = os.path.join(BASE_DIR, "photos")
DATA_DIR   = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")

INDEX_PATH = os.path.join(DATA_DIR, "faces.index")
META_PATH  = os.path.join(DATA_DIR, "faces.meta.jsonl")


app = FastAPI()

# Se le cartelle non esistono, fallo capire subito nei log
if not os.path.isdir(PHOTOS_DIR):
    raise RuntimeError(f"Directory photos non trovata: {PHOTOS_DIR}")
if not os.path.isdir(DATA_DIR):
    raise RuntimeError(f"Directory data non trovata: {DATA_DIR}")
if not os.path.isdir(STATIC_DIR):
    raise RuntimeError(f"Directory static non trovata: {STATIC_DIR}")

app.mount("/photos", StaticFiles(directory=PHOTOS_DIR), name="photos")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def norm(v: np.ndarray) -> np.ndarray:
    v = v.astype("float32")
    return v / (np.linalg.norm(v) + 1e-8)


print("Carico motore facciale...")
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(1024, 1024))


print("Carico indice...")
if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
    raise RuntimeError(
        f"Indice non trovato. Manca {INDEX_PATH} o {META_PATH}. "
        "Prima esegui: python index_folder.py"
    )

index = faiss.read_index(INDEX_PATH)

meta = []
with open(META_PATH, "r", encoding="utf-8") as f:
    for line in f:
        meta.append(json.loads(line))


def get_embedding(image_bgr: np.ndarray):
    faces = face_app.get(image_bgr)
    if not faces:
        return None
    faces.sort(key=lambda f: f.det_score, reverse=True)
    emb = faces[0].embedding
    return norm(emb)


@app.post("/match_selfie")
async def match_selfie(
    file: UploadFile = File(...),
    top_k_faces: int = 80,
    min_score: float = 0.30
):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "error": "Immagine non valida"}

    emb = get_embedding(img)
    if emb is None:
        return {"ok": False, "error": "Nessun volto rilevato nel selfie"}

    if index.ntotal == 0:
        return {"ok": False, "error": "Nessuna faccia indicizzata"}

    D, I = index.search(emb.reshape(1, -1), top_k_faces)

    # deduplica per foto: tieni lo score migliore
    photo_best = {}
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1:
            continue
        if score < min_score:
            continue

        photo_id = meta[idx]["photo_id"]
        if (photo_id not in photo_best) or (score > photo_best[photo_id]):
            photo_best[photo_id] = score

    results = [{"photo_id": k, "score": float(v)} for k, v in photo_best.items()]
    results.sort(key=lambda x: x["score"], reverse=True)

    return {"ok": True, "count": len(results), "results": results}

import os
import json
import numpy as np
import cv2
import faiss

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from insightface.app import FaceAnalysis

# =========================
# PATH ASSOLUTI SICURI
# =========================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR = os.path.join(BASE_DIR, "photos")
DATA_DIR   = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(PHOTOS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

INDEX_PATH = os.path.join(DATA_DIR, "faces.index")

# il tuo file è "faces.meta.jsonl" (dallo screenshot)
META_PATH  = os.path.join(DATA_DIR, "faces.meta.jsonl")
if not os.path.exists(META_PATH):
    # fallback se per caso l'hai chiamato in modo diverso
    alt = os.path.join(DATA_DIR, "faces.meta.jsonl")
    if os.path.exists(alt):
        META_PATH = alt

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


@app.get("/health")
def health():
    return {"ok": True}


def norm(v: np.ndarray) -> np.ndarray:
    v = v.astype("float32")
    return v / (np.linalg.norm(v) + 1e-8)


# =========================
# CARICO MOTORE FACCIALE
# =========================
print("Carico motore facciale (CPU)...")
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
# det_size più basso = meno RAM/CPU; se vuoi più sensibilità, prova (1024,1024)
face_app.prepare(ctx_id=-1, det_size=(640, 640))


# =========================
# CARICO INDICE + META
# =========================
print("Carico indice...")
if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
    raise RuntimeError(
        "Indice o meta non trovati.\n"
        "Prima esegui localmente:\n"
        "  python index_folder.py\n"
        "Poi fai commit & push di:\n"
        "  data/faces.index\n"
        "  data/faces.meta.jsonl\n"
    )

index = faiss.read_index(INDEX_PATH)

meta = []
with open(META_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        meta.append(json.loads(line))

print(f"Indice caricato. Vettori totali: {index.ntotal}. Meta righe: {len(meta)}")


def get_embedding(image_bgr: np.ndarray):
    faces = face_app.get(image_bgr)
    if not faces:
        return None

    # prendo il volto più "sicuro" (score detection più alto)
    faces.sort(key=lambda f: f.det_score, reverse=True)
    emb = faces[0].embedding
    return norm(emb)


@app.post("/match_selfie")
async def match_selfie(
    file: UploadFile = File(...),
    top_k_faces: int = 200,
    min_score: float = 0.30,
    max_results: int = 30,
):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"ok": False, "error": "Immagine non valida"}, status_code=400)

    emb = get_embedding(img)
    if emb is None:
        return {"ok": False, "error": "Nessun volto rilevato nel selfie"}

    if index.ntotal == 0:
        return {"ok": False, "error": "Indice vuoto: nessuna faccia indicizzata"}

    # Faiss con IndexFlatIP + vettori normalizzati => score ~ cosine similarity
    D, I = index.search(emb.reshape(1, -1), int(top_k_faces))

    # deduplica per foto: tieni lo score migliore per ogni foto
    photo_best = {}
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1:
            continue
        if score < float(min_score):
            continue
        if idx >= len(meta):
            continue

        photo_id = meta[idx].get("photo_id")
        if not photo_id:
            continue

        # tieni il migliore per quella foto
        if (photo_id not in photo_best) or (score > photo_best[photo_id]):
            photo_best[photo_id] = float(score)

    results = [{"photo_id": k, "score": v, "url": f"/photos/{k}"} for k, v in photo_best.items()]
    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[: int(max_results)]

    return {"ok": True, "count": len(results), "results": results}
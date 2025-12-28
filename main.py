from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import cv2
import faiss

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from insightface.app import FaceAnalysis


# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PHOTOS_DIR = BASE_DIR / "photos"
STATIC_DIR = BASE_DIR / "static"

INDEX_PATH = DATA_DIR / "faces.index"
META_PATH = DATA_DIR / "faces.meta.jsonl"
INDEX_DIM = 512  # InsightFace buffalo_l = 512


# -------------------------
# App
# -------------------------
app = FastAPI(title="Face Site")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache
face_app: FaceAnalysis | None = None
faiss_index: faiss.Index | None = None
meta_rows: List[Dict[str, Any]] = []


# -------------------------
# Helpers
# -------------------------
def _load_meta_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _ensure_ready():
    if face_app is None or faiss_index is None or not meta_rows:
        raise HTTPException(
            status_code=503,
            detail="Service not ready: model/index/meta not loaded."
        )


def _read_image_from_upload(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return img


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


# -------------------------
# Startup
# -------------------------
@app.on_event("startup")
def startup():
    global face_app, faiss_index, meta_rows

    # 1) Load model
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    # 2) Load index + meta
    if not INDEX_PATH.exists() or not META_PATH.exists():
        # Non crashiamo l’app: risponde 503 finché non ci sono i file
        faiss_index = None
        meta_rows = []
        return

    try:
        faiss_index = faiss.read_index(str(INDEX_PATH))
    except Exception as e:
        faiss_index = None
        meta_rows = []
        raise RuntimeError(f"Cannot read FAISS index: {e}")

    meta_rows = _load_meta_jsonl(META_PATH)


# -------------------------
# Routes: UI + Photos
# -------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/photo/{filename}")
def get_photo(filename: str):
    # sicurezza base
    safe_name = Path(filename).name
    path = PHOTOS_DIR / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Photo not found")
    return FileResponse(path)


@app.get("/health")
def health():
    return {"ok": True}


# -------------------------
# Core: match selfie
# -------------------------
@app.post("/match_selfie")
async def match_selfie(
    file: UploadFile = File(...),
    top_k_faces: int = Query(80, ge=1, le=500),
    min_score: float = Query(0.08, ge=0.0, le=1.0),
):
    _ensure_ready()

    # read upload
    file_bytes = await file.read()
    img = _read_image_from_upload(file_bytes)

    # detect faces on selfie
    assert face_app is not None
    faces = face_app.get(img)
    if not faces:
        return JSONResponse({"ok": True, "count": 0, "results": []})

    # prendi il volto più grande
    faces_sorted = sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True
    )
    emb = faces_sorted[0].embedding.astype("float32")
    emb = _normalize(emb).reshape(1, -1)

    # search FAISS (IndexFlatIP => score ~ cosine se normalizzato)
    assert faiss_index is not None
    D, I = faiss_index.search(emb, top_k_faces)

    results: List[Dict[str, Any]] = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0:
            continue
        if score < float(min_score):
            continue

        if idx >= len(meta_rows):
            continue

        row = meta_rows[idx]
        photo_id = row.get("photo_id") or row.get("filename") or row.get("id")
        if not photo_id:
            continue

        results.append({
            "photo_id": str(photo_id),
            "score": float(score),
        })

    return {"ok": True, "count": len(results), "results": resulimport os
import io
import json
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"

INDEX_PATH = Path(os.getenv("INDEX_PATH", str(DATA_DIR / "faces.index")))
META_PATH = Path(os.getenv("META_PATH", str(DATA_DIR / "faces.meta.jsonl")))

# Prova più posizioni per la cartella foto
CANDIDATE_PHOTO_DIRS = []
env_photos = os.getenv("PHOTOS_DIR")
if env_photos:
    CANDIDATE_PHOTO_DIRS.append(Path(env_photos))

CANDIDATE_PHOTO_DIRS += [
    BASE_DIR / "photos",
    BASE_DIR.parent / "photos",
    BASE_DIR / "backend" / "photos",
    BASE_DIR.parent / "backend" / "photos",
]

PHOTOS_DIR = None
for p in CANDIDATE_PHOTO_DIRS:
    if p.exists() and p.is_dir():
        PHOTOS_DIR = p
        break
if PHOTOS_DIR is None:
    PHOTOS_DIR = BASE_DIR / "photos"  # fallback


# =========================
# APP
# =========================
app = FastAPI(title="Face Match API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # poi restringi al tuo dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve i file statici (UI)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# =========================
# GLOBAL CACHE
# =========================
_face_app = None
_faiss_index = None
_meta_rows: Optional[List[Dict[str, Any]]] = None


# =========================
# HELPERS
# =========================
def _load_face_engine():from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import cv2
import faiss

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from insightface.app import FaceAnalysis


# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PHOTOS_DIR = BASE_DIR / "photos"
STATIC_DIR = BASE_DIR / "static"

INDEX_PATH = DATA_DIR / "faces.index"
META_PATH = DATA_DIR / "faces.meta.jsonl"
INDEX_DIM = 512  # InsightFace buffalo_l = 512


# -------------------------
# App
# -------------------------
app = FastAPI(title="Face Site")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache
face_app: FaceAnalysis | None = None
faiss_index: faiss.Index | None = None
meta_rows: List[Dict[str, Any]] = []


# -------------------------
# Helpers
# -------------------------
def _load_meta_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _ensure_ready():
    if face_app is None or faiss_index is None or not meta_rows:
        raise HTTPException(
            status_code=503,
            detail="Service not ready: model/index/meta not loaded."
        )


def _read_image_from_upload(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return img


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


# -------------------------
# Startup
# -------------------------
@app.on_event("startup")
def startup():
    global face_app, faiss_index, meta_rows

    # 1) Load model
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    # 2) Load index + meta
    if not INDEX_PATH.exists() or not META_PATH.exists():
        # Non crashiamo l’app: risponde 503 finché non ci sono i file
        faiss_index = None
        meta_rows = []
        return

    try:
        faiss_index = faiss.read_index(str(INDEX_PATH))
    except Exception as e:
        faiss_index = None
        meta_rows = []
        raise RuntimeError(f"Cannot read FAISS index: {e}")

    meta_rows = _load_meta_jsonl(META_PATH)


# -------------------------
# Routes: UI + Photos
# -------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/photo/{filename}")
def get_photo(filename: str):
    # sicurezza base
    safe_name = Path(filename).name
    path = PHOTOS_DIR / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Photo not found")
    return FileResponse(path)


@app.get("/health")
def health():
    return {"ok": True}


# -------------------------
# Core: match selfie
# -------------------------
@app.post("/match_selfie")
async def match_selfie(
    file: UploadFile = File(...),
    top_k_faces: int = Query(80, ge=1, le=500),
    min_score: float = Query(0.08, ge=0.0, le=1.0),
):
    _ensure_ready()

    # read upload
    file_bytes = await file.read()
    img = _read_image_from_upload(file_bytes)

    # detect faces on selfie
    assert face_app is not None
    faces = face_app.get(img)
    if not faces:
        return JSONResponse({"ok": True, "count": 0, "results": []})

    # prendi il volto più grande
    faces_sorted = sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True
    )
    emb = faces_sorted[0].embedding.astype("float32")
    emb = _normalize(emb).reshape(1, -1)

    # search FAISS (IndexFlatIP => score ~ cosine se normalizzato)
    assert faiss_index is not None
    D, I = faiss_index.search(emb, top_k_faces)

    results: List[Dict[str, Any]] = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0:
            continue
        if score < float(min_score):
            continue

        if idx >= len(meta_rows):
            continue

        row = meta_rows[idx]
        photo_id = row.get("photo_id") or row.get("filename") or row.get("id")
        if not photo_id:
            continue

        results.append({
            "photo_id": str(photo_id),
            "score": float(score),
        })

    return {"ok": True, "count": len(results), "results": results}from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import cv2
import faiss

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from insightface.app import FaceAnalysis


# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PHOTOS_DIR = BASE_DIR / "photos"
STATIC_DIR = BASE_DIR / "static"

INDEX_PATH = DATA_DIR / "faces.index"
META_PATH = DATA_DIR / "faces.meta.jsonl"
INDEX_DIM = 512  # InsightFace buffalo_l = 512


# -------------------------
# App
# -------------------------
app = FastAPI(title="Face Site")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache
face_app: FaceAnalysis | None = None
faiss_index: faiss.Index | None = None
meta_rows: List[Dict[str, Any]] = []


# -------------------------
# Helpers
# -------------------------
def _load_meta_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _ensure_ready():
    if face_app is None or faiss_index is None or not meta_rows:
        raise HTTPException(
            status_code=503,
            detail="Service not ready: model/index/meta not loaded."
        )


def _read_image_from_upload(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return img


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


# -------------------------
# Startup
# -------------------------
@app.on_event("startup")
def startup():
    global face_app, faiss_index, meta_rows

    # 1) Load model
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    # 2) Load index + meta
    if not INDEX_PATH.exists() or not META_PATH.exists():
        # Non crashiamo l’app: risponde 503 finché non ci sono i file
        faiss_index = None
        meta_rows = []
        return

    try:
        faiss_index = faiss.read_index(str(INDEX_PATH))
    except Exception as e:
        faiss_index = None
        meta_rows = []
        raise RuntimeError(f"Cannot read FAISS index: {e}")

    meta_rows = _load_meta_jsonl(META_PATH)


# -------------------------
# Routes: UI + Photos
# -------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/photo/{filename}")
def get_photo(filename: str):
    # sicurezza base
    safe_name = Path(filename).name
    path = PHOTOS_DIR / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Photo not found")
    return FileResponse(path)


@app.get("/health")
def health():
    return {"ok": True}


# -------------------------
# Core: match selfie
# -------------------------
@app.post("/match_selfie")
async def match_selfie(
    file: UploadFile = File(...),
    top_k_faces: int = Query(80, ge=1, le=500),
    min_score: float = Query(0.08, ge=0.0, le=1.0),
):
    _ensure_ready()

    # read upload
    file_bytes = await file.read()
    img = _read_image_from_upload(file_bytes)

    # detect faces on selfie
    assert face_app is not None
    faces = face_app.get(img)
    if not faces:
        return JSONResponse({"ok": True, "count": 0, "results": []})

    # prendi il volto più grande
    faces_sorted = sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True
    )
    emb = faces_sorted[0].embedding.astype("float32")
    emb = _normalize(emb).reshape(1, -1)

    # search FAISS (IndexFlatIP => score ~ cosine se normalizzato)
    assert faiss_index is not None
    D, I = faiss_index.search(emb, top_k_faces)

    results: List[Dict[str, Any]] = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0:
            continue
        if score < float(min_score):
            continue

        if idx >= len(meta_rows):
            continue

        row = meta_rows[idx]
        photo_id = row.get("photo_id") or row.get("filename") or row.get("id")
        if not photo_id:
            continue

        results.append({
            "photo_id": str(photo_id),
            "score": float(score),
        })

    return {"ok": True, "count": len(results), "results": results}
    global _face_app
    if _face_app is not None:
        return _face_app

    from insightface.app import FaceAnalysis

    providers = ["CPUExecutionProvider"]
    model_name = os.getenv("INSIGHTFACE_MODEL", "buffalo_l")

    det = int(os.getenv("DET_SIZE", "640"))  # 640 buon compromesso su CPU
    fa = FaceAnalysis(name=model_name, providers=providers)
    fa.prepare(ctx_id=0, det_size=(det, det))

    _face_app = fa
    return _face_app


def _read_meta_rows(meta_path: Path) -> List[Dict[str, Any]]:
    if not meta_path.exists():
        raise RuntimeError(f"Meta non trovato: {meta_path}")

    txt = meta_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        raise RuntimeError("Meta vuoto.")

    # JSONL (1 record per riga)
    if "\n" in txt and txt.lstrip().startswith("{"):
        rows = []
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows

    # JSON lista
    data = json.loads(txt)
    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "rows" in data and isinstance(data["rows"], list):
        return data["rows"]

    raise RuntimeError("Formato meta non supportato (usa JSONL o JSON lista).")


def _load_faiss_index():
    global _faiss_index, _meta_rows
    if _faiss_index is not None and _meta_rows is not None:
        return _faiss_index, _meta_rows

    import faiss

    if not INDEX_PATH.exists():
        raise RuntimeError(
            f"Indice non trovato: {INDEX_PATH}. "
            f"Crea l’indice localmente e pusha data/faces.index + data/faces.meta.jsonl"
        )
    if not META_PATH.exists():
        raise RuntimeError(f"Meta non trovato: {META_PATH}")

    _faiss_index = faiss.read_index(str(INDEX_PATH))
    _meta_rows = _read_meta_rows(META_PATH)

    return _faiss_index, _meta_rows


def _img_to_rgb_np(b: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(b)).convert("RGB")
    return np.array(img)


def _pick_best_face(faces: list):
    # prende la faccia “più importante”: area bbox più grande
    best = None
    best_area = -1
    for f in faces:
        bbox = getattr(f, "bbox", None)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        area = float(max(0, x2 - x1) * max(0, y2 - y1))
        if area > best_area:
            best_area = area
            best = f
    return best


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def _safe_photo_path(photo_id: str) -> Path:
    # niente path traversal tipo ../../
    name = Path(photo_id).name
    return PHOTOS_DIR / name


def _guess_mime(path: Path) -> str:
    mt, _ = mimetypes.guess_type(str(path))
    return mt or "application/octet-stream"


# =========================
# ROUTES UI
# =========================
@app.get("/", response_class=HTMLResponse)
def home():
    """
    Serve la UI.
    Se hai static/index.html lo mostra.
    """
    index_html = STATIC_DIR / "index.html"
    if index_html.exists():
        return FileResponse(str(index_html), media_type="text/html; charset=utf-8")

    # fallback minimale se manca l’HTML
    return HTMLResponse(
        "<h2>UI non trovata</h2><p>Manca <b>backend/static/index.html</b></p>",
        status_code=200,
    )


@app.get("/health")
def health():
    return {
        "ok": True,
        "photos_dir": str(PHOTOS_DIR),
        "index_exists": INDEX_PATH.exists(),
        "meta_exists": META_PATH.exists(),
    }


# =========================
# SERVE FOTO
# =========================
@app.get("/photos/{photo_id}")
def get_photo(photo_id: str):
    path = _safe_photo_path(photo_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Photo not found")
    return FileResponse(str(path), media_type=_guess_mime(path))


# =========================
# DEBUG
# =========================
@app.get("/debug/photos")
def debug_photos():
    exists = PHOTOS_DIR.exists()
    items = []
    if exists:
        items = sorted([p.name for p in PHOTOS_DIR.iterdir() if p.is_file()])[:200]
    return {
        "photos_dir": str(PHOTOS_DIR),
        "exists": exists,
        "count": len(list(PHOTOS_DIR.iterdir())) if exists else 0,
        "sample": items,
    }


@app.get("/debug/index")
def debug_index():
    idx, meta = _load_faiss_index()
    return {
        "index_path": str(INDEX_PATH),
        "meta_path": str(META_PATH),
        "index_vectors": int(getattr(idx, "ntotal", -1)),
        "meta_rows": len(meta),
    }


# =========================
# MATCH ENDPOINT
# =========================
@app.post("/match_selfie")
async def match_selfie(
    file: UploadFile = File(...),
    top_k_faces: int = Query(80, ge=1, le=500),
    min_score: float = Query(0.30, ge=-1.0, le=1.0),
):
    """
    Ritorna foto_id ordinate per score decrescente.
    Score: inner product (cosine se embeddings normalizzati).
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")

    fa = _load_face_engine()
    idx, meta = _load_faiss_index()

    img = _img_to_rgb_np(raw)
    faces = fa.get(img)

    if not faces:
        return {"ok": True, "count": 0, "results": [], "reason": "no_face_detected"}

    best = _pick_best_face(faces)
    if best is None or getattr(best, "embedding", None) is None:
        return {"ok": True, "count": 0, "results": [], "reason": "no_embedding"}

    q = np.array(best.embedding, dtype=np.float32)
    q = _normalize(q).reshape(1, -1)

    # search
    D, I = idx.search(q, top_k_faces)

    # dedup per foto (tieni il punteggio migliore)
    best_by_photo: Dict[str, float] = {}
    for score, j in zip(D[0].tolist(), I[0].tolist()):
        if j < 0:
            continue
        if j >= len(meta):
            continue

        row = meta[j]
        photo_id = row.get("photo_id") or row.get("filename") or row.get("path")
        if not photo_id:
            continue

        photo_id = Path(photo_id).name  # solo nome file
        prev = best_by_photo.get(photo_id)
        if prev is None or float(score) > prev:
            best_by_photo[photo_id] = float(score)

    # filtra e ordina
    results = [
        {"photo_id": pid, "score": sc, "url": f"/photos/{pid}"}
        for pid, sc in best_by_photo.items()
        if sc >= float(min_score)
    ]
    results.sort(key=lambda x: x["score"], reverse=True)

    return {"ok": True, "count": len(results), "results": results}import os
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# prova prima ./photos (Render di solito qui), se non esiste usa ../photos (setup locale)
PHOTOS_DIR_CANDIDATE_1 = os.path.join(BASE_DIR, "photos")
PHOTOS_DIR_CANDIDATE_2 = os.path.join(BASE_DIR, "..", "photos")

PHOTOS_DIR = PHOTOS_DIR_CANDIDATE_1 if os.path.isdir(PHOTOS_DIR_CANDIDATE_1) else PHOTOS_DIR_CANDIDATE_2

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
app.mount("/photos", StaticFiles(directory=PHOTOS_DIR), name="photos")

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

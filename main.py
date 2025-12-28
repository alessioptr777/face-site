from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import cv2
import faiss

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from insightface.app import FaceAnalysis

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PHOTOS_DIR = BASE_DIR / "photos"
STATIC_DIR = BASE_DIR / "static"

INDEX_PATH = DATA_DIR / "faces.index"
META_PATH = DATA_DIR / "faces.meta.jsonl"
INDEX_DIM = 512

app = FastAPI(title="Face Match API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

face_app: Optional[FaceAnalysis] = None
faiss_index: Optional[faiss.Index] = None
meta_rows: List[Dict[str, Any]] = []

@app.get("/")
def root():
    return {"ok": True}

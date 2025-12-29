# File principale dell'API FaceSite
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import cv2
import faiss

from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
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

APP_VERSION = os.getenv("APP_VERSION", "0.1.0")
APP_NAME = os.getenv("APP_NAME", "FaceSite API")

# Configurazione logging
class UTCFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return dt.strftime(datefmt) if datefmt else dt.isoformat()

root_logger = logging.getLogger()
if not root_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(UTCFormatter("%(asctime)s %(levelname)s %(name)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S UTC"))
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)

logger = logging.getLogger(__name__)

# Crea cartelle necessarie all'avvio
PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Log diagnostico all'avvio
logger.info("=" * 60)
logger.info("STARTUP DIAGNOSTICS")
logger.info(f"BASE_DIR (absolute): {BASE_DIR.resolve()}")
logger.info(f"PHOTOS_DIR (absolute): {PHOTOS_DIR.resolve()}")
logger.info(f"PHOTOS_DIR exists: {PHOTOS_DIR.exists()}")
logger.info(f"Current working directory: {os.getcwd()}")
if PHOTOS_DIR.exists():
    try:
        photo_files = list(PHOTOS_DIR.iterdir())
        logger.info(f"Photos found: {len(photo_files)}")
        logger.info(f"First 10 files: {[p.name for p in photo_files[:10]]}")
    except Exception as e:
        logger.error(f"Error listing photos: {e}")
else:
    logger.warning("PHOTOS_DIR does not exist!")
logger.info("=" * 60)

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

# NON usare mount per /photo - usiamo endpoint esplicito per controllo migliore
# app.mount("/photo", StaticFiles(directory=str(PHOTOS_DIR)), name="photos")
logger.info(f"Will serve photos from: {PHOTOS_DIR.resolve()}")

face_app: Optional[FaceAnalysis] = None
faiss_index: Optional[faiss.Index] = None
meta_rows: List[Dict[str, Any]] = []

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTPException: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "code": exc.status_code
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception: {type(exc).__name__}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "code": "INTERNAL_ERROR"
            }
        }
    )


@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": APP_NAME,
        "version": APP_VERSION,
        "time_utc": datetime.now(timezone.utc).isoformat()
    }

@app.get("/debug/paths")
def debug_paths():
    """Endpoint di debug per verificare paths in produzione"""
    cwd = os.getcwd()
    photos_absolute = str(PHOTOS_DIR.resolve())
    photos_exists = PHOTOS_DIR.exists()
    photos_files = []
    if PHOTOS_DIR.exists():
        try:
            photos_files = [p.name for p in PHOTOS_DIR.iterdir() if p.is_file()][:20]
        except Exception as e:
            photos_files = [f"Error listing: {str(e)}"]
    
    return {
        "base_dir": str(BASE_DIR.resolve()),
        "photos_dir": str(PHOTOS_DIR),
        "photos_dir_absolute": photos_absolute,
        "photos_exists": photos_exists,
        "current_working_directory": cwd,
        "photos_files_count": len(photos_files) if isinstance(photos_files, list) else 0,
        "photos_files": photos_files,
        "static_dir": str(STATIC_DIR.resolve()),
        "static_exists": STATIC_DIR.exists(),
    }

@app.get("/photo/{filename:path}")
async def serve_photo(filename: str):
    """Endpoint per servire le foto"""
    # Decodifica il filename (potrebbe essere URL encoded)
    try:
        from urllib.parse import unquote
        filename = unquote(filename)
    except:
        pass
    
    photo_path = PHOTOS_DIR / filename
    
    # Sicurezza: previeni directory traversal
    try:
        resolved_path = photo_path.resolve()
        resolved_photos_dir = PHOTOS_DIR.resolve()
        resolved_path.relative_to(resolved_photos_dir)
    except (ValueError, OSError):
        logger.warning(f"Directory traversal attempt blocked: {filename}")
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Log per debug
    logger.info(f"Serving photo request: {filename} -> {resolved_path}")
    logger.info(f"File exists: {photo_path.exists()}, is_file: {photo_path.is_file() if photo_path.exists() else False}")
    
    if not photo_path.exists():
        logger.error(f"Photo not found: {filename} (path: {resolved_path})")
        raise HTTPException(status_code=404, detail=f"Photo not found: {filename}")
    
    if not photo_path.is_file():
        logger.error(f"Path is not a file: {filename} (path: {resolved_path})")
        raise HTTPException(status_code=404, detail=f"Photo not found: {filename}")
    
    return FileResponse(photo_path)

@app.post("/match_selfie")
async def match_selfie(
    selfie: UploadFile = File(...),
    top_k_faces: int = Query(120),
    min_score: float = Query(0.08)
):
    return {
        "ok": True,
        "filename": selfie.filename,
        "top_k_faces": top_k_faces,
        "min_score": min_score,
        "count": 0,
        "matches": []
    }

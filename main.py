# File principale dell'API FaceSite
import json
import logging
import os
import hashlib
import secrets
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import cv2
import faiss

from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw, ImageFont
import io

# Stripe per pagamenti
try:
    import stripe
    STRIPE_AVAILABLE = True
except ImportError:
    STRIPE_AVAILABLE = False

# Cloudinary per storage esterno (opzionale)
try:
    import cloudinary
    import cloudinary.api
    from cloudinary.utils import cloudinary_url
    CLOUDINARY_AVAILABLE = True
except ImportError:
    CLOUDINARY_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PHOTOS_DIR = BASE_DIR / "photos"
STATIC_DIR = BASE_DIR / "static"

INDEX_PATH = DATA_DIR / "faces.index"
META_PATH = DATA_DIR / "faces.meta.jsonl"
BACK_PHOTOS_PATH = DATA_DIR / "back_photos.jsonl"  # Foto senza volti (di spalle)
DOWNLOADS_TRACK_PATH = DATA_DIR / "downloads_track.jsonl"  # Tracking download per cleanup
INDEX_DIM = 512

# Configurazione cleanup download
DOWNLOAD_EXPIRY_DAYS = int(os.getenv("DOWNLOAD_EXPIRY_DAYS", "7"))  # Giorni prima di cancellare
MAX_DOWNLOADS_PER_PHOTO = int(os.getenv("MAX_DOWNLOADS_PER_PHOTO", "3"))  # Max download prima di cancellare

APP_VERSION = os.getenv("APP_VERSION", "0.1.0")
APP_NAME = os.getenv("APP_NAME", "TenerifePictures API")

# Sistema prezzi
def calculate_price(photo_count: int) -> int:
    """Calcola il prezzo in centesimi di euro in base al numero di foto"""
    if photo_count == 1:
        return 2000  # €20.00
    elif photo_count == 2:
        return 4000  # €40.00
    elif photo_count == 3:
        return 3500  # €35.00
    elif photo_count == 4:
        return 4000  # €40.00
    elif photo_count == 5:
        return 4500  # €45.00
    elif 6 <= photo_count <= 11:
        return 5000  # €50.00
    else:  # 12+
        return 6000  # €60.00

# Directory per ordini e download tokens
ORDERS_DIR = DATA_DIR / "orders"
ORDERS_DIR.mkdir(parents=True, exist_ok=True)

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

# Configurazione Stripe (dopo logger)
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
USE_STRIPE = STRIPE_AVAILABLE and bool(STRIPE_SECRET_KEY)

# Log diagnostico Stripe
logger.info(f"Stripe diagnostic: STRIPE_AVAILABLE={STRIPE_AVAILABLE}, STRIPE_SECRET_KEY present={bool(STRIPE_SECRET_KEY)}, STRIPE_SECRET_KEY length={len(STRIPE_SECRET_KEY) if STRIPE_SECRET_KEY else 0}")

if USE_STRIPE:
    stripe.api_key = STRIPE_SECRET_KEY
    logger.info("Stripe configured successfully")
else:
    if not STRIPE_AVAILABLE:
        logger.warning("Stripe not configured - stripe package not available")
    elif not STRIPE_SECRET_KEY:
        logger.warning("Stripe not configured - STRIPE_SECRET_KEY environment variable not set or empty")
    else:
        logger.warning("Stripe not configured - payment features disabled")

# Configurazione Cloudinary (opzionale) - dopo logger
CLOUDINARY_URL = os.getenv("CLOUDINARY_URL", "")
USE_CLOUDINARY = CLOUDINARY_AVAILABLE and bool(CLOUDINARY_URL)

if USE_CLOUDINARY:
    cloudinary.config()
    logger.info("Cloudinary configured - using external storage")
else:
    logger.info("Cloudinary not configured - using local file storage")

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
back_photos: List[Dict[str, Any]] = []  # Foto senza volti (di spalle)

# Funzioni helper
def _normalize(v: np.ndarray) -> np.ndarray:
    """Normalizza un vettore"""
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def _load_meta_jsonl(meta_path: Path) -> List[Dict[str, Any]]:
    """Carica i metadata dal file JSONL"""
    rows = []
    if not meta_path.exists():
        return rows
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
    return rows

def _track_download(photo_id: str):
    """Traccia un download per cleanup futuro"""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        download_record = {
            "photo_id": photo_id,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(DOWNLOADS_TRACK_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(download_record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Error tracking download: {e}")

def _cleanup_downloaded_photos():
    """Cancella foto scaricate dopo X giorni o dopo N download"""
    if not DOWNLOADS_TRACK_PATH.exists():
        return
    
    try:
        # Carica tutti i download
        downloads = _load_meta_jsonl(DOWNLOADS_TRACK_PATH)
        if not downloads:
            return
        
        # Raggruppa per photo_id
        photo_downloads: Dict[str, List[datetime]] = {}
        for dl in downloads:
            photo_id = dl.get("photo_id")
            if not photo_id:
                continue
            try:
                downloaded_at = datetime.fromisoformat(dl.get("downloaded_at", "").replace("Z", "+00:00"))
            except:
                continue
            
            if photo_id not in photo_downloads:
                photo_downloads[photo_id] = []
            photo_downloads[photo_id].append(downloaded_at)
        
        # Trova foto da cancellare
        now = datetime.now(timezone.utc)
        photos_to_delete = set()
        
        for photo_id, download_times in photo_downloads.items():
            # Cancella se scaricata più di MAX_DOWNLOADS_PER_PHOTO volte
            if len(download_times) >= MAX_DOWNLOADS_PER_PHOTO:
                photos_to_delete.add(photo_id)
                logger.info(f"Photo {photo_id} marked for deletion: {len(download_times)} downloads (max: {MAX_DOWNLOADS_PER_PHOTO})")
                continue
            
            # Cancella se prima download più vecchia di DOWNLOAD_EXPIRY_DAYS giorni
            oldest_download = min(download_times)
            days_ago = (now - oldest_download).days
            if days_ago >= DOWNLOAD_EXPIRY_DAYS:
                photos_to_delete.add(photo_id)
                logger.info(f"Photo {photo_id} marked for deletion: {days_ago} days old (max: {DOWNLOAD_EXPIRY_DAYS})")
        
        # Cancella foto
        deleted_count = 0
        for photo_id in photos_to_delete:
            photo_path = PHOTOS_DIR / photo_id
            if photo_path.exists() and photo_path.is_file():
                try:
                    photo_path.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted photo: {photo_id}")
                except Exception as e:
                    logger.error(f"Error deleting photo {photo_id}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleanup completed: {deleted_count} photos deleted")
        
        # Pulisci tracking file (rimuovi record di foto cancellate)
        if photos_to_delete:
            remaining_downloads = [
                dl for dl in downloads 
                if dl.get("photo_id") not in photos_to_delete
            ]
            with open(DOWNLOADS_TRACK_PATH, "w", encoding="utf-8") as f:
                for dl in remaining_downloads:
                    f.write(json.dumps(dl, ensure_ascii=False) + "\n")
    
    except Exception as e:
        logger.error(f"Error in cleanup: {e}")

def _read_image_from_bytes(file_bytes: bytes):
    """Legge un'immagine da bytes"""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")
    return img

def _add_watermark(image_path: Path) -> bytes:
    """Aggiunge watermark pattern ripetuto su tutta l'immagine"""
    try:
        # Apri immagine con Pillow
        img = Image.open(image_path)
        
        # Converti in RGB se necessario
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Converti in RGBA per watermark
        img_rgba = img.convert('RGBA')
        watermark = Image.new('RGBA', img_rgba.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)
        
        # Testo watermark
        text = "TENERIFEPICTURES"
        
        # Calcola dimensione font (circa 8% dell'altezza immagine per pattern più visibile)
        font_size = max(60, int(img.height * 0.08))
        
        try:
            # Prova a usare font di sistema
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                # Fallback a font default
                font = ImageFont.load_default()
        
        # Calcola dimensioni testo
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Spaziatura tra watermark (circa 1.5x la larghezza del testo)
        spacing_x = int(text_width * 1.5)
        spacing_y = int(text_height * 1.5)
        
        # Disegna pattern ripetuto su tutta l'immagine
        # Angolo di 45 gradi per pattern diagonale
        angle = 45
        opacity = 120  # Più trasparente ma più intenso (0-255)
        
        # Calcola quante volte ripetere
        num_repeats_x = int(img.width / spacing_x) + 2
        num_repeats_y = int(img.height / spacing_y) + 2
        
        # Disegna pattern
        for i in range(-num_repeats_y, num_repeats_y):
            for j in range(-num_repeats_x, num_repeats_x):
                x = j * spacing_x
                y = i * spacing_y
                
                # Offset per pattern diagonale
                offset_x = (i * spacing_x * 0.3)
                offset_y = (j * spacing_y * 0.3)
                
                x_pos = int(x + offset_x)
                y_pos = int(y + offset_y)
                
                # Disegna ombra leggera
                draw.text((x_pos + 1, y_pos + 1), text, font=font, fill=(0, 0, 0, opacity // 2))
                # Disegna testo principale (bianco semi-trasparente)
                draw.text((x_pos, y_pos), text, font=font, fill=(255, 255, 255, opacity))
        
        # Combina watermark con immagine
        img_with_watermark = Image.alpha_composite(img_rgba, watermark).convert('RGB')
        
        # Salva in bytes
        output = io.BytesIO()
        img_with_watermark.save(output, format='JPEG', quality=85)
        output.seek(0)
        return output.getvalue()
    
    except Exception as e:
        logger.error(f"Error adding watermark: {e}")
        # Fallback: ritorna immagine originale
        with open(image_path, 'rb') as f:
            return f.read()

def _ensure_ready():
    """Verifica che face_app e faiss_index siano caricati"""
    if face_app is None:
        raise HTTPException(status_code=503, detail="Face recognition not initialized")
    if faiss_index is None:
        raise HTTPException(status_code=503, detail="Face index not loaded")

@app.on_event("startup")
async def startup():
    """Carica il modello e l'indice all'avvio"""
    global face_app, faiss_index, meta_rows, back_photos
    
    logger.info("Loading face recognition model...")
    try:
        face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("Face recognition model loaded")
    except Exception as e:
        logger.error(f"Error loading face model: {e}")
        face_app = None
        return
    
    # Carica indice FAISS e metadata
    if not INDEX_PATH.exists() or not META_PATH.exists():
        logger.warning(f"Index files not found: {INDEX_PATH} or {META_PATH}")
        logger.warning("Face matching will not work until index is created")
        faiss_index = None
        meta_rows = []
        back_photos = []
        return
    
    try:
        logger.info(f"Loading FAISS index from {INDEX_PATH}")
        faiss_index = faiss.read_index(str(INDEX_PATH))
        logger.info(f"FAISS index loaded: {faiss_index.ntotal} vectors")
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        faiss_index = None
        meta_rows = []
        back_photos = []
        return
    
    try:
        logger.info(f"Loading metadata from {META_PATH}")
        meta_rows = _load_meta_jsonl(META_PATH)
        logger.info(f"Metadata loaded: {len(meta_rows)} records")
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        meta_rows = []
    
    # Carica foto di spalle (senza volti)
    try:
        if BACK_PHOTOS_PATH.exists():
            logger.info(f"Loading back photos from {BACK_PHOTOS_PATH}")
            back_photos = _load_meta_jsonl(BACK_PHOTOS_PATH)
            logger.info(f"Back photos loaded: {len(back_photos)} records")
        else:
            logger.info("Back photos file not found - skipping")
            back_photos = []
    except Exception as e:
        logger.error(f"Error loading back photos: {e}")
        back_photos = []
    
    # Esegui cleanup all'avvio
    logger.info("Running initial cleanup...")
    _cleanup_downloaded_photos()

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

@app.get("/album", response_class=HTMLResponse)
def album():
    """Pagina album con i risultati delle foto"""
    return FileResponse(STATIC_DIR / "album.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": APP_NAME,
        "version": APP_VERSION,
        "time_utc": datetime.now(timezone.utc).isoformat()
    }

@app.post("/admin/cleanup")
async def cleanup_downloads():
    """Endpoint admin per eseguire cleanup manuale delle foto scaricate"""
    try:
        _cleanup_downloaded_photos()
        return {
            "ok": True,
            "message": "Cleanup completed"
        }
    except Exception as e:
        logger.error(f"Error in manual cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup error: {str(e)}")

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
async def serve_photo(
    filename: str, 
    request: Request,
    paid: bool = Query(False, description="Se true, serve foto senza watermark (solo se pagata)")
):
    """Endpoint per servire le foto - con watermark se non pagata"""
    logger.info(f"=== PHOTO REQUEST ===")
    logger.info(f"Request path: {request.url.path}")
    logger.info(f"Filename parameter: {filename}")
    logger.info(f"Paid: {paid}")
    
    # Decodifica il filename (potrebbe essere URL encoded)
    try:
        from urllib.parse import unquote
        filename = unquote(filename)
        logger.info(f"Decoded filename: {filename}")
    except Exception as e:
        logger.warning(f"Error decoding filename: {e}")
    
    # Rimuovi estensione per Cloudinary (se presente)
    filename_no_ext = filename.rsplit('.', 1)[0] if '.' in filename else filename
    
    # Se Cloudinary è configurato, prova a servire da lì
    if USE_CLOUDINARY:
        try:
            # Cloudinary usa il filename senza estensione come public_id
            url, options = cloudinary_url(
                filename_no_ext,
                format="jpg",
                secure=True
            )
            logger.info(f"Cloudinary URL generated: {url}")
            
            # Verifica che l'immagine esista su Cloudinary
            try:
                cloudinary.api.resource(filename_no_ext)
                logger.info(f"Photo found on Cloudinary: {filename_no_ext}")
                # Redirect al CDN di Cloudinary
                return RedirectResponse(url=url, status_code=302)
            except cloudinary.api.NotFound:
                logger.warning(f"Photo not found on Cloudinary: {filename_no_ext}, falling back to local")
                # Fallback a file locale
            except Exception as e:
                logger.warning(f"Cloudinary error: {e}, falling back to local")
        except Exception as e:
            logger.warning(f"Cloudinary error: {e}, falling back to local storage")
    
    # Fallback: servire da file locale
    photo_path = PHOTOS_DIR / filename
    logger.info(f"Photo path (local): {photo_path}")
    
    # Sicurezza: previeni directory traversal
    try:
        resolved_path = photo_path.resolve()
        resolved_photos_dir = PHOTOS_DIR.resolve()
        relative_path = resolved_path.relative_to(resolved_photos_dir)
        logger.info(f"Resolved path: {resolved_path}")
        logger.info(f"Relative path check OK: {relative_path}")
    except (ValueError, OSError) as e:
        logger.error(f"Directory traversal attempt blocked: {filename} - {e}")
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Log per debug
    logger.info(f"Checking file: exists={photo_path.exists()}, is_file={photo_path.is_file() if photo_path.exists() else False}")
    
    if not photo_path.exists():
        logger.error(f"Photo not found: {filename}")
        logger.error(f"Full path checked: {resolved_path}")
        # Lista i file disponibili per debug
        available_files = [p.name for p in PHOTOS_DIR.iterdir() if p.is_file()][:10]
        logger.error(f"Available files (first 10): {available_files}")
        logger.error(f"Looking for exact match of: '{filename}'")
        # Prova a trovare file simili
        similar = [f for f in available_files if filename.lower() in f.lower() or f.lower() in filename.lower()]
        if similar:
            logger.error(f"Similar files found: {similar}")
        raise HTTPException(status_code=404, detail=f"Photo not found: {filename}")
    
    if not photo_path.is_file():
        logger.error(f"Path is not a file: {filename}")
        raise HTTPException(status_code=404, detail=f"Photo not found: {filename}")
    
    # Se non pagata, aggiungi watermark
    if not paid:
        logger.info(f"Serving photo with watermark: {filename}")
        watermarked_bytes = _add_watermark(photo_path)
        return Response(content=watermarked_bytes, media_type="image/jpeg")
    
    # Se pagata, serve originale (ma traccia download solo se pagata)
    logger.info(f"Returning original file (paid): {resolved_path}")
    _track_download(filename)
    
    return FileResponse(resolved_path)

@app.post("/match_selfie")
async def match_selfie(
    selfie: UploadFile = File(...),
    top_k_faces: int = Query(120),
    min_score: float = Query(0.08),
    tour_date: Optional[str] = Query(None, description="Data del tour (YYYY-MM-DD) per filtrare foto di spalle")
):
    """Endpoint per il face matching: trova foto simili al selfie + foto di spalle se tour_date fornita"""
    _ensure_ready()
    
    try:
        # Leggi l'immagine dal selfie
        file_bytes = await selfie.read()
        img = _read_image_from_bytes(file_bytes)
        
        # Rileva i volti nel selfie
        assert face_app is not None
        faces = face_app.get(img)
        
        # Mostriamo TUTTE le foto: matchate (con volti) + foto di spalle/ombra/silhouette (senza volti)
        matched_results: List[Dict[str, Any]] = []
        seen_photos = set()  # Evita duplicati
        
        if faces:
            # Prendi il volto più grande (presumibilmente il selfie principale)
            faces_sorted = sorted(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True
            )
            
            # Estrai embedding e normalizza
            emb = faces_sorted[0].embedding.astype("float32")
            emb = _normalize(emb).reshape(1, -1)
            
            # Cerca nell'indice FAISS (IndexFlatIP => score ~ cosine similarity se normalizzato)
            assert faiss_index is not None
            D, I = faiss_index.search(emb, top_k_faces)
            
            # Costruisci i risultati delle foto matchate
            for score, idx in zip(D[0].tolist(), I[0].tolist()):
                if idx < 0 or idx >= len(meta_rows):
                    continue
                if score < float(min_score):
                    continue
                
                row = meta_rows[idx]
                photo_id = row.get("photo_id") or row.get("filename") or row.get("id")
                if not photo_id:
                    continue
                
                # Evita duplicati (stessa foto con facce diverse)
                if photo_id in seen_photos:
                    continue
                seen_photos.add(photo_id)
                
                matched_results.append({
                    "photo_id": str(photo_id),
                    "score": float(score),
                    "has_face": True,
                })
                logger.debug(f"Added matched result: photo_id={photo_id}, score={score:.4f}")
        
        # Aggiungi foto di spalle/ombra/silhouette (senza volti visibili)
        back_results: List[Dict[str, Any]] = []
        
        # Se tour_date fornita, filtra per data, altrimenti mostra tutte le foto di spalle
        if tour_date:
            # Normalizza formato data (accetta YYYY-MM-DD o YYYYMMDD)
            normalized_date = tour_date.replace("-", "") if "-" not in tour_date else tour_date
            if len(normalized_date) == 8:  # YYYYMMDD
                normalized_date = f"{normalized_date[:4]}-{normalized_date[4:6]}-{normalized_date[6:8]}"
            
            seen_back_photos = set()
            for back_photo in back_photos:
                photo_id = back_photo.get("photo_id")
                if not photo_id:
                    continue
                
                if photo_id in seen_back_photos:
                    continue
                seen_back_photos.add(photo_id)
                
                # Evita duplicati con foto già matchate
                if photo_id in seen_photos:
                    continue
                seen_photos.add(photo_id)
                
                # Filtra per data del tour
                photo_tour_date = back_photo.get("tour_date")
                if photo_tour_date and normalized_date in photo_tour_date:
                    back_results.append({
                        "photo_id": str(photo_id),
                        "score": 0.0,  # Foto di spalle non hanno score di matching
                        "has_face": False,
                        "is_back_photo": True,
                    })
        else:
            # Senza tour_date, mostra tutte le foto di spalle/ombra/silhouette
            seen_back_photos = set()
            for back_photo in back_photos:
                photo_id = back_photo.get("photo_id")
                if not photo_id:
                    continue
                
                if photo_id in seen_back_photos:
                    continue
                seen_back_photos.add(photo_id)
                
                # Evita duplicati con foto già matchate
                if photo_id in seen_photos:
                    continue
                seen_photos.add(photo_id)
                
                back_results.append({
                    "photo_id": str(photo_id),
                    "score": 0.0,
                    "has_face": False,
                    "is_back_photo": True,
                })
        
        # Combina risultati: prima foto matchate, poi foto di spalle
        all_results = matched_results + back_results
        
        logger.info(f"Match completed: {len(matched_results)} matched photos, {len(back_results)} back photos")
        if all_results:
            logger.info(f"First 3 photo_ids: {[r['photo_id'] for r in all_results[:3]]}")
        
        return {
            "ok": True,
            "count": len(all_results),
            "matches": all_results,  # Per compatibilità
            "results": all_results,    # Per compatibilità con frontend
            "matched_count": len(matched_results),
            "back_photos_count": len(back_results),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in match_selfie: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# ========== CARRELLO E PAGAMENTI ==========

# Storage carrelli in memoria (in produzione usa Redis o database)
carts: Dict[str, List[str]] = {}  # session_id -> [photo_ids]

def _get_cart(session_id: str) -> List[str]:
    """Ottiene il carrello per una sessione"""
    return carts.get(session_id, [])

def _add_to_cart(session_id: str, photo_id: str):
    """Aggiunge foto al carrello"""
    if session_id not in carts:
        carts[session_id] = []
    if photo_id not in carts[session_id]:
        carts[session_id].append(photo_id)

def _remove_from_cart(session_id: str, photo_id: str):
    """Rimuove foto dal carrello"""
    if session_id in carts:
        carts[session_id] = [p for p in carts[session_id] if p != photo_id]

def _clear_cart(session_id: str):
    """Svuota il carrello"""
    if session_id in carts:
        del carts[session_id]

@app.get("/cart")
async def get_cart(session_id: str = Query(..., description="ID sessione")):
    """Ottiene il contenuto del carrello"""
    photo_ids = _get_cart(session_id)
    price = calculate_price(len(photo_ids))
    
    return {
        "ok": True,
        "photo_ids": photo_ids,
        "count": len(photo_ids),
        "price_cents": price,
        "price_euros": price / 100.0
    }

@app.post("/cart/add")
async def add_to_cart(
    session_id: str = Query(..., description="ID sessione"),
    photo_id: str = Query(..., description="ID foto da aggiungere")
):
    """Aggiunge una foto al carrello"""
    _add_to_cart(session_id, photo_id)
    photo_ids = _get_cart(session_id)
    price = calculate_price(len(photo_ids))
    
    return {
        "ok": True,
        "photo_ids": photo_ids,
        "count": len(photo_ids),
        "price_cents": price,
        "price_euros": price / 100.0
    }

@app.delete("/cart/remove")
async def remove_from_cart(
    session_id: str = Query(..., description="ID sessione"),
    photo_id: str = Query(..., description="ID foto da rimuovere")
):
    """Rimuove una foto dal carrello"""
    _remove_from_cart(session_id, photo_id)
    photo_ids = _get_cart(session_id)
    price = calculate_price(len(photo_ids)) if photo_ids else 0
    
    return {
        "ok": True,
        "photo_ids": photo_ids,
        "count": len(photo_ids),
        "price_cents": price,
        "price_euros": price / 100.0
    }

@app.post("/checkout")
async def create_checkout(
    request: Request,
    session_id: str = Query(..., description="ID sessione")
):
    """Crea una sessione di checkout Stripe"""
    if not USE_STRIPE:
        raise HTTPException(status_code=503, detail="Stripe not configured")
    
    photo_ids = _get_cart(session_id)
    if not photo_ids:
        raise HTTPException(status_code=400, detail="Cart is empty")
    
    price_cents = calculate_price(len(photo_ids))
    
    try:
        # Costruisci URL base
        base_url = str(request.base_url).rstrip('/')
        
        # Crea checkout session Stripe
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'eur',
                    'product_data': {
                        'name': f'{len(photo_ids)} foto da TenerifePictures',
                        'description': f'Download di {len(photo_ids)} foto in alta qualità',
                    },
                    'unit_amount': price_cents,
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=f'{base_url}/checkout/success?session_id={{CHECKOUT_SESSION_ID}}&cart_session={session_id}',
            cancel_url=f'{base_url}/checkout/cancel?session_id={session_id}',
            metadata={
                'session_id': session_id,
                'photo_ids': ','.join(photo_ids),
                'photo_count': str(len(photo_ids)),
            }
        )
        
        return {
            "ok": True,
            "checkout_url": checkout_session.url,
            "session_id": checkout_session.id
        }
    except Exception as e:
        logger.error(f"Error creating Stripe checkout: {e}")
        raise HTTPException(status_code=500, detail=f"Checkout error: {str(e)}")

@app.get("/checkout/success")
async def checkout_success(
    session_id: str = Query(..., description="Stripe session ID"),
    cart_session: str = Query(..., description="Cart session ID")
):
    """Pagina di successo dopo pagamento"""
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pagamento completato - TenerifePictures</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
                margin: 0;
                background: linear-gradient(135deg, #7b74ff, #5f58ff);
                color: #fff;
            }}
            .container {{
                text-align: center;
                padding: 40px;
                background: rgba(255,255,255,0.1);
                border-radius: 20px;
                backdrop-filter: blur(10px);
            }}
            h1 {{ font-size: 32px; margin: 0 0 20px; }}
            p {{ font-size: 18px; margin: 10px 0; }}
            a {{
                display: inline-block;
                margin-top: 20px;
                padding: 12px 24px;
                background: #fff;
                color: #5f58ff;
                text-decoration: none;
                border-radius: 8px;
                font-weight: 600;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>✅ Pagamento completato!</h1>
            <p>Le tue foto sono state sbloccate.</p>
            <p>Puoi scaricarle ora dalla pagina principale.</p>
            <a href="/">Torna alla home</a>
        </div>
    </body>
    </html>
    """)

@app.get("/checkout/cancel")
async def checkout_cancel(session_id: str = Query(..., description="Session ID")):
    """Pagina di annullamento pagamento"""
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pagamento annullato - TenerifePictures</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
                margin: 0;
                background: linear-gradient(135deg, #7b74ff, #5f58ff);
                color: #fff;
            }}
            .container {{
                text-align: center;
                padding: 40px;
                background: rgba(255,255,255,0.1);
                border-radius: 20px;
                backdrop-filter: blur(10px);
            }}
            h1 {{ font-size: 32px; margin: 0 0 20px; }}
            p {{ font-size: 18px; margin: 10px 0; }}
            a {{
                display: inline-block;
                margin-top: 20px;
                padding: 12px 24px;
                background: #fff;
                color: #5f58ff;
                text-decoration: none;
                border-radius: 8px;
                font-weight: 600;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>❌ Pagamento annullato</h1>
            <p>Il pagamento è stato annullato.</p>
            <a href="/">Torna alla home</a>
        </div>
    </body>
    </html>
    """)

@app.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    """Webhook Stripe per confermare pagamenti"""
    if not USE_STRIPE:
        raise HTTPException(status_code=503, detail="Stripe not configured")
    
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Gestisci evento checkout completato
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        metadata = session.get('metadata', {})
        session_id = metadata.get('session_id')
        photo_ids_str = metadata.get('photo_ids', '')
        
        if session_id and photo_ids_str:
            photo_ids = photo_ids_str.split(',')
            
            # Salva ordine
            order_id = session.get('id')
            order_data = {
                'order_id': order_id,
                'stripe_session_id': session.get('id'),
                'session_id': session_id,
                'photo_ids': photo_ids,
                'amount_cents': session.get('amount_total', 0),
                'paid_at': datetime.now(timezone.utc).isoformat(),
                'status': 'paid'
            }
            
            order_file = ORDERS_DIR / f"{order_id}.json"
            with open(order_file, 'w', encoding='utf-8') as f:
                json.dump(order_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Order completed: {order_id} - {len(photo_ids)} photos")
    
    return {"ok": True}

@app.get("/download/{photo_id:path}")
async def download_photo(
    photo_id: str,
    order_id: str = Query(..., description="ID ordine Stripe"),
    session_id: str = Query(..., description="ID sessione carrello")
):
    """Download protetto di una foto (solo dopo pagamento)"""
    # Verifica che l'ordine esista e sia pagato
    order_file = ORDERS_DIR / f"{order_id}.json"
    if not order_file.exists():
        raise HTTPException(status_code=404, detail="Order not found")
    
    try:
        with open(order_file, 'r', encoding='utf-8') as f:
            order_data = json.load(f)
        
        if order_data.get('status') != 'paid':
            raise HTTPException(status_code=403, detail="Order not paid")
        
        # Verifica che la foto sia nell'ordine
        photo_ids = order_data.get('photo_ids', [])
        if photo_id not in photo_ids:
            raise HTTPException(status_code=403, detail="Photo not in order")
        
        # Verifica che la sessione corrisponda
        if order_data.get('session_id') != session_id:
            raise HTTPException(status_code=403, detail="Invalid session")
        
        # Servi foto originale (senza watermark)
        photo_path = PHOTOS_DIR / photo_id
        if not photo_path.exists():
            raise HTTPException(status_code=404, detail="Photo not found")
        
        # Traccia download
        _track_download(photo_id)
        
        return FileResponse(photo_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading photo: {e}")
        raise HTTPException(status_code=500, detail=f"Download error: {str(e)}")

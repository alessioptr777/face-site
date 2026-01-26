# File principale dell'API FaceSite
# BUILD_VERSION: 2026-01-23-SUCCESS-PAGE-FIX
# FORCE_RELOAD: Questo commento forza Render a ricompilare il file
APP_BUILD_ID = "deploy-2026-01-23-test-mode-fix-v2"

# Carica variabili d'ambiente da .env (PRIMA di qualsiasi os.getenv)
from pathlib import Path
from dotenv import load_dotenv

# Calcola PROJECT_ROOT come la cartella padre di /backend
PROJECT_ROOT = Path(__file__).resolve().parents[1]
dotenv_path = PROJECT_ROOT / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Loaded .env from {dotenv_path}")
else:
    print(f".env not found at {dotenv_path}, using system environment variables")

import json
import logging
import os
import hashlib
import secrets
import math
import asyncio
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

try:
    import asyncpg  # type: ignore
except Exception:
    asyncpg = None  # type: ignore

import numpy as np
import cv2
import faiss

from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Request, Form, Body, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response, RedirectResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import io

# Inizializza logger PRIMA di qualsiasi uso
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend.main")

# Stripe per pagamenti
try:
    import stripe
    STRIPE_AVAILABLE = True
except ImportError:
    STRIPE_AVAILABLE = False

# Cloudflare R2 (S3 compatible) per storage esterno (opzionale)
try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    ClientError = None


# Database: PostgreSQL (opzionale - stateless mode)
try:
    # asyncpg puede no estar instalado en algunos entornos
    POSTGRES_AVAILABLE = asyncpg is not None
except Exception:
    POSTGRES_AVAILABLE = False

# Verifica configurazione PostgreSQL (opzionale)
DATABASE_URL = os.getenv("DATABASE_URL")
# Supporta anche DATABASE_URL che inizia con postgres:// (senza 'ql')
USE_POSTGRES = POSTGRES_AVAILABLE and DATABASE_URL is not None and (
    DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://")
)

# Stateless mode: se DATABASE_URL manca, l'app funziona comunque
if not USE_POSTGRES:
    logger.warning("DB disabled (stateless mode): DATABASE_URL not configured or asyncpg unavailable")

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent  # Root del repository
DATA_DIR = BASE_DIR / "data"
PHOTOS_DIR = BASE_DIR / "photos"
STATIC_DIR = REPO_ROOT / "static"  # Static files dalla root del repo

# R2_ONLY_MODE: disabilita completamente filesystem per foto e index
R2_ONLY_MODE = os.getenv("R2_ONLY_MODE", "1") == "1"

# R2_PHOTOS_PREFIX: prefisso per le foto su R2 (default vuoto o "photos/")
R2_PHOTOS_PREFIX = os.getenv("R2_PHOTOS_PREFIX", "")

# Configurazione indexing automatico
# Accetta "1", "true", "yes", "on" come valori validi (case-insensitive)
_indexing_enabled_str = os.getenv("INDEXING_ENABLED", "1").strip().lower()
INDEXING_ENABLED = _indexing_enabled_str in {"1", "true", "yes", "on"}
# Intervallo più breve (60s) per sync più reattivo - rileva cambiamenti R2 velocemente
INDEXING_INTERVAL_SECONDS = int(os.getenv("INDEXING_INTERVAL_SECONDS", "60"))

# Path per file index/tracking (disabilitati in R2_ONLY_MODE)
INDEX_PATH = DATA_DIR / "faces.index"
META_PATH = DATA_DIR / "faces.meta.jsonl"
DOWNLOADS_TRACK_PATH = DATA_DIR / "downloads_track.jsonl"  # Tracking download per cleanup
INDEX_DIM = 512

# Soglia per filtrare facce "deboli" durante il filtro foto
# Facce con det_score < questa soglia vengono ignorate nel filtro issubset(valid_faces)
MIN_FACE_DET_SCORE_FOR_FILTER = float(os.getenv("MIN_FACE_DET_SCORE_FOR_FILTER", "0.75"))

# Configurazione cleanup download
DOWNLOAD_EXPIRY_DAYS = int(os.getenv("DOWNLOAD_EXPIRY_DAYS", "7"))  # Giorni prima di cancellare
MAX_DOWNLOADS_PER_PHOTO = int(os.getenv("MAX_DOWNLOADS_PER_PHOTO", "3"))  # Max download prima di cancellare

# Configurazione family members
MAX_FAMILY_MEMBERS = int(os.getenv("MAX_FAMILY_MEMBERS", "8"))  # Max membri famiglia per email

APP_VERSION = os.getenv("APP_VERSION", "0.1.0")
APP_NAME = os.getenv("APP_NAME", "TenerifePictures API")

# Sistema prezzi
def calculate_price(photo_count: int) -> int:
    """Calcola il prezzo in centesimi di euro in base al numero di foto"""
    # TEST MODE: tutte le foto costano 50 centesimi (minimo Stripe)
    return 50  # €0.50 (TEST MODE - minimo Stripe, cambiare per produzione)
    
    # Prezzi produzione (commentati):
    # if photo_count == 1:
    #     return 2000  # €20.00
    # elif photo_count == 2:
    #     return 4000  # €40.00
    # elif photo_count == 3:
    #     return 3500  # €35.00
    # elif photo_count == 4:
    #     return 4000  # €40.00
    # elif photo_count == 5:
    #     return 4500  # €45.00
    # elif 6 <= photo_count <= 11:
    #     return 5000  # €50.00
    # else:  # 12+
    #     return 6000  # €60.00

def _build_checkout_metadata(session_id: Optional[str], email: Optional[str], photo_ids: List[str]) -> Dict[str, str]:
    """Costruisce metadata per Stripe checkout, gestendo limite 500 caratteri.
    
    Se la stringa photo_ids supera 450 caratteri, salva in memoria e usa token.
    """
    photo_ids_str = ','.join(photo_ids)
    use_token = len(photo_ids_str) > 450
    
    metadata = {
        'photo_count': str(len(photo_ids)),
    }
    
    if session_id:
        metadata['session_id'] = session_id
    if email:
        metadata['email'] = email
    
    if use_token:
        # Genera token univoco e salva photo_ids in memoria
        token = secrets.token_urlsafe(32)
        checkout_photo_ids[token] = photo_ids
        logger.info(f"Photo IDs too long ({len(photo_ids_str)} chars), using token: {token[:16]}...")
        metadata['photo_ids_token'] = token
    else:
        # Usa lista diretta nei metadata (retrocompatibilità)
        metadata['photo_ids'] = photo_ids_str
        logger.info(f"Photo IDs fit in metadata ({len(photo_ids_str)} chars)")
    
    return metadata

# Directory per ordini e download tokens
ORDERS_DIR = DATA_DIR / "orders"
ORDERS_DIR.mkdir(parents=True, exist_ok=True)

# Pool di connessioni PostgreSQL (inizializzato all'avvio)
db_pool = None  # type: ignore

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
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")

# Attiva TEST MODE automaticamente in localhost se Stripe non è configurato
STRIPE_TEST_MODE_ENV = os.getenv("STRIPE_TEST_MODE", "0") == "1"
# Se non è esplicitamente disabilitato e Stripe non è disponibile, attiva TEST MODE in localhost
if not STRIPE_TEST_MODE_ENV and not USE_STRIPE:
    # Rileva se siamo in localhost controllando PUBLIC_BASE_URL
    is_localhost = "localhost" in PUBLIC_BASE_URL or "127.0.0.1" in PUBLIC_BASE_URL or "0.0.0.0" in PUBLIC_BASE_URL
    if is_localhost:
        STRIPE_TEST_MODE = True
        logger.info("🔧 Auto-enabled STRIPE_TEST_MODE for localhost (Stripe not configured)")
    else:
        STRIPE_TEST_MODE = False
else:
    STRIPE_TEST_MODE = STRIPE_TEST_MODE_ENV

# Storage per sessioni test (solo in test mode)
# Chiave: session_id, Valore: {"photo_ids": [...], "customer_email": "..."}
test_sessions: Dict[str, Dict[str, Any]] = {}

# Storage per photo_ids dei checkout Stripe (per evitare limite 500 caratteri nei metadata)
# Chiave: token univoco, Valore: List[str] di photo_ids
checkout_photo_ids: Dict[str, List[str]] = {}

# Admin token per endpoint protetti
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

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

logger.info(
    "R2 startup check: BOTO3_AVAILABLE=%s | env_endpoint_present=%s env_endpoint_len=%s | env_bucket_present=%s | env_access_key_present=%s env_access_key_len=%s | env_secret_present=%s",
    BOTO3_AVAILABLE,
    bool(os.getenv("R2_ENDPOINT_URL") or os.getenv("R2_ENDPOINT") or os.getenv("S3_ENDPOINT_URL")),
    len((os.getenv("R2_ENDPOINT_URL") or os.getenv("R2_ENDPOINT") or os.getenv("S3_ENDPOINT_URL") or "")),
    bool(os.getenv("R2_BUCKET")),
    bool(os.getenv("R2_ACCESS_KEY_ID")),
    len((os.getenv("R2_ACCESS_KEY_ID") or "")),
    bool(os.getenv("R2_SECRET_ACCESS_KEY")),
)

# Configurazione Cloudflare R2 (S3 compatible) - dopo logger
# Variabili standardizzate: R2_ENDPOINT_URL, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "").strip()
R2_PUBLIC_BASE_URL = os.getenv("R2_PUBLIC_BASE_URL", "").strip()  # URL pubblico R2 (es: https://pub-xxxxx.r2.dev o cdn.metaproos.com) - OBBLIGATORIO in produzione
# Supporto per variabili legacy (alias)
if not R2_ENDPOINT_URL:
    R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT", "").strip()
if not R2_ENDPOINT_URL:
    R2_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "").strip()

R2_BUCKET = os.getenv("R2_BUCKET", "").strip()
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "").strip()
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()

# Photo assets configuration (thumb and watermarked preview)
THUMB_PREFIX = "thumbs/"
WM_PREFIX = "wm/"
THUMB_MAX_SIDE = 600  # Lato lungo per thumbnail (griglia)
THUMB_QUALITY = 70  # JPEG quality 65-75
WM_MAX_SIDE = 2000  # Lato lungo per watermarked preview (zoom) - 2000px per zoom più bello
WM_QUALITY = 75  # JPEG quality 70-80

# Flag per abilitare generazione runtime (solo per debug/emergenze)
ENABLE_RUNTIME_PREVIEW_GENERATION = os.getenv("ENABLE_RUNTIME_PREVIEW_GENERATION", "0") == "1"

# Controllo e pulizia endpoint: rimuovi path se presente
if R2_ENDPOINT_URL:
    from urllib.parse import urlparse, urlunparse
    parsed = urlparse(R2_ENDPOINT_URL)
    if "/metaproos" in parsed.path.lower() or "/photos" in parsed.path.lower() or "/bucket" in parsed.path.lower():
        logger.warning(f"R2_ENDPOINT_URL contains path component: {parsed.path}. Removing path, keeping only base URL.")
        # Ricostruisci URL senza path
        cleaned = urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))
        R2_ENDPOINT_URL = cleaned
        logger.info(f"Cleaned R2_ENDPOINT_URL: {R2_ENDPOINT_URL}")

USE_R2 = BOTO3_AVAILABLE and bool(R2_ENDPOINT_URL) and bool(R2_BUCKET) and bool(R2_ACCESS_KEY_ID) and bool(R2_SECRET_ACCESS_KEY)

logger.info(
    "R2 resolved config: endpoint_present=%s endpoint_len=%s | bucket_present=%s bucket_len=%s | USE_R2=%s",
    bool(R2_ENDPOINT_URL),
    len(R2_ENDPOINT_URL or ""),
    bool(R2_BUCKET),
    len(R2_BUCKET or ""),
    USE_R2,
)

# Inizializza client S3/R2 se configurato
r2_client = None
if USE_R2:
    try:
        r2_client = boto3.client(
            's3',
            endpoint_url=R2_ENDPOINT_URL,  # Solo base URL, senza path
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            config=Config(signature_version='s3v4')
        )
        logger.info("R2 client created OK (no credentials printed). Now testing connection...")
        # Test connessione: verifica accesso al bucket specifico (più affidabile di list_buckets su R2)
        try:
            r2_client.head_bucket(Bucket=R2_BUCKET)
            masked_key = ("****" + R2_ACCESS_KEY_ID[-4:]) if R2_ACCESS_KEY_ID else "missing"
            logger.info(
                "R2 configured successfully - "
                f"endpoint={R2_ENDPOINT_URL}, bucket={R2_BUCKET}, access_key_id={masked_key}"
            )
        except Exception as e:
            # Prova a estrarre info dall'errore boto3 (AccessDenied, NoSuchBucket, ecc.)
            err_code = None
            err_msg = str(e)
            try:
                err_code = getattr(e, "response", {}).get("Error", {}).get("Code")
                err_msg2 = getattr(e, "response", {}).get("Error", {}).get("Message")
                if err_msg2:
                    err_msg = err_msg2
            except Exception:
                pass

            masked_key = ("****" + R2_ACCESS_KEY_ID[-4:]) if R2_ACCESS_KEY_ID else "missing"
            logger.warning(
                "R2 connection test failed (head_bucket). "
                f"code={err_code}, message={err_msg}, endpoint={R2_ENDPOINT_URL}, bucket={R2_BUCKET}, access_key_id={masked_key}"
            )
            logger.warning("R2 will be disabled. Check R2 token permissions for this bucket and the endpoint URL.")
            USE_R2 = False
            r2_client = None
    except Exception as e:
        logger.error(f"Error initializing R2 client: {e}")
        USE_R2 = False
        r2_client = None
else:
    logger.warning("=" * 80)
    logger.warning("⚠️  R2 NOT CONFIGURED - Indexing will not work!")
    logger.warning("=" * 80)
    if not BOTO3_AVAILABLE:
        logger.warning("❌ R2 not configured - boto3 package not available")
    elif not R2_ENDPOINT_URL:
        logger.warning("❌ R2 not configured - R2_ENDPOINT_URL not set")
    elif not R2_BUCKET:
        logger.warning("❌ R2 not configured - R2_BUCKET not set")
    elif not R2_ACCESS_KEY_ID:
        logger.warning("❌ R2 not configured - R2_ACCESS_KEY_ID not set")
    elif not R2_SECRET_ACCESS_KEY:
        logger.warning("❌ R2 not configured - R2_SECRET_ACCESS_KEY not set")
    else:
        logger.warning("❌ R2 not configured - using local file storage")
    logger.warning(f"Current values: BOTO3_AVAILABLE={BOTO3_AVAILABLE}, R2_ENDPOINT_URL present={bool(R2_ENDPOINT_URL)}, R2_BUCKET present={bool(R2_BUCKET)}, R2_ACCESS_KEY_ID present={bool(R2_ACCESS_KEY_ID)}, R2_SECRET_ACCESS_KEY present={bool(R2_SECRET_ACCESS_KEY)}")
    logger.warning("=" * 80)
    logger.info("R2 final status: USE_R2=%s | r2_client_is_none=%s", USE_R2, r2_client is None)

logger.info(
    "R2 final status: "
    f"BOTO3_AVAILABLE={BOTO3_AVAILABLE}, USE_R2={USE_R2}, endpoint_set={bool(R2_ENDPOINT_URL)}, bucket_set={bool(R2_BUCKET)}"
)

# Log diagnostico R2 (dopo configurazione completa)
# Log diagnostico R2 (dopo configurazione completa)
# NOTA: R2_SECRET_ACCESS_KEY non viene mai loggato (né valore né lunghezza) per sicurezza
logger.info(
    "R2 diagnostic: BOTO3_AVAILABLE=%s, R2_ENDPOINT_URL present=%s len=%s, R2_BUCKET present=%s, R2_ACCESS_KEY_ID present=%s, R2_SECRET_ACCESS_KEY present=%s (masked), USE_R2=%s, resolved_endpoint=%s, resolved_bucket=%s",
    BOTO3_AVAILABLE,
    bool(os.getenv("R2_ENDPOINT_URL") or os.getenv("R2_ENDPOINT") or os.getenv("S3_ENDPOINT_URL")),
    len((os.getenv("R2_ENDPOINT_URL") or os.getenv("R2_ENDPOINT") or os.getenv("S3_ENDPOINT_URL") or "")),
    bool(os.getenv("R2_BUCKET")),
    bool(os.getenv("R2_ACCESS_KEY_ID")),
    bool(os.getenv("R2_SECRET_ACCESS_KEY")),  # Solo presenza, mai il valore o la lunghezza
    USE_R2,
    bool(R2_ENDPOINT_URL),
    bool(R2_BUCKET),
)

# === R2 existence check helper ===
async def _r2_object_exists(key: str) -> bool:
    """R2 is the source of truth for whether a photo exists."""
    if not USE_R2 or not r2_client or not R2_BUCKET:
        return False

    def _head() -> bool:
        try:
            r2_client.head_object(Bucket=R2_BUCKET, Key=key)
            return True
        except Exception as e:
            # botocore ClientError has a response dict; treat 404/NoSuchKey/NotFound as missing
            try:
                code = getattr(e, "response", {}).get("Error", {}).get("Code")
                if code in ("404", "NoSuchKey", "NotFound"):
                    return False
            except Exception:
                pass
            # Other errors should be visible in logs
            logger.warning(f"R2 head_object error for key={key}: {e}")
            return False

    return await asyncio.to_thread(_head)

async def _filter_missing_r2_photos(email: str, photo_ids: List[str], use_cache: bool = True) -> Tuple[List[str], List[str]]:
    """
    Filtra foto che non esistono più in R2 usando cache (veloce) e le marca come 'deleted' nel DB.
    Returns: (kept_photo_ids, missing_photo_ids)
    """
    if not USE_R2 or not r2_client or not R2_BUCKET or not photo_ids:
        return (photo_ids, [])
    
    kept: List[str] = []
    missing: List[str] = []
    
    # Usa cache per ottenere set delle chiavi esistenti
    if use_cache:
        r2_keys_set = await get_r2_keys_set_cached()
        
        # Filtra usando il set (velocissimo)
        for photo_id in photo_ids:
            if photo_id in r2_keys_set:
                kept.append(photo_id)
            else:
                missing.append(photo_id)
    else:
        # Fallback: controlla una per una (più lento, solo se cache non disponibile)
        semaphore = asyncio.Semaphore(10)
        
        async def check_photo(photo_id: str):
            async with semaphore:
                exists = await _r2_object_exists(photo_id)
                return (photo_id, exists)
        
        results = await asyncio.gather(*[check_photo(pid) for pid in photo_ids])
        
        for photo_id, exists in results:
            if exists:
                kept.append(photo_id)
            else:
                missing.append(photo_id)
    
    # Marca le foto mancanti come 'deleted' nel DB (solo se non già deleted)
    if missing:
        logger.info(f"[R2_FILTER] filtered_out_missing={len(missing)} for email={email}")
        for photo_id in missing:
            try:
                # Aggiorna status, r2_exists e r2_last_checked
                await _db_execute_write(
                    """UPDATE user_photos 
                       SET status = 'deleted', r2_exists = FALSE, r2_last_checked = NOW()
                       WHERE email = $1 AND photo_id = $2 AND (status != 'deleted' OR r2_exists = TRUE)""",
                    (email, photo_id)
                )
            except Exception as e:
                logger.error(f"Error marking photo {photo_id} as deleted: {e}")
    
    return (kept, missing)

# Cache per chiavi R2 (TTL 60 secondi)
_r2_keys_cache: Optional[dict] = None
_r2_keys_cache_lock = asyncio.Lock()
R2_KEYS_CACHE_TTL = 120  # secondi

async def get_r2_keys_set_cached() -> set:
    """
    Ottiene il set delle chiavi esistenti in R2 con cache (TTL 120s).
    Evita di listare il bucket ad ogni richiesta.
    """
    global _r2_keys_cache
    
    async with _r2_keys_cache_lock:
        now = datetime.now(timezone.utc)
        
        # Se cache valida, ritorna
        if _r2_keys_cache and (now - _r2_keys_cache['fetched_at']).total_seconds() < R2_KEYS_CACHE_TTL:
            return _r2_keys_cache['keys_set']
        
        # Cache scaduta o non esiste: lista bucket
        if not USE_R2 or not r2_client:
            return set()
        
        try:
            keys_set = set()
            paginator = r2_client.get_paginator('list_objects_v2')
            prefix = R2_PHOTOS_PREFIX if R2_PHOTOS_PREFIX else ""
            
            for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=prefix):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    # Ignora file di sistema
                    if key not in ['faces.index', 'faces.meta.jsonl']:
                        keys_set.add(key)
            
            # Aggiorna cache
            _r2_keys_cache = {
                'keys_set': keys_set,
                'fetched_at': now
            }
            
            logger.info(f"[R2_CACHE] Updated cache with {len(keys_set)} keys")
            return keys_set
        except Exception as e:
            logger.error(f"Error listing R2 keys for cache: {e}")
            # Se errore, ritorna cache vecchia se esiste, altrimenti set vuoto
            if _r2_keys_cache:
                return _r2_keys_cache['keys_set']
            return set()


# Crea cartelle necessarie all'avvio
PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Log diagnostico all'avvio
logger.info("=" * 80)
logger.info("🚀 APPLICATION STARTUP")
logger.info("=" * 80)
logger.info(f"📂 BASE_DIR (absolute): {BASE_DIR.resolve()}")
logger.info(f"📂 Current working directory: {os.getcwd()}")
logger.info(f"📂 __file__ location: {Path(__file__).resolve()}")
logger.info("")
if R2_ONLY_MODE:
    logger.info(f"📁 PHOTOS_DIR: DISABLED (R2_ONLY_MODE enabled - photos served only from R2)")
else:
    logger.info(f"📁 PHOTOS_DIR (absolute): {PHOTOS_DIR.resolve()}")
    logger.info(f"   PHOTOS_DIR exists: {PHOTOS_DIR.exists()}")
    if PHOTOS_DIR.exists():
        try:
            photo_files = list(PHOTOS_DIR.iterdir())
            logger.info(f"   Photos found: {len(photo_files)}")
            if photo_files:
                logger.info(f"   First 5 files: {[p.name for p in photo_files[:5]]}")
        except Exception as e:
            logger.error(f"   Error listing photos: {e}")
    else:
        logger.warning("   ⚠️  PHOTOS_DIR does not exist!")
logger.info("=" * 80)

app = FastAPI(title="Face Match API", version="2026-01-07-03-35")

# Exception handler per non loggare 404 su favicon/apple-touch-icon (richieste automatiche browser)
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler unificato per HTTPException: non logga 404 su favicon/apple-touch-icon"""
    # Non loggare come errore i 404 su favicon/apple-touch-icon (richieste automatiche browser)
    if exc.status_code == 404 and request.url.path in ["/favicon.ico", "/apple-touch-icon.png", "/apple-touch-icon-precomposed.png"]:
        # Ritorna 404 senza loggare (i browser richiedono automaticamente questi file)
        return JSONResponse(status_code=404, content={"detail": exc.detail})
    # Per tutti gli altri errori, logga normalmente
    logger.error(f"HTTPException: {exc.status_code} - {exc.detail} - Path: {request.url.path}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware per disabilitare cache su TUTTE le risposte
@app.middleware("http")
async def no_cache_middleware(request: Request, call_next):
    """Disabilita cache per tutte le risposte (HTML, API, static)"""
    response = await call_next(request)
    # Aggiungi header no-cache a tutte le risposte
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Debug endpoint: shows which code version/file is running on the server.
@app.get("/debug/build")
async def debug_build():
    """Debug endpoint: shows which code version/file is running on the server."""
    try:
        file_path = str(Path(__file__).resolve())
    except Exception:
        file_path = str(__file__)

    return {
        "app_build_id": APP_BUILD_ID,
        "file": file_path,
        "cwd": os.getcwd(),
        "app_name": APP_NAME
    }

@app.get("/debug/index-status")
def debug_index_status():
    """Debug endpoint: shows status of FAISS index and metadata files."""
    faces_index_exists = INDEX_PATH.exists()
    faces_meta_exists = META_PATH.exists()
    
    return {
        "base_dir": str(BASE_DIR.resolve()),
        "data_dir": str(DATA_DIR.resolve()),
        "faces_index_path": str(INDEX_PATH.resolve()),
        "faces_meta_path": str(META_PATH.resolve()),
        "faces_index_exists": faces_index_exists,
        "faces_meta_exists": faces_meta_exists,
        "faces_index_size_bytes": INDEX_PATH.stat().st_size if faces_index_exists else 0,
        "faces_meta_size_bytes": META_PATH.stat().st_size if faces_meta_exists else 0,
        "faiss_loaded": faiss_index is not None,
        "meta_loaded": len(meta_rows) > 0
    }

@app.get("/dev/mock-success")
async def dev_mock_success():
    """Endpoint mock per sviluppo locale - restituisce dati simulati di ordine pagato"""
    try:
        # Prendi alcune foto disponibili (da R2 o da PHOTOS_DIR)
        photo_ids = []
        
        if R2_ONLY_MODE and r2_client:
            # Se R2_ONLY_MODE, lista foto da R2
            try:
                keys_set = await get_r2_keys_set_cached()
                # Prendi 8 foto a caso (escludi file di sistema)
                photo_keys = [k for k in keys_set if not k.startswith('faces.') and not k.startswith('wm/') and not k.startswith('thumbs/')]
                photo_keys = [k for k in photo_keys if any(k.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.heic'])]
                photo_ids = photo_keys[:8]
            except Exception as e:
                logger.warning(f"Error getting R2 keys for mock: {e}")
        elif PHOTOS_DIR.exists():
            # Se PHOTOS_DIR esiste, prendi foto da lì
            try:
                photo_files = list(PHOTOS_DIR.glob("*.jpg")) + list(PHOTOS_DIR.glob("*.jpeg")) + list(PHOTOS_DIR.glob("*.png"))
                photo_files = [f for f in photo_files if not f.name.endswith('_watermarked.jpg')]
                photo_ids = [f.name for f in photo_files[:8]]
            except Exception as e:
                logger.warning(f"Error listing photos for mock: {e}")
        
        # Se non abbiamo foto, crea ID mock
        if not photo_ids:
            photo_ids = [f"mock_photo_{i+1}.jpg" for i in range(8)]
        
        # Costruisci paid_photos con URL
        from urllib.parse import quote
        paid_photos = []
        for photo_id in photo_ids:
            # Usa gli stessi URL pattern del progetto
            photo_id_encoded = quote(photo_id, safe='')
            thumb_url = f"/photo/{photo_id_encoded}?variant=thumb"
            full_url = f"/photo/{photo_id_encoded}?paid=true"
            
            paid_photos.append({
                "photo_id": photo_id,
                "thumb_url": thumb_url,
                "full_url": full_url
            })
        
        return {
            "ok": True,
            "status": "paid",
            "order_code": "TEST-ORDER-1234",
            "customer_email": "test@example.com",
            "session_id": "mock_session_dev",
            "photo_ids": photo_ids,  # Per compatibilità con /stripe/verify
            "paid_photos": paid_photos,  # Formato esteso con URL
            "payment_status": "paid",
            "amount_total": 4700,  # 47 euro in centesimi
            "currency": "eur"
        }
    except Exception as e:
        logger.error(f"Error in dev/mock-success: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Mock error: {str(e)}")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# NON usare mount per /photo - usiamo endpoint esplicito per controllo migliore
# app.mount("/photo", StaticFiles(directory=str(PHOTOS_DIR)), name="photos")
logger.info(f"Will serve photos from: {PHOTOS_DIR.resolve()}")
# === PHOTO ENDPOINT ===

face_app: Optional[FaceAnalysis] = None
faiss_index: Optional[faiss.Index] = None
meta_rows: List[Dict[str, Any]] = []
indexing_lock: Optional[asyncio.Lock] = None  # Lock globale per indicizzazione automatica

# Funzioni helper
# ========== DATABASE (PostgreSQL) ==========

def _normalize_email(email: str) -> str:
    """Normalizza l'email: minuscolo, senza spazi"""
    if not email:
        return email
    return email.strip().lower()

async def _init_database():
    """Inizializza il database PostgreSQL con le tabelle necessarie (disabilitato in stateless mode)"""
    global db_pool
    
    if not USE_POSTGRES:
        logger.info("Database initialization skipped (stateless mode)")
        return
    
    try:
        logger.info(f"Initializing PostgreSQL database: {DATABASE_URL[:30]}...")
        
        # Crea pool di connessioni
        db_pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=2,
            max_size=10,
            command_timeout=60
        )
        logger.info("PostgreSQL connection pool created")
        
        # Usa il pool per creare le tabelle
        async with db_pool.acquire() as conn:
            # Tabella utenti
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    email VARCHAR(255) PRIMARY KEY,
                    selfie_embedding BYTEA,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_login_at TIMESTAMP,
                    last_selfie_at TIMESTAMP
                )
            """)
            
            # Tabella foto utente
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_photos (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255) NOT NULL,
                    photo_id VARCHAR(255) NOT NULL,
                    found_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    paid_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    status VARCHAR(50) NOT NULL DEFAULT 'found',
                    source_member_id INTEGER NULL,
                    FOREIGN KEY (email) REFERENCES users(email),
                    UNIQUE(email, photo_id)
                )
            """)
            
            
            # Tabella ordini
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id VARCHAR(255) PRIMARY KEY,
                    email VARCHAR(255) NOT NULL,
                    stripe_session_id VARCHAR(255),
                    photo_ids TEXT NOT NULL,
                    amount_cents INTEGER NOT NULL,
                    paid_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    download_token VARCHAR(255) UNIQUE,
                    FOREIGN KEY (email) REFERENCES users(email)
                )
            """)
            
            # Tabella carrelli
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS carts (
                    session_id VARCHAR(255) PRIMARY KEY,
                    photo_ids TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabella foto indicizzate (per tracking indicizzazione automatica)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS indexed_photos (
                    photo_id VARCHAR(255) PRIMARY KEY,
                    indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            
            # Tabella membri famiglia
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS family_members (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255) NOT NULL,
                    member_name VARCHAR(120),
                    selfie_embedding BYTEA NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (email) REFERENCES users(email) ON DELETE CASCADE
                )
            """)
            
            # Tabella profili volto utente (per auto-match)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_face_profiles (
                    email VARCHAR(255) PRIMARY KEY,
                    face_embedding BYTEA NOT NULL,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_auto_match_at TIMESTAMP NULL,
                    FOREIGN KEY (email) REFERENCES users(email) ON DELETE CASCADE
                )
            """)
            
            # Tabella photo_assets per tracciare thumb e wm precomputati
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS photo_assets (
                    photo_id VARCHAR(255) PRIMARY KEY,
                    thumb_key TEXT,
                    wm_key TEXT,
                    thumb_ready BOOLEAN DEFAULT FALSE,
                    wm_ready BOOLEAN DEFAULT FALSE,
                    assets_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (photo_id) REFERENCES indexed_photos(photo_id) ON DELETE CASCADE
                )
            """)
            
            # Migrazione: aggiungi colonna source_member_id se non esiste (per DB già esistenti)
            try:
                await conn.execute("ALTER TABLE user_photos ADD COLUMN IF NOT EXISTS source_member_id INTEGER")
            except Exception as e:
                logger.warning(f"Could not add source_member_id column (might already exist): {e}")
            
            # Migrazione: aggiungi colonne r2_exists e r2_last_checked (per R2 source of truth)
            try:
                await conn.execute("ALTER TABLE user_photos ADD COLUMN IF NOT EXISTS r2_exists BOOLEAN NOT NULL DEFAULT TRUE")
            except Exception as e:
                logger.warning(f"Could not add r2_exists column (might already exist): {e}")
            
            try:
                await conn.execute("ALTER TABLE user_photos ADD COLUMN IF NOT EXISTS r2_last_checked TIMESTAMP NULL")
            except Exception as e:
                logger.warning(f"Could not add r2_last_checked column (might already exist): {e}")
            
            # Indici per performance
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_user_photos_email ON user_photos(email)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_user_photos_status ON user_photos(status)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_user_photos_expires ON user_photos(expires_at)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_user_photos_source_member ON user_photos(source_member_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_user_photos_r2_exists ON user_photos(r2_exists)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_user_photos_photo_id ON user_photos(photo_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_family_members_email ON family_members(email)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_email ON orders(email)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_token ON orders(download_token)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_user_face_profiles_email ON user_face_profiles(email)")
        
        logger.info("PostgreSQL database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing PostgreSQL database: {e}", exc_info=True)
        raise

# Helper functions per database PostgreSQL (con pool) - disabilitate in stateless mode
async def _db_execute(query: str, params: tuple = ()):
    """Esegue una query e restituisce i risultati (usa pool PostgreSQL)"""
    if not USE_POSTGRES or db_pool is None:
        return []  # Stateless mode: ritorna lista vuota
    
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
        return [dict(row) for row in rows]

async def _db_execute_one(query: str, params: tuple = ()):
    """Esegue una query e restituisce un solo risultato"""
    if not USE_POSTGRES or db_pool is None:
        return None  # Stateless mode: ritorna None
    
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(query, *params)
        return dict(row) if row else None

async def _db_execute_write(query: str, params: tuple = ()):
    """Esegue una query di scrittura (INSERT, UPDATE, DELETE) - disabilitata in stateless mode"""
    if not USE_POSTGRES or db_pool is None:
        return  # Stateless mode: no-op
    
    async with db_pool.acquire() as conn:
        await conn.execute(query, *params)

async def _get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Recupera un utente per email"""
    try:
        email = _normalize_email(email)
        row = await _db_execute_one(
            "SELECT * FROM users WHERE email = $1",
            (email,)
        )
        return row
    except Exception as e:
        logger.error(f"Error getting user: {e}")
    return None

async def _create_or_update_user(email: str, selfie_embedding: Optional[bytes] = None) -> bool:
    """Crea o aggiorna un utente (salva solo con email, selfie opzionale)"""
    try:
        email = _normalize_email(email)
        
        # Verifica se esiste
        exists = await _db_execute_one(
            "SELECT email FROM users WHERE email = $1",
            (email,)
        )
        
        if exists:
            # Aggiorna
            if selfie_embedding:
                await _db_execute_write("""
                    UPDATE users 
                    SET selfie_embedding = $1, last_login_at = NOW(), last_selfie_at = NOW()
                    WHERE email = $2
                """, (selfie_embedding, email))
            else:
                # Aggiorna solo last_login_at (non sovrascrivere selfie_embedding esistente)
                await _db_execute_write("""
                    UPDATE users 
                    SET last_login_at = NOW()
                    WHERE email = $1
                """, (email,))
        else:
            # Crea nuovo utente (solo con email, selfie opzionale)
            # created_at ha DEFAULT CURRENT_TIMESTAMP, quindi non serve passarlo
            if selfie_embedding:
                await _db_execute_write("""
                    INSERT INTO users (email, selfie_embedding, last_login_at, last_selfie_at)
                    VALUES ($1, $2, NOW(), NOW())
                """, (email, selfie_embedding))
            else:
                await _db_execute_write("""
                    INSERT INTO users (email, last_login_at)
                    VALUES ($1, NOW())
                """, (email,))
        
        return True
    except Exception as e:
        logger.error(f"Error creating/updating user: {e}")
    return False

async def _save_user_face_profile(email: str, face_embedding: np.ndarray) -> bool:
    """Salva o aggiorna il profilo volto di un utente per auto-match"""
    try:
        email = _normalize_email(email)
        embedding_bytes = face_embedding.tobytes()
        
        await _db_execute_write("""
            INSERT INTO user_face_profiles (email, face_embedding, updated_at)
            VALUES ($1, $2, NOW())
            ON CONFLICT (email) DO UPDATE
            SET face_embedding = EXCLUDED.face_embedding,
                updated_at = NOW()
        """, (email, embedding_bytes))
        
        # Stateless: non salvare face profile
        return True
    except Exception as e:
        logger.error(f"Error saving face profile: {e}")
        return False

async def auto_match_new_photos_for_email(email: str) -> int:
    """
    Auto-match nuove foto per un utente usando il profilo volto salvato.
    Rate-limited: esegue solo se last_auto_match_at è NULL o più vecchio di 60 secondi.
    Returns: numero di nuove foto aggiunte
    """
    try:
        email = _normalize_email(email)
        
        # Recupera profilo completo (una sola query)
        profile = await _db_execute_one(
            "SELECT face_embedding, last_auto_match_at FROM user_face_profiles WHERE email = $1",
            (email,)
        )
        
        if not profile or not profile.get('face_embedding'):
            # Log solo a livello debug per non intasare i log
            logger.debug(f"No face profile found for {email}, skipping auto-match")
            return 0
        
        # Verifica rate-limit: controlla last_auto_match_at
        if profile.get('last_auto_match_at'):
            last_match = profile['last_auto_match_at']
            # PostgreSQL restituisce datetime object o None
            if last_match:
                # Se è già un datetime, usalo direttamente
                if isinstance(last_match, datetime):
                    # Se non ha timezone, assumi UTC
                    if last_match.tzinfo is None:
                        last_match = last_match.replace(tzinfo=timezone.utc)
                elif isinstance(last_match, str):
                    # Se è stringa, parsala
                    try:
                        last_match = datetime.fromisoformat(last_match.replace('Z', '+00:00'))
                    except:
                        # Fallback: ignora rate-limit se parsing fallisce
                        last_match = None
                
                if last_match:
                    now = datetime.now(timezone.utc)
                    if last_match.tzinfo is None:
                        last_match = last_match.replace(tzinfo=timezone.utc)
                    time_diff = (now - last_match).total_seconds()
                    if time_diff < 60:
                        logger.info(f"Auto-match skipped for {email}: rate-limited (last match {time_diff:.1f}s ago)")
                        return 0  # NON aggiornare last_auto_match_at se rate-limited
        
        embedding_bytes = profile['face_embedding']
        face_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        
        if len(face_embedding) != INDEX_DIM:
            logger.warning(f"Face embedding for {email} has wrong dimension: {len(face_embedding)}")
            return 0
        
        # Verifica che FAISS index sia disponibile
        if faiss_index is None or faiss_index.ntotal == 0 or len(meta_rows) == 0:
            logger.debug(f"FAISS index not available for auto-match ({email})")
            return 0
        
        # Normalizza embedding e cerca match
        face_emb = _normalize(face_embedding).reshape(1, -1)
        top_k = min(200, faiss_index.ntotal)
        min_score = 0.25  # Stesso default di /match_selfie
        
        D, I = faiss_index.search(face_emb, top_k)
        
        # Recupera foto già esistenti per questa email (per evitare duplicati)
        existing_photos = await _db_execute(
            "SELECT photo_id FROM user_photos WHERE email = $1",
            (email,)
        )
        existing_photo_ids = {row['photo_id'] for row in existing_photos}
        
        # Processa risultati e aggiungi nuove foto
        new_photos_count = 0
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0 or idx >= len(meta_rows):
                continue
            if float(score) < min_score:
                continue
            
            row = meta_rows[idx]
            photo_id = row.get("photo_id")
            if not photo_id:
                continue
            
            # Salva solo se non esiste già
            if photo_id not in existing_photo_ids:
                try:
                    await _add_user_photo(email, photo_id, "found")
                    existing_photo_ids.add(photo_id)  # Evita duplicati nello stesso batch
                    new_photos_count += 1
                except Exception as e:
                    logger.warning(f"Error adding photo {photo_id} for {email}: {e}")
        
        # Aggiorna last_auto_match_at SOLO se abbiamo eseguito il match (non se rate-limited)
        await _db_execute_write(
            "UPDATE user_face_profiles SET last_auto_match_at = NOW() WHERE email = $1",
            (email,)
        )
        
        if new_photos_count > 0:
            logger.info(f"Auto-matched {new_photos_count} new photos for {email}")
        else:
            logger.debug(f"Auto-match completed for {email}: no new photos found")
        
        return new_photos_count
    except Exception as e:
        logger.error(f"Error in auto_match_new_photos_for_email for {email}: {e}", exc_info=True)
        return 0

async def _add_user_photo(email: str, photo_id: str, status: str = "found") -> bool:
    """Aggiunge una foto trovata per un utente"""
    try:
        email = _normalize_email(email)
        
        # Verifica se esiste già (e qual è lo stato attuale)
        exists = await _db_execute_one(
            "SELECT id, status, expires_at FROM user_photos WHERE email = $1 AND photo_id = $2",
            (email, photo_id)
        )
        
        # PostgreSQL: usa NOW() e INTERVAL per evitare problemi con timezone
        days = 30 if status == "paid" else 90

        if exists:
            current_status = (exists.get("status") or "").lower()

            # NON degradare mai una foto già pagata a 'found' (succede quando match_selfie ri-salva la lista)
            if current_status == "paid" and status != "paid":
                await _db_execute_write(
                    """
                    UPDATE user_photos
                    SET found_at = NOW()
                    WHERE email = $1 AND photo_id = $2
                    """,
                    (email, photo_id)
                )
                return True

            # Se è stata marcata come deleted, non riattivarla automaticamente
            if current_status == "deleted" and status != "paid":
                return True

            # Aggiorna normalmente (può anche promuovere a 'paid')
            await _db_execute_write(f"""
                UPDATE user_photos 
                SET found_at = NOW(), status = $1, expires_at = NOW() + INTERVAL '{days} days'
                WHERE email = $2 AND photo_id = $3
            """, (status, email, photo_id))
        else:
            # Inserisci nuovo - usa f-string per INTERVAL
            await _db_execute_write(f"""
                INSERT INTO user_photos (email, photo_id, found_at, status, expires_at)
                VALUES ($1, $2, NOW(), $3, NOW() + INTERVAL '{days} days')
            """, (email, photo_id, status))
        
        return True
    except Exception as e:
        logger.error(f"Error adding user photo: {e}")
    return False

async def _mark_photo_paid(email: str, photo_id: str) -> bool:
    """Marca una foto come pagata (crea record se non esiste)"""
    try:
        email = _normalize_email(email)
        
        # Verifica se esiste già
        exists = await _db_execute_one("""
            SELECT id FROM user_photos 
            WHERE email = $1 AND photo_id = $2
        """, (email, photo_id))
        
        # PostgreSQL: usa NOW() e INTERVAL per evitare problemi con timezone
        if exists:
            # Aggiorna record esistente
            await _db_execute_write("""
                UPDATE user_photos 
                SET paid_at = NOW(), status = 'paid', expires_at = NOW() + INTERVAL '30 days'
                WHERE email = $1 AND photo_id = $2
            """, (email, photo_id))
        else:
            # Crea nuovo record (foto pagata senza essere stata trovata prima)
            await _db_execute_write("""
                INSERT INTO user_photos (email, photo_id, found_at, paid_at, status, expires_at)
                VALUES ($1, $2, NOW(), NOW(), 'paid', NOW() + INTERVAL '30 days')
            """, (email, photo_id))
        
        logger.info(f"✅ Photo marked as paid: {email} - {photo_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Error marking photo paid: {e}", exc_info=True)
        logger.error(f"Exception type: {type(e).__name__}")
    return False

async def _get_user_paid_photos(email: str) -> List[str]:
    """Recupera lista foto pagate per un utente (non scadute)"""
    try:
        email = _normalize_email(email)
        
        # Prima verifica tutte le foto per questo utente (per debug)
        all_rows = await _db_execute("""
            SELECT photo_id, status, expires_at FROM user_photos 
            WHERE email = $1
        """, (email,))
        logger.info(f"All photos for {email}: {len(all_rows)} total")
        for row in all_rows:
            logger.info(f"  - {row['photo_id']}: status={row['status']}, expires_at={row['expires_at']}")
        
        # Poi recupera solo quelle pagate e non scadute
        # NON includere mai status='deleted' e assicurati che r2_exists=TRUE
        # PostgreSQL: usa NOW() per evitare problemi con timezone
        rows = await _db_execute("""
            SELECT photo_id FROM user_photos 
            WHERE email = $1 
              AND status = 'paid' 
              AND status != 'deleted'
              AND expires_at > NOW() 
              AND r2_exists = TRUE
        """, (email,))
        photo_ids = [row['photo_id'] for row in rows]
        
        # NOTA: R2_FILTER viene applicato in /user/photos usando cache (più veloce)
        # Non applicarlo qui per evitare doppio filtro
        
        logger.info(f"Paid photos (not expired) for {email}: {len(photo_ids)} photos - {photo_ids}")
        return photo_ids
    except Exception as e:
        logger.error(f"Error getting paid photos for {email}: {e}", exc_info=True)
    return []

async def _get_user_found_photos(email: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Recupera tutte le foto trovate per un utente (con limite opzionale per performance)"""
    try:
        email = _normalize_email(email)
        # Limite default di 200 foto per evitare problemi di performance
        # Se serve di più, si può aumentare o implementare paginazione
        limit_clause = f"LIMIT {limit}" if limit else "LIMIT 200"
        
        rows = await _db_execute(f"""
            SELECT photo_id, found_at, paid_at, expires_at, status 
            FROM user_photos 
            WHERE email = $1 AND status IN ('found', 'paid') AND r2_exists = TRUE
            ORDER BY found_at DESC
            {limit_clause}
        """, (email,))
        
        # R2 is the source of truth: filtra sempre le foto che non esistono più in R2
        if rows:
            photo_ids = [row.get("photo_id") for row in rows if row.get("photo_id")]
            if photo_ids:
                # NOTA: R2_FILTER viene applicato in /user/photos usando cache (più veloce)
                # Non applicarlo qui per evitare doppio filtro
                kept_ids_set = set(photo_ids)
                rows = [row for row in rows if row.get("photo_id") in kept_ids_set]
        
        return rows
    except Exception as e:
        logger.error(f"Error getting found photos: {e}")
    return []

async def _match_selfie_embedding(selfie_embedding: bytes, threshold: float = 0.7) -> Optional[str]:
    """Confronta selfie embedding con quelli salvati, ritorna email se match"""
    
    try:
        rows = await _db_execute(
            "SELECT email, selfie_embedding FROM users WHERE selfie_embedding IS NOT NULL"
        )
        
        selfie_emb = np.frombuffer(selfie_embedding, dtype=np.float32)
        selfie_emb = _normalize(selfie_emb)
        
        best_match = None
        best_score = 0.0
        
        for row in rows:
            saved_emb = np.frombuffer(row['selfie_embedding'], dtype=np.float32)
            saved_emb = _normalize(saved_emb)
            
            # Calcola cosine similarity
            score = np.dot(selfie_emb, saved_emb)
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = row['email']
        
        if best_match:
            logger.info(f"Selfie matched with user: {best_match} (score: {best_score:.4f})")
            return best_match
    except Exception as e:
        logger.error(f"Error matching selfie: {e}")
    return None

async def _create_order(email: str, order_id: str, stripe_session_id: str, photo_ids: List[str], amount_cents: int) -> Optional[str]:
    """Crea un ordine e ritorna download token"""
    try:
        email = _normalize_email(email)
        logger.info(f"_create_order called: email={email}, order_id={order_id}, photo_count={len(photo_ids)}")
        download_token = secrets.token_urlsafe(32)
        logger.info(f"Generated download_token: {download_token[:20]}...")
        
        # PostgreSQL: usa NOW() per evitare problemi con timezone
        # paid_at ha DEFAULT CURRENT_TIMESTAMP, ma lo passiamo esplicitamente per chiarezza
        await _db_execute_write("""
            INSERT INTO orders (order_id, email, stripe_session_id, photo_ids, amount_cents, paid_at, download_token)
            VALUES ($1, $2, $3, $4, $5, NOW(), $6)
        """, (order_id, email, stripe_session_id, json.dumps(photo_ids), amount_cents, download_token))
        logger.info("Order inserted into database")
        
        # Marca foto come pagate
        logger.info(f"Starting to mark {len(photo_ids)} photos as paid for {email}")
        for i, photo_id in enumerate(photo_ids):
            logger.info(f"Marking photo {i+1}/{len(photo_ids)}: {photo_id}")
            result = await _mark_photo_paid(email, photo_id)
            if result:
                logger.info(f"✅ Successfully marked {photo_id} as paid")
            else:
                logger.error(f"❌ Failed to mark {photo_id} as paid")
        logger.info(f"Completed marking photos as paid. Total: {len(photo_ids)}")
        
        logger.info("Order committed successfully")
        return download_token
    except Exception as e:
        logger.error(f"Error creating order: {e}", exc_info=True)
        logger.error(f"Exception type: {type(e).__name__}")
    return None

async def _get_order_by_token(token: str) -> Optional[Dict[str, Any]]:
    """Recupera ordine per download token"""
    try:
        row = await _db_execute_one(
            "SELECT * FROM orders WHERE download_token = $1",
            (token,)
        )

        if row:
            order = dict(row)
            order['photo_ids'] = json.loads(order['photo_ids']) if order.get('photo_ids') else []
            return order
    except Exception as e:
        logger.error(f"Error getting order: {e}")
    return None


async def _cleanup_expired_photos():
    """Elimina foto scadute dal database e dal filesystem"""
    try:
        # Trova foto scadute
        # PostgreSQL: usa NOW() per evitare problemi con timezone
        expired = await _db_execute("""
            SELECT email, photo_id, status FROM user_photos
            WHERE expires_at < NOW() AND status != 'deleted'
        """, ())
        
        deleted_count = 0
        for row in expired:
            email = row['email']
            photo_id = row['photo_id']
            # Marca come deleted nel database
            await _db_execute_write("""
                UPDATE user_photos SET status = 'deleted' 
                WHERE email = $1 AND photo_id = $2
            """, (email, photo_id))
            
            # Foto eliminate dal database (file su R2, non eliminiamo fisicamente per ora)
            # Nota: le foto sono su R2, l'eliminazione fisica può essere fatta manualmente se necessario
            deleted_count += 1
            logger.info(f"Marked expired photo as deleted in DB: {photo_id} (file remains on R2)")
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired photos ({deleted_count} files deleted)")
    except Exception as e:
        logger.error(f"Error cleaning up expired photos: {e}")

# Email system disabled - all email functions removed

# ========== FUNZIONI HELPER ESISTENTI ==========

def _normalize(v: np.ndarray) -> np.ndarray:
    """Normalizza un vettore"""
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def _generate_multi_embeddings_from_image(img: np.ndarray, num_embeddings: int = 2) -> List[np.ndarray]:
    """
    Genera 3-4 embeddings di riferimento per migliorare il riconoscimento di foto difficili:
    - emb0: originale
    - emb1: flip orizzontale
    - emb2: rotazione leggera +5 gradi (per catturare profili)
    - emb3: rotazione leggera -5 gradi (per catturare profili)
    Usa UNA sola detection: crop dal bbox + (se disponibile) alignment con landmarks.
    """
    assert face_app is not None

    faces = face_app.get(img)
    if not faces:
        return []
    faces_sorted = sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True
    )
    if not faces_sorted:
        return []
    main_face = faces_sorted[0]

    # Prova a usare alignment con kps se disponibile
    aligned = None
    try:
        from insightface.utils import face_align
        if hasattr(main_face, "kps") and main_face.kps is not None:
            aligned = face_align.norm_crop(img, main_face.kps)
    except Exception:
        aligned = None

    if aligned is None:
        # Fallback: crop da bbox
        h, w = img.shape[:2]
        x1, y1, x2, y2 = main_face.bbox
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(w, int(x2)); y2 = min(h, int(y2))
        aligned = img[y1:y2, x1:x2].copy()

    embeddings = []
    
    # embedding originale: usa face.embedding (già calcolato nella detection)
    emb0 = _normalize(main_face.embedding.astype(np.float32))
    embeddings.append(emb0)

    # embedding flip: prova prima con recognition model, poi fallback a seconda detection
    emb1 = None
    try:
        recog = getattr(face_app, "models", {}).get("recognition") if hasattr(face_app, "models") else None
        if recog is not None:
            flip_crop = cv2.flip(aligned, 1)
            emb1 = _normalize(recog.get(flip_crop).astype(np.float32))
    except Exception:
        emb1 = None

    # Fallback: se recog.get() fallisce, fai una seconda detection sull'immagine flippata
    if emb1 is None:
        try:
            flip_img = cv2.flip(img, 1)
            flip_faces = face_app.get(flip_img)
            if flip_faces:
                flip_faces_sorted = sorted(
                    flip_faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                    reverse=True
                )
                if flip_faces_sorted:
                    flip_face = flip_faces_sorted[0]
                    emb1 = _normalize(flip_face.embedding.astype(np.float32))
        except Exception:
            pass

    if emb1 is None:
        emb1 = emb0.copy()  # Duplica originale se flip fallisce
    embeddings.append(emb1)

    # Embeddings con rotazioni leggere per catturare profili (solo se abbiamo recog model)
    try:
        recog = getattr(face_app, "models", {}).get("recognition") if hasattr(face_app, "models") else None
        if recog is not None:
            # Rotazione +5 gradi (profilo sinistro)
            center = (aligned.shape[1] // 2, aligned.shape[0] // 2)
            M_rot_pos = cv2.getRotationMatrix2D(center, 5, 1.0)
            rotated_pos = cv2.warpAffine(aligned, M_rot_pos, (aligned.shape[1], aligned.shape[0]))
            try:
                emb2 = _normalize(recog.get(rotated_pos).astype(np.float32))
                embeddings.append(emb2)
            except Exception:
                pass

            # Rotazione -5 gradi (profilo destro)
            M_rot_neg = cv2.getRotationMatrix2D(center, -5, 1.0)
            rotated_neg = cv2.warpAffine(aligned, M_rot_neg, (aligned.shape[1], aligned.shape[0]))
            try:
                emb3 = _normalize(recog.get(rotated_neg).astype(np.float32))
                embeddings.append(emb3)
            except Exception:
                pass
    except Exception:
        pass

    # Se abbiamo meno di 2 embeddings, duplica l'originale
    if len(embeddings) < 2:
        logger.warning(f"[MULTI_EMB] Only {len(embeddings)} embeddings generated, duplicating original")
        while len(embeddings) < 2:
            embeddings.append(emb0.copy())

    logger.info(f"[MULTI_EMB] Generated {len(embeddings)} reference embeddings (original+flip+rotations)")
    return embeddings

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
    """Traccia un download per cleanup futuro (disabilitato in R2_ONLY_MODE)"""
    if R2_ONLY_MODE:
        # In R2_ONLY_MODE non tracciamo download su filesystem
        return
    
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
    """Cancella foto scaricate dopo X giorni o dopo N download (disabilitato in R2_ONLY_MODE)"""
    if R2_ONLY_MODE:
        # In R2_ONLY_MODE non usiamo filesystem per tracking
        return
    
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
        
        # Cancella foto (ora su R2, non eliminiamo fisicamente)
        deleted_count = 0
        for photo_id in photos_to_delete:
            # Foto su R2: non eliminiamo fisicamente, solo dal database
            # L'eliminazione fisica da R2 può essere fatta manualmente se necessario
            deleted_count += 1
            logger.info(f"Marked photo for deletion in DB: {photo_id} (file remains on R2)")
            try:
                # TODO: Se necessario, aggiungere eliminazione da R2 qui
                pass
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
    """Legge un'immagine da bytes, supporta anche HEIC convertendolo in JPEG"""
    # Prova prima con OpenCV (formati standard: JPEG, PNG, etc.)
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is not None:
        return img
    
    # Se OpenCV fallisce, prova con PIL (supporta più formati incluso HEIC se pillow-heif è installato)
    try:
        from io import BytesIO
        from PIL import Image
        
        # Prova a registrare supporto HEIC se disponibile
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except ImportError:
            pass  # pillow-heif non disponibile, continua comunque
        
        # Prova a leggere con PIL
        pil_img = Image.open(BytesIO(file_bytes))
        
        # Converti in RGB se necessario
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        # Converti PIL Image in numpy array per OpenCV
        img_array = np.array(pil_img)
        # PIL usa RGB, OpenCV usa BGR, quindi convertiamo
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img
    except Exception as e:
        logger.error(f"Error reading image with PIL: {e}")
        raise HTTPException(
            status_code=400, 
            detail=f"Formato immagine non supportato. Formati supportati: JPEG, PNG, HEIC (richiede pillow-heif). Errore: {str(e)}"
        )

def _read_selfie_image_with_resize(file_bytes: bytes, max_side: int = 1024) -> np.ndarray:
    """
    Legge selfie con correzione EXIF (se possibile) e resize per velocità.
    Logga dimensioni input e output.
    """
    from io import BytesIO
    try:
        from PIL import Image, ImageOps
        pil_img = Image.open(BytesIO(file_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        input_w, input_h = pil_img.size
        resized_w, resized_h = input_w, input_h
        if max(input_w, input_h) > max_side:
            if input_w >= input_h:
                resized_w = max_side
                resized_h = int(input_h * (max_side / input_w))
            else:
                resized_h = max_side
                resized_w = int(input_w * (max_side / input_h))
            pil_img = pil_img.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
        logger.info(f"[SELFIE] input_w={input_w} input_h={input_h} resized_w={resized_w} resized_h={resized_h}")
        img_array = np.array(pil_img)
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    except Exception:
        # Fallback OpenCV + resize (senza EXIF)
        img = _read_image_from_bytes(file_bytes)
        h, w = img.shape[:2]
        resized_w, resized_h = w, h
        if max(w, h) > max_side:
            if w >= h:
                resized_w = max_side
                resized_h = int(h * (max_side / w))
            else:
                resized_h = max_side
                resized_w = int(w * (max_side / h))
            img = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        logger.info(f"[SELFIE] input_w={w} input_h={h} resized_w={resized_w} resized_h={resized_h}")
        return img

def _generate_robust_selfie_embedding(img: np.ndarray) -> np.ndarray:
    """
    Genera embedding robusto del selfie creando 7 varianti e facendo media degli embedding.
    
    Varianti:
    1. Originale
    2. Crop leggero (5% da ogni lato)
    3. Brightness +10%
    4. Brightness -10%
    5. Contrast +10%
    6. Sharpening leggero
    7. Flip orizzontale
    
    Returns:
        np.ndarray: Embedding normalizzato (media degli embedding delle varianti)
    """
    assert face_app is not None
    
    # Converti OpenCV (BGR) a PIL (RGB) per le trasformazioni
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    embeddings = []
    
    # 1. Originale
    faces = face_app.get(img)
    if faces:
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )
        if faces_sorted:
            embeddings.append(faces_sorted[0].embedding.astype("float32"))
    
    # 2. Crop leggero (5% da ogni lato)
    w, h = pil_img.size
    crop_box = (int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95))
    img_crop = pil_img.crop(crop_box)
    img_crop_cv = cv2.cvtColor(np.array(img_crop), cv2.COLOR_RGB2BGR)
    faces = face_app.get(img_crop_cv)
    if faces:
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )
        if faces_sorted:
            embeddings.append(faces_sorted[0].embedding.astype("float32"))
    
    # 3. Brightness +10%
    enhancer = ImageEnhance.Brightness(pil_img)
    img_bright_plus = enhancer.enhance(1.1)
    img_bright_plus_cv = cv2.cvtColor(np.array(img_bright_plus), cv2.COLOR_RGB2BGR)
    faces = face_app.get(img_bright_plus_cv)
    if faces:
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )
        if faces_sorted:
            embeddings.append(faces_sorted[0].embedding.astype("float32"))
    
    # 4. Brightness -10%
    enhancer = ImageEnhance.Brightness(pil_img)
    img_bright_minus = enhancer.enhance(0.9)
    img_bright_minus_cv = cv2.cvtColor(np.array(img_bright_minus), cv2.COLOR_RGB2BGR)
    faces = face_app.get(img_bright_minus_cv)
    if faces:
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )
        if faces_sorted:
            embeddings.append(faces_sorted[0].embedding.astype("float32"))
    
    # 5. Contrast +10%
    enhancer = ImageEnhance.Contrast(pil_img)
    img_contrast = enhancer.enhance(1.1)
    img_contrast_cv = cv2.cvtColor(np.array(img_contrast), cv2.COLOR_RGB2BGR)
    faces = face_app.get(img_contrast_cv)
    if faces:
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )
        if faces_sorted:
            embeddings.append(faces_sorted[0].embedding.astype("float32"))
    
    # 6. Sharpening leggero
    img_sharp = pil_img.filter(ImageFilter.SHARPEN)
    img_sharp_cv = cv2.cvtColor(np.array(img_sharp), cv2.COLOR_RGB2BGR)
    faces = face_app.get(img_sharp_cv)
    if faces:
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )
        if faces_sorted:
            embeddings.append(faces_sorted[0].embedding.astype("float32"))
    
    # 7. Flip orizzontale
    img_flip = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    img_flip_cv = cv2.cvtColor(np.array(img_flip), cv2.COLOR_RGB2BGR)
    faces = face_app.get(img_flip_cv)
    if faces:
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )
        if faces_sorted:
            embeddings.append(faces_sorted[0].embedding.astype("float32"))
    
    # Se non abbiamo embedding, ritorna None (gestito dal chiamante)
    if not embeddings:
        return None
    
    # Media degli embedding e normalizza
    embeddings_array = np.array(embeddings)
    mean_embedding = np.mean(embeddings_array, axis=0)
    normalized = _normalize(mean_embedding)
    
    logger.info(f"[ROBUST_EMB] Generated {len(embeddings)} variant embeddings, averaged and normalized")
    
    return normalized

def _generate_selfie_embeddings_variants(img: np.ndarray) -> List[np.ndarray]:
    """
    Genera i 7 embeddings normalizzati delle varianti del selfie (non la media).
    
    Varianti:
    1. Originale
    2. Crop leggero (5% da ogni lato)
    3. Brightness +10%
    4. Brightness -10%
    5. Contrast +10%
    6. Sharpening leggero
    7. Flip orizzontale
    
    Returns:
        List[np.ndarray]: Lista di 7 embeddings normalizzati (può essere < 7 se alcune varianti non hanno facce)
    """
    assert face_app is not None
    
    # Converti OpenCV (BGR) a PIL (RGB) per le trasformazioni
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    embeddings = []
    
    # 1. Originale
    faces = face_app.get(img)
    if faces:
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )
        if faces_sorted:
            embeddings.append(_normalize(faces_sorted[0].embedding.astype("float32")))
    
    # 2. Crop leggero (5% da ogni lato)
    w, h = pil_img.size
    crop_box = (int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95))
    img_crop = pil_img.crop(crop_box)
    img_crop_cv = cv2.cvtColor(np.array(img_crop), cv2.COLOR_RGB2BGR)
    faces = face_app.get(img_crop_cv)
    if faces:
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )
        if faces_sorted:
            embeddings.append(_normalize(faces_sorted[0].embedding.astype("float32")))
    
    # 3. Brightness +10%
    enhancer = ImageEnhance.Brightness(pil_img)
    img_bright_plus = enhancer.enhance(1.1)
    img_bright_plus_cv = cv2.cvtColor(np.array(img_bright_plus), cv2.COLOR_RGB2BGR)
    faces = face_app.get(img_bright_plus_cv)
    if faces:
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )
        if faces_sorted:
            embeddings.append(_normalize(faces_sorted[0].embedding.astype("float32")))
    
    # 4. Brightness -10%
    enhancer = ImageEnhance.Brightness(pil_img)
    img_bright_minus = enhancer.enhance(0.9)
    img_bright_minus_cv = cv2.cvtColor(np.array(img_bright_minus), cv2.COLOR_RGB2BGR)
    faces = face_app.get(img_bright_minus_cv)
    if faces:
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )
        if faces_sorted:
            embeddings.append(_normalize(faces_sorted[0].embedding.astype("float32")))
    
    # 5. Contrast +10%
    enhancer = ImageEnhance.Contrast(pil_img)
    img_contrast = enhancer.enhance(1.1)
    img_contrast_cv = cv2.cvtColor(np.array(img_contrast), cv2.COLOR_RGB2BGR)
    faces = face_app.get(img_contrast_cv)
    if faces:
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )
        if faces_sorted:
            embeddings.append(_normalize(faces_sorted[0].embedding.astype("float32")))
    
    # 6. Sharpening leggero
    img_sharp = pil_img.filter(ImageFilter.SHARPEN)
    img_sharp_cv = cv2.cvtColor(np.array(img_sharp), cv2.COLOR_RGB2BGR)
    faces = face_app.get(img_sharp_cv)
    if faces:
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )
        if faces_sorted:
            embeddings.append(_normalize(faces_sorted[0].embedding.astype("float32")))
    
    # 7. Flip orizzontale
    img_flip = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    img_flip_cv = cv2.cvtColor(np.array(img_flip), cv2.COLOR_RGB2BGR)
    faces = face_app.get(img_flip_cv)
    if faces:
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )
        if faces_sorted:
            embeddings.append(_normalize(faces_sorted[0].embedding.astype("float32")))
    
    logger.info(f"[VARIANTS_EMB] Generated {len(embeddings)} variant embeddings (normalized)")
    
    return embeddings

def _generate_occlusion_variants(img: np.ndarray) -> List[np.ndarray]:
    """
    Genera 5 embeddings normalizzati per varianti con occlusioni (cappelli/occhiali/ombra).
    
    Varianti:
    1. Random Erasing 1: maschera rettangolare su 20-30% del volto
    2. Random Erasing 2: maschera rettangolare su 25-35% del volto
    3. Simula occhiali: banda orizzontale scura su zona occhi
    4. Simula cappello: banda scura su fronte
    5. Low light gamma + shadow
    
    Returns:
        List[np.ndarray]: Lista di 5 embeddings normalizzati (può essere < 5 se alcune varianti non hanno facce)
    """
    assert face_app is not None
    import random
    
    # Converti OpenCV (BGR) a PIL (RGB) per le trasformazioni
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    w, h = pil_img.size
    
    embeddings = []
    
    # 1. Random Erasing 1: maschera rettangolare su 20-30% del volto
    img_erased1 = pil_img.copy()
    erase_w = int(w * random.uniform(0.20, 0.30))
    erase_h = int(h * random.uniform(0.20, 0.30))
    erase_x = random.randint(0, w - erase_w)
    erase_y = random.randint(0, h - erase_h)
    # Riempi con colore casuale (simula occlusione)
    fill_color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
    for x in range(erase_x, min(erase_x + erase_w, w)):
        for y in range(erase_y, min(erase_y + erase_h, h)):
            img_erased1.putpixel((x, y), fill_color)
    img_erased1_cv = cv2.cvtColor(np.array(img_erased1), cv2.COLOR_RGB2BGR)
    faces = face_app.get(img_erased1_cv)
    if faces:
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )
        if faces_sorted:
            embeddings.append(_normalize(faces_sorted[0].embedding.astype("float32")))
    
    # 2. Random Erasing 2: maschera rettangolare su 25-35% del volto
    img_erased2 = pil_img.copy()
    erase_w = int(w * random.uniform(0.25, 0.35))
    erase_h = int(h * random.uniform(0.25, 0.35))
    erase_x = random.randint(0, w - erase_w)
    erase_y = random.randint(0, h - erase_h)
    fill_color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
    for x in range(erase_x, min(erase_x + erase_w, w)):
        for y in range(erase_y, min(erase_y + erase_h, h)):
            img_erased2.putpixel((x, y), fill_color)
    img_erased2_cv = cv2.cvtColor(np.array(img_erased2), cv2.COLOR_RGB2BGR)
    faces = face_app.get(img_erased2_cv)
    if faces:
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )
        if faces_sorted:
            embeddings.append(_normalize(faces_sorted[0].embedding.astype("float32")))
    
    # 3. Simula occhiali: banda orizzontale scura su zona occhi (circa 20% altezza, centro verticale)
    img_glasses = pil_img.copy()
    glasses_y_start = int(h * 0.35)
    glasses_y_end = int(h * 0.55)
    glasses_color = (20, 20, 20)  # Scuro
    for x in range(w):
        for y in range(glasses_y_start, glasses_y_end):
            img_glasses.putpixel((x, y), glasses_color)
    img_glasses_cv = cv2.cvtColor(np.array(img_glasses), cv2.COLOR_RGB2BGR)
    faces = face_app.get(img_glasses_cv)
    if faces:
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )
        if faces_sorted:
            embeddings.append(_normalize(faces_sorted[0].embedding.astype("float32")))
    
    # 4. Simula cappello: banda scura su fronte (circa 15% altezza, in alto)
    img_hat = pil_img.copy()
    hat_y_end = int(h * 0.15)
    hat_color = (15, 15, 15)  # Molto scuro
    for x in range(w):
        for y in range(hat_y_end):
            img_hat.putpixel((x, y), hat_color)
    img_hat_cv = cv2.cvtColor(np.array(img_hat), cv2.COLOR_RGB2BGR)
    faces = face_app.get(img_hat_cv)
    if faces:
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )
        if faces_sorted:
            embeddings.append(_normalize(faces_sorted[0].embedding.astype("float32")))
    
    # 5. Low light gamma + shadow
    # Applica gamma correction per simulare low light
    img_array = np.array(pil_img).astype(np.float32) / 255.0
    gamma = 1.8  # Più scuro
    img_array = np.power(img_array, gamma)
    img_array = (img_array * 255.0).astype(np.uint8)
    img_lowlight = Image.fromarray(img_array)
    # Aggiungi shadow (angolo in alto a sinistra)
    shadow_y_end = int(h * 0.4)
    shadow_x_end = int(w * 0.4)
    for x in range(shadow_x_end):
        for y in range(shadow_y_end):
            pixel = img_lowlight.getpixel((x, y))
            shadow_pixel = tuple(max(0, c - 40) for c in pixel)
            img_lowlight.putpixel((x, y), shadow_pixel)
    img_lowlight_cv = cv2.cvtColor(np.array(img_lowlight), cv2.COLOR_RGB2BGR)
    faces = face_app.get(img_lowlight_cv)
    if faces:
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )
        if faces_sorted:
            embeddings.append(_normalize(faces_sorted[0].embedding.astype("float32")))
    
    logger.info(f"[OCCLUSION_EMB] Generated {len(embeddings)} occlusion variant embeddings (normalized)")
    
    return embeddings

# Cache per overlay watermark (chiave: (width, height) -> overlay Image)
# NOTA: Cache svuotata per forzare rigenerazione con nuovo watermark "MetaProos"
_watermark_overlay_cache: Dict[Tuple[int, int], Image.Image] = {}

# Cache per logo watermark (chiave: target_size -> Image)
_logo_watermark_cache: Dict[int, Image.Image] = {}

def _load_logo_for_watermark(target_size: int) -> Optional[Image.Image]:
    """Carica logo Metaproos bianco e lo prepara per watermark con opacità 40-50%"""
    # Controlla cache
    if target_size in _logo_watermark_cache:
        return _logo_watermark_cache[target_size].copy()
    
    # Carica logo da metaproos-mark-3600.png
    logo_path = BASE_DIR / "assets" / "branding" / "metaproos-mark-3600.png"
    
    if not logo_path.exists():
        logger.warning(f"Logo not found: {logo_path}")
        return None
    
    try:
        logo_img = Image.open(logo_path)
        # Converti in RGBA se necessario
        if logo_img.mode != 'RGBA':
            logo_img = logo_img.convert('RGBA')
    except Exception as e:
        logger.error(f"Error loading logo from {logo_path}: {e}")
        return None
    
    # Ridimensiona mantenendo aspect ratio
    logo_img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Applica opacità 70% per rendere il logo più visibile nei punti di incrocio (alpha 179 = 70% opaco) - aumentato del 20%
    alpha_target = 179  # 70% opacità, 30% trasparenza - aumentato del 20%
    logo_data = logo_img.getdata()
    new_data = []
    for item in logo_data:
        if len(item) == 4:
            r, g, b, a = item
            # Combina alpha esistente con alpha target, converti tutto a bianco
            new_alpha = int((a / 255.0) * alpha_target)
            new_data.append((255, 255, 255, new_alpha))
        else:
            new_data.append((255, 255, 255, alpha_target))
    
    logo_img.putdata(new_data)
    
    # Salva in cache
    _logo_watermark_cache[target_size] = logo_img.copy()
    return logo_img

def _create_watermark_overlay(width: int, height: int) -> Image.Image:
    """Watermark con linee diagonali a X (senza incrocio) e testo MetaProos ripetuto.
    - Testo fisso: "MetaProos" (M e P maiuscole) e NON deve mai cambiare.
    - Pattern ripetuto su tutta l'immagine.
    - Due linee diagonali che suggeriscono una X ma lasciano un gap centrale.
    - La scritta è orizzontale e posizionata nel punto in cui le linee si incrocierebbero.
    """
    from pathlib import Path
    from PIL import Image, ImageDraw, ImageFont

    WATERMARK_TEXT = "MetaProos"  # FISSO: non usare APP_NAME, env, config, ecc.

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    base = max(260, int(min(width, height) * 0.28))
    step_x = base
    step_y = int(base * 0.62)

    line_alpha = 90
    text_alpha = 110
    line_width = max(2, int(min(width, height) / 900))

    font_size = max(28, int(min(width, height) / 18))
    font = None
    for fp in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        try:
            if Path(fp).exists():
                font = ImageFont.truetype(fp, font_size)
                break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()

    try:
        bbox = draw.textbbox((0, 0), WATERMARK_TEXT, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        text_w, text_h = draw.textsize(WATERMARK_TEXT, font=font)

    seg_len = int(base * 0.42)
    gap = int(max(text_w, text_h) * 0.75)

    start_x = -step_x
    start_y = -step_y

    for y in range(start_y, height + step_y, step_y):
        x_offset = (step_x // 2) if ((y // step_y) % 2) else 0
        for x in range(start_x, width + step_x, step_x):
            cx = x + x_offset
            cy = y

            # \\ spezzata
            x1a, y1a = cx - seg_len, cy - seg_len
            x1b, y1b = cx - gap // 2, cy - gap // 2
            x1c, y1c = cx + gap // 2, cy + gap // 2
            x1d, y1d = cx + seg_len, cy + seg_len

            # / spezzata
            x2a, y2a = cx - seg_len, cy + seg_len
            x2b, y2b = cx - gap // 2, cy + gap // 2
            x2c, y2c = cx + gap // 2, cy - gap // 2
            x2d, y2d = cx + seg_len, cy - seg_len

            draw.line([(x1a, y1a), (x1b, y1b)], fill=(0, 0, 0, line_alpha), width=line_width)
            draw.line([(x1c, y1c), (x1d, y1d)], fill=(0, 0, 0, line_alpha), width=line_width)
            draw.line([(x2a, y2a), (x2b, y2b)], fill=(0, 0, 0, line_alpha), width=line_width)
            draw.line([(x2c, y2c), (x2d, y2d)], fill=(0, 0, 0, line_alpha), width=line_width)

            tx = cx - text_w // 2
            ty = cy - text_h // 2
            draw.text((tx, ty), WATERMARK_TEXT, font=font, fill=(0, 0, 0, text_alpha))

    return overlay

def _add_watermark(image_path: Path) -> bytes:
    """Aggiunge watermark pattern premium a griglia (stile GetPica) con logo Metaproos, linee e nodi"""
    logger.info(f"WATERMARK DEBUG: _add_watermark called for {image_path}")
    logger.info(f"WATERMARK DEBUG: This function will use text='MetaProos' (forced, never 'MetaProos' variants)")
    try:
        # Apri immagine con Pillow
        img = Image.open(image_path)
        
        # Converti in RGB se necessario
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Converti in RGBA per watermark
        img_rgba = img.convert('RGBA')
        
        # Crea overlay watermark (con cache)
        watermark_overlay = _create_watermark_overlay(img.width, img.height)
        
        # Combina watermark con immagine
        img_with_watermark = Image.alpha_composite(img_rgba, watermark_overlay).convert('RGB')
        
        # Salva in bytes
        output = io.BytesIO()
        img_with_watermark.save(output, format='JPEG', quality=88)
        output.seek(0)
        return output.getvalue()
    
    except Exception as e:
        logger.error(f"Error adding watermark: {e}", exc_info=True)
        # Fallback: ritorna immagine originale
        with open(image_path, 'rb') as f:
            return f.read()

def _add_watermark_from_bytes(image_bytes: bytes) -> bytes:
    """Aggiunge watermark a un'immagine da bytes (per foto da R2)"""
    try:
        # Apri immagine da bytes con Pillow
        img = Image.open(io.BytesIO(image_bytes))
        
        # Converti in RGB se necessario
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Converti in RGBA per watermark
        img_rgba = img.convert('RGBA')
        
        # Crea overlay watermark (con cache)
        watermark_overlay = _create_watermark_overlay(img.width, img.height)
        
        # Combina watermark con immagine
        img_with_watermark = Image.alpha_composite(img_rgba, watermark_overlay).convert('RGB')
        
        # Salva in bytes
        output = io.BytesIO()
        img_with_watermark.save(output, format='JPEG', quality=88)
        output.seek(0)
        return output.getvalue()
    
    except Exception as e:
        logger.error(f"Error adding watermark from bytes: {e}", exc_info=True)
        # Fallback: ritorna immagine originale
        return image_bytes

def _generate_thumb(image_bytes: bytes) -> bytes:
    """Genera thumbnail JPEG da bytes (max 600px lato lungo, qualità 70)"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Calcola dimensioni mantenendo aspect ratio
        w, h = img.size
        max_side = THUMB_MAX_SIDE
        if w > h:
            new_w = max_side
            new_h = int(h * (max_side / w))
        else:
            new_h = max_side
            new_w = int(w * (max_side / h))
        
        # Resize
        img.thumbnail((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Salva in bytes
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=THUMB_QUALITY, optimize=True)
        output.seek(0)
        return output.getvalue()
    except Exception as e:
        logger.error(f"Error generating thumb: {e}", exc_info=True)
        raise

def _generate_wm_preview(image_bytes: bytes) -> bytes:
    """Genera versione watermarked precomputata (max 2000px lato lungo, qualità 75)"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Ridimensiona se troppo grande (max 2000px lato lungo)
        w, h = img.size
        max_side = WM_MAX_SIDE
        if w > max_side or h > max_side:
            if w > h:
                new_w = max_side
                new_h = int(h * (max_side / w))
            else:
                new_h = max_side
                new_w = int(w * (max_side / h))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Converti in RGBA per watermark
        img_rgba = img.convert('RGBA')
        
        # Crea overlay watermark
        watermark_overlay = _create_watermark_overlay(img_rgba.width, img_rgba.height)
        
        # Combina watermark con immagine
        img_with_watermark = Image.alpha_composite(img_rgba, watermark_overlay).convert('RGB')
        
        # Salva in bytes
        output = io.BytesIO()
        img_with_watermark.save(output, format='JPEG', quality=WM_QUALITY, optimize=True)
        output.seek(0)
        return output.getvalue()
    except Exception as e:
        logger.error(f"Error generating wm preview: {e}", exc_info=True)
        raise

def _get_r2_public_url(r2_key: str) -> str:
    """
    Costruisce l'URL pubblico per un oggetto R2.
    
    Args:
        r2_key: Chiave R2 dell'oggetto (es: "thumbs/IMG_2914.jpg", "wm/IMG_2914.jpg", "IMG_2914.jpg")
    
    Returns:
        URL pubblico completo (es: "https://pub-xxxxx.r2.dev/thumbs/IMG_2914.jpg")
    """
    if not R2_PUBLIC_BASE_URL:
        raise ValueError("R2_PUBLIC_BASE_URL not configured")
    
    # Rimuovi trailing slash se presente
    base_url = R2_PUBLIC_BASE_URL.rstrip('/')
    # Rimuovi leading slash da r2_key se presente
    key = r2_key.lstrip('/')
    
    return f"{base_url}/{key}"

async def ensure_previews_for_photo(r2_key: str) -> tuple[bool, bool]:
    """
    Assicura che thumb e wm preview esistano per una foto.
    
    Args:
        r2_key: Chiave R2 della foto (filename semplice, es. "IMG_1016.jpg")
    
    Returns:
        (thumb_created, wm_created): True se creato in questa chiamata, False se già esisteva
    """
    # Ignora se è già un thumb o wm
    if r2_key.startswith(THUMB_PREFIX) or r2_key.startswith(WM_PREFIX):
        return False, False
    
    if not USE_R2 or r2_client is None:
        return False, False
    
    # Estrai filename base (rimuovi eventuali path)
    from pathlib import Path
    filename = Path(r2_key).name
    
    # Costruisci chiavi thumb/wm semplici
    thumb_key = f"{THUMB_PREFIX}{filename}"
    wm_key = f"{WM_PREFIX}{filename}"
    thumb_created = False
    wm_created = False
    original_bytes = None
    
    # Verifica thumb (skip intelligente)
    try:
        r2_client.head_object(Bucket=R2_BUCKET, Key=thumb_key)
        # Esiste già, skip
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code in ('404', 'NoSuchKey'):
            # Non esiste, genera
            try:
                # Scarica originale
                original_bytes = await _r2_get_object_bytes(r2_key, mark_missing=True)
                # Genera thumb
                thumb_bytes = _generate_thumb(original_bytes)
                # Upload su R2
                r2_client.put_object(Bucket=R2_BUCKET, Key=thumb_key, Body=thumb_bytes, ContentType='image/jpeg')
                thumb_created = True
                logger.info(f"[PREVIEW] Generated thumb: {thumb_key}")
            except Exception as e:
                logger.error(f"[PREVIEW] Error generating thumb for {r2_key}: {e}")
        else:
            logger.error(f"[PREVIEW] R2 error checking thumb {thumb_key}: {e}")
    except Exception as e:
        logger.error(f"[PREVIEW] Unexpected error checking thumb {thumb_key}: {e}")
    
    # Verifica wm (skip intelligente)
    try:
        r2_client.head_object(Bucket=R2_BUCKET, Key=wm_key)
        # Esiste già, skip
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code in ('404', 'NoSuchKey'):
            # Non esiste, genera
            try:
                # Scarica originale se non già scaricato
                if original_bytes is None:
                    original_bytes = await _r2_get_object_bytes(r2_key, mark_missing=True)
                # Genera wm
                wm_bytes = _generate_wm_preview(original_bytes)
                # Upload su R2
                r2_client.put_object(Bucket=R2_BUCKET, Key=wm_key, Body=wm_bytes, ContentType='image/jpeg')
                wm_created = True
                logger.info(f"[PREVIEW] Generated wm: {wm_key}")
            except Exception as e:
                logger.error(f"[PREVIEW] Error generating wm for {r2_key}: {e}")
        else:
            logger.error(f"[PREVIEW] R2 error checking wm {wm_key}: {e}")
    except Exception as e:
        logger.error(f"[PREVIEW] Unexpected error checking wm {wm_key}: {e}")
    
    return thumb_created, wm_created

def _normalize_photo_key(name: str) -> str:
    """
    Normalizza una chiave foto rimuovendo tutti i prefissi (wm/, thumbs/) in modo robusto.
    
    Args:
        name: Nome della foto che può contenere prefissi (es. "wm/Alessione.JPG", "thumbs/thumbs/_MIT0161.jpg")
    
    Returns:
        Nome base della foto senza prefissi (es. "Alessione.JPG", "_MIT0161.jpg")
    """
    if not name:
        return ""
    s = name.strip().lstrip("/")
    # Rimuovi prefissi ripetuti finché ce ne sono
    while True:
        if s.startswith("wm/"):
            s = s[3:]
            continue
        if s.startswith("thumbs/"):
            s = s[7:]
            continue
        break
    return s

async def _mark_photo_missing_in_r2(photo_id: str):
    """Marca una foto come mancante in R2 (404) - aggiorna DB"""
    try:
        await _db_execute_write(
            """UPDATE user_photos 
               SET status = 'deleted', r2_exists = FALSE, r2_last_checked = NOW()
               WHERE photo_id = $1 AND (status != 'deleted' OR r2_exists = TRUE)""",
            (photo_id,)
        )
        logger.info(f"[R2_404] Marked photo as deleted (missing in R2): {photo_id}")
    except Exception as e:
        logger.error(f"Error marking photo {photo_id} as missing in R2: {e}")

async def _r2_get_object_bytes(key: str, mark_missing: bool = True) -> bytes:
    """Legge un oggetto da R2 e restituisce i bytes. Se 404, marca come deleted nel DB (solo se mark_missing=True)."""
    if not USE_R2 or r2_client is None:
        raise ValueError("R2 not configured")
    
    try:
        response = r2_client.get_object(Bucket=R2_BUCKET, Key=key)
        return response['Body'].read()
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        
        if error_code == 'NoSuchKey':
            logger.warning(f"Photo not found in R2: key={key}")
            # Marca come deleted nel DB solo se mark_missing=True (non per thumb/wm)
            if mark_missing and not key.startswith(THUMB_PREFIX) and not key.startswith(WM_PREFIX):
                await _mark_photo_missing_in_r2(key)
            raise HTTPException(status_code=404, detail=f"Photo not found: {key}")
        else:
            logger.error(f"R2 error reading photo: code={error_code}, message={error_message}, key={key}")
            raise HTTPException(status_code=500, detail=f"Error reading photo from R2")
    except Exception as e:
        logger.error(f"Unexpected error reading from R2: {type(e).__name__}: {e}, key={key}")
        raise HTTPException(status_code=500, detail=f"Error reading photo from R2")

def _ensure_ready():
    """Verifica che face_app sia caricato (indice può essere vuoto)"""
    if face_app is None:
        raise HTTPException(status_code=503, detail="Face recognition not initialized")
    # Indice può essere vuoto - non fallire se None o vuoto

@app.on_event("startup")
async def startup():
    """Carica il modello e l'indice all'avvio"""
    global face_app, faiss_index, meta_rows, db_pool
    
    # Inizializza database PostgreSQL all'avvio
    logger.info("=" * 80)
    logger.info("🗄️  DATABASE CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"POSTGRES_AVAILABLE: {POSTGRES_AVAILABLE}")
    logger.info(f"DATABASE_URL present: {bool(DATABASE_URL)}")
    if DATABASE_URL:
        # Mostra solo i primi caratteri per sicurezza
        masked_url = DATABASE_URL[:20] + "..." if len(DATABASE_URL) > 20 else DATABASE_URL
        logger.info(f"DATABASE_URL: {masked_url}")
    
    logger.info("✅ Using PostgreSQL database (obbligatorio)")
    await _init_database()
    logger.info("✅ PostgreSQL database initialized and ready")
    logger.info("=" * 80)
    
    # Log R2_ONLY_MODE
    # Leggi R2_ONLY_MODE come variabile locale per evitare UnboundLocalError
    r2_only_mode = str(os.getenv("R2_ONLY_MODE", "0")).strip().lower() in {"1","true","yes","on"}
    logger.info("=" * 80)
    logger.info("📦 R2_ONLY_MODE CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"R2_ONLY_MODE enabled: {r2_only_mode}")
    if r2_only_mode:
        logger.info("✅ R2_ONLY_MODE: Filesystem disabled for photos and index files")
        logger.info("   - Photos served ONLY from R2")
        logger.info("   - Index files (faces.index, faces.meta.jsonl, downloads_track.jsonl) stored on R2")
    else:
        logger.info("⚠️  R2_ONLY_MODE disabled: Filesystem fallback enabled")
    logger.info("=" * 80)
    
    logger.info("Loading face recognition model...")
    try:
        face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        # Aumentato det_size per migliorare detection su profilo/ombra
        face_app.prepare(ctx_id=0, det_size=(1024, 1024))
        logger.info("Face recognition model loaded")
    except Exception as e:
        logger.error(f"Error loading face model: {e}")
        face_app = None
        return
    
    # In R2_ONLY_MODE: carica indice e metadata da R2
    if r2_only_mode:
        logger.info("R2_ONLY_MODE: Loading index files from R2...")
        if USE_R2 and r2_client:
            try:
                # Carica indice FAISS da R2
                logger.info("Loading FAISS index from R2...")
                index_bytes = await _r2_get_object_bytes("faces.index")
                # Salva temporaneamente in memoria e carica con FAISS
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(index_bytes)
                    tmp_path = tmp_file.name
                try:
                    faiss_index = faiss.read_index(tmp_path)
                    logger.info(f"FAISS index loaded from R2: {faiss_index.ntotal} vectors")
                finally:
                    # Pulisci file temporaneo
                    import os as os_module
                    try:
                        os_module.unlink(tmp_path)
                    except:
                        pass
            except HTTPException as e:
                if e.status_code == 404:
                    logger.info("FAISS index not found in R2 - starting with empty index")
                    # Crea indice vuoto (non disabilita il match, solo parte vuoto)
                    faiss_index = faiss.IndexFlatIP(INDEX_DIM)
                    meta_rows = []
                    # Salva indice vuoto su R2
                    try:
                        import tempfile
                        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                            tmp_path = tmp_file.name
                        faiss.write_index(faiss_index, tmp_path)
                        with open(tmp_path, 'rb') as f:
                            index_bytes = f.read()
                        r2_client.put_object(Bucket=R2_BUCKET, Key="faces.index", Body=index_bytes)
                        logger.info("Empty FAISS index created and uploaded to R2")
                        import os as os_module
                        os_module.unlink(tmp_path)
                    except Exception as save_err:
                        logger.warning(f"Could not save empty index to R2 (will be created on first indexing): {save_err}")
                else:
                    logger.error(f"Error loading FAISS index from R2: {e}")
                    faiss_index = None
            except Exception as e:
                logger.error(f"Error loading FAISS index from R2: {e}")
                # Crea indice vuoto anche in caso di errore generico
                logger.info("Creating empty FAISS index as fallback")
                faiss_index = faiss.IndexFlatIP(INDEX_DIM)
                meta_rows = []
            
            try:
                # Carica metadata da R2
                logger.info("Loading metadata from R2...")
                meta_bytes = await _r2_get_object_bytes("faces.meta.jsonl")
                meta_text = meta_bytes.decode('utf-8')
                meta_rows = []
                for line in meta_text.strip().split('\n'):
                    if line.strip():
                        meta_rows.append(json.loads(line))
                logger.info(f"Metadata loaded from R2: {len(meta_rows)} records")
            except HTTPException as e:
                if e.status_code == 404:
                    logger.info("Metadata not found in R2 - starting with empty metadata")
                    meta_rows = []
                    # Salva metadata vuoto su R2
                    try:
                        r2_client.put_object(Bucket=R2_BUCKET, Key="faces.meta.jsonl", Body=b"")
                        logger.info("Empty metadata file created and uploaded to R2")
                    except Exception as save_err:
                        logger.warning(f"Could not save empty metadata to R2 (will be created on first indexing): {save_err}")
                else:
                    logger.error(f"Error loading metadata from R2: {e}")
                    meta_rows = []
            except Exception as e:
                logger.error(f"Error loading metadata from R2: {e}")
                meta_rows = []
            
            
        else:
            # OPZIONE 1: Fallback automatico a filesystem mode
            logger.warning("=" * 80)
            logger.warning("⚠️  R2_ONLY_MODE=True ma R2 non configurato!")
            logger.warning("⚠️  Fallback automatico a filesystem mode (R2_ONLY_MODE=False)")
            logger.warning("=" * 80)
            r2_only_mode = False
            # Continua con il caricamento da filesystem (vedi else sotto)
    else:
        # Carica indice FAISS e metadata (solo se R2_ONLY_MODE è disabilitato)
        if not INDEX_PATH.exists() or not META_PATH.exists():
            logger.info("Index files not found - creating empty index")
            # Crea indice vuoto
            faiss_index = faiss.IndexFlatIP(INDEX_DIM)
            meta_rows = []
            # Salva indice vuoto su filesystem
            try:
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                faiss.write_index(faiss_index, str(INDEX_PATH))
                META_PATH.write_text("", encoding='utf-8')
                logger.info("Empty FAISS index created on filesystem")
            except Exception as save_err:
                logger.error(f"Error saving empty index: {save_err}")
        else:
            try:
                logger.info(f"Loading FAISS index from {INDEX_PATH}")
                faiss_index = faiss.read_index(str(INDEX_PATH))
                logger.info(f"FAISS index loaded: {faiss_index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                faiss_index = None
            
            try:
                logger.info(f"Loading metadata from {META_PATH}")
                meta_rows = _load_meta_jsonl(META_PATH)
                logger.info(f"Metadata loaded: {len(meta_rows)} records")
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
                meta_rows = []
    
    # Inizializza indice vuoto se non esiste
    if faiss_index is None:
        logger.info("Initializing empty FAISS index")
        faiss_index = faiss.IndexFlatIP(INDEX_DIM)
        meta_rows = []
    
    # Esegui cleanup iniziale
    logger.info("Running initial cleanup...")
    await _cleanup_expired_photos()
    
    # Inizializza lock globale per indicizzazione (evita run simultanei)
    global indexing_lock
    if indexing_lock is None:
        indexing_lock = asyncio.Lock()
    
    # Funzione per indicizzare nuove foto da R2
    async def index_new_r2_photos():
        """FULL REBUILD: Ricostruisce completamente l'indice FAISS e metadata da R2 (solo per uso admin manuale)"""
        global faiss_index, meta_rows, _r2_keys_cache
        
        logger.info(f"[INDEXING] Checking conditions: USE_R2={USE_R2}, r2_client={r2_client is not None}, R2_ONLY_MODE={R2_ONLY_MODE}")
        if not USE_R2 or not r2_client or not R2_ONLY_MODE:
            logger.warning(f"[INDEXING] Skipping - conditions not met: USE_R2={USE_R2}, r2_client={r2_client is not None}, R2_ONLY_MODE={R2_ONLY_MODE}")
            return
        
        # Evita run simultanei
        if indexing_lock.locked():
            logger.info("[INDEXING] Already running, skipping")
            return
        
        async with indexing_lock:
            try:
                start_time = datetime.now(timezone.utc)
                logger.info("[INDEXING] FULL REBUILD start")
                
                # ========== STEP 1: Lista tutte le foto originali in R2 ==========
                paginator = r2_client.get_paginator('list_objects_v2')
                prefix = R2_PHOTOS_PREFIX if R2_PHOTOS_PREFIX else ""
                
                original_photos = []
                for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=prefix):
                    for obj in page.get('Contents', []):
                        key = obj['Key']
                        
                        # Escludi file di sistema
                        if key.startswith("faces.") or key.startswith("downloads_track"):
                            continue
                        # Escludi preview (wm/ e thumbs/)
                        if key.startswith("wm/") or key.startswith("thumbs/"):
                            continue
                        # Include solo immagini originali
                        if not any(key.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.heic']):
                            continue
                        
                        # NON normalizzare: usa la chiave R2 completa come r2_key
                        original_photos.append(key)
                
                logger.info(f"[INDEXING] originals_in_r2={len(original_photos)}")
                
                # ========== STEP 2: Se lista vuota, reset index ==========
                if len(original_photos) == 0:
                    # Crea indice vuoto
                    faiss_index = faiss.IndexFlatIP(INDEX_DIM)
                    meta_rows = []
                    
                    # Salva su R2 (sovrascrivi)
                    try:
                        import tempfile
                        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                            tmp_path = tmp_file.name
                        faiss.write_index(faiss_index, tmp_path)
                        with open(tmp_path, 'rb') as f:
                            index_bytes = f.read()
                        r2_client.put_object(Bucket=R2_BUCKET, Key="faces.index", Body=index_bytes)
                        
                        # Salva metadata vuoto
                        r2_client.put_object(Bucket=R2_BUCKET, Key="faces.meta.jsonl", Body=b"")
                        
                        import os as os_module
                        os_module.unlink(tmp_path)
                        
                        # Invalida cache
                        _r2_keys_cache = None
                        
                        logger.info("[INDEXING] FULL REBUILD: 0 originals in R2 -> index reset to 0")
                        logger.info("[INDEXING] FULL REBUILD done, saved index+meta to R2")
                        return 0
                    except Exception as e:
                        logger.error(f"[INDEXING] Error saving empty index to R2: {e}")
                        return 0
                
                # ========== STEP 3: FULL REBUILD da zero ==========
                # Crea nuovo indice vuoto
                new_faiss_index = faiss.IndexFlatIP(INDEX_DIM)
                new_meta_rows = []
                faces_total = 0
                previews_generated_count = 0
                
                # Processa tutte le foto originali
                for photo_idx, r2_key in enumerate(original_photos):
                    try:
                        # Estrai display_name (basename)
                        from pathlib import Path
                        display_name = Path(r2_key).name
                        
                        # Scarica foto da R2
                        photo_bytes = await _r2_get_object_bytes(r2_key)
                        
                        # Decodifica immagine
                        nparr = np.frombuffer(photo_bytes, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if img is None:
                            continue
                        
                        # Estrai volti
                        faces = face_app.get(img)
                        
                        if not faces:
                            # Foto senza volti -> salta
                            continue
                        
                        # Per ogni volto, crea embedding e metadata
                        for face in faces:
                            embedding = face.embedding.astype(np.float32)
                            embedding = _normalize(embedding)
                            
                            # Aggiungi embedding al nuovo indice
                            embedding_reshaped = embedding.reshape(1, -1)
                            new_faiss_index.add(embedding_reshaped)
                            
                            # Metadata con r2_key e display_name
                            bbox = face.bbox
                            new_meta_rows.append({
                                "r2_key": r2_key,  # Chiave R2 completa
                                "display_name": display_name,  # Solo nome file
                                "photo_id": r2_key,  # Compatibilità retroattiva
                                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                                "det_score": float(face.det_score)
                            })
                            faces_total += 1
                        
                        # Genera preview (rate limited: max 3 foto per ciclo)
                        # IMPORTANTE: se fallisce la preview, non bloccare il rebuild, logga e continua
                        if photo_idx < 3:
                            try:
                                thumb_created, wm_created = await ensure_previews_for_photo(r2_key)
                                if thumb_created:
                                    previews_generated_count += 1
                                if wm_created:
                                    previews_generated_count += 1
                            except Exception as preview_err:
                                logger.error(f"[INDEXING] Failed to generate previews for {r2_key}: {preview_err}, continuing rebuild")
                                # Non bloccare il rebuild, continua con la prossima foto
                        
                    except Exception as e:
                        logger.error(f"[INDEXING] Error processing photo {r2_key}: {e}")
                        continue
                
                # ========== STEP 4: Aggiorna variabili globali ==========
                faiss_index = new_faiss_index
                meta_rows = new_meta_rows
                
                vectors_in_index = faiss_index.ntotal
                meta_rows_count = len(meta_rows)
                
                logger.info(f"[INDEXING] faces_total={faces_total} vectors_in_index={vectors_in_index} meta_rows={meta_rows_count}")
                
                # ========== STEP 5: Salva su R2 (sovrascrivi sempre) ==========
                try:
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                    faiss.write_index(faiss_index, tmp_path)
                    with open(tmp_path, 'rb') as f:
                        index_bytes = f.read()
                    r2_client.put_object(Bucket=R2_BUCKET, Key="faces.index", Body=index_bytes)
                    
                    # Salva metadata
                    meta_lines = [json.dumps(m, ensure_ascii=False) for m in meta_rows]
                    meta_bytes = '\n'.join(meta_lines).encode('utf-8')
                    r2_client.put_object(Bucket=R2_BUCKET, Key="faces.meta.jsonl", Body=meta_bytes)
                    
                    import os as os_module
                    os_module.unlink(tmp_path)
                    
                    # Invalida cache
                    _r2_keys_cache = None
                    
                    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                    logger.info(f"[INDEXING] FULL REBUILD done, saved index+meta to R2 (elapsed: {elapsed:.2f}s, previews: {previews_generated_count})")
                    
                except Exception as e:
                    logger.error(f"[INDEXING] Error saving rebuilt index to R2: {e}")
                    raise
                
            except Exception as e:
                logger.error(f"[INDEXING] Error in FULL REBUILD: {e}", exc_info=True)
            
            return len(original_photos)
    
    # Avvia task periodico per cleanup (ogni 6 ore) e indicizzazione (ogni 60 secondi)
    async def periodic_tasks():
        while True:
            try:
                await asyncio.sleep(6 * 60 * 60)  # 6 ore
                logger.info("Running periodic cleanup...")
                await _cleanup_expired_photos()
            except Exception as e:
                logger.error(f"Error in periodic tasks: {e}")
    
    async def indexing_task():
        """Task periodico per indicizzazione automatica (check veloce + sync incrementale se necessario)
        
        Funziona in background senza bloccare l'app:
        - Check ogni N secondi (default 60s) se R2 è cambiato
        - Usa hash veloce per evitare sync inutili
        - Sync incrementale solo quando necessario (add/remove foto)
        - Completamente asincrono e non-blocking
        """
        if not INDEXING_ENABLED:
            logger.info("[INDEXING] Periodic indexing task DISABLED (INDEXING_ENABLED=false)")
            return
        
        logger.info(f"[INDEXING] ✅ Auto-sync ENABLED - checking R2 every {INDEXING_INTERVAL_SECONDS}s")
        logger.info(f"[INDEXING] 📸 Just upload/delete photos on R2 - sync happens automatically in background!")
        
        while True:
            try:
                await asyncio.sleep(INDEXING_INTERVAL_SECONDS)
                # Check veloce (non blocca, usa hash per evitare sync inutili)
                logger.info(f"[INDEXING] Auto-sync check (interval={INDEXING_INTERVAL_SECONDS}s)")
                await maybe_sync_index()
            except Exception as e:
                logger.error(f"[INDEXING] Error in indexing task: {e}", exc_info=True)
                # Continua anche in caso di errore (non bloccare il task)
                await asyncio.sleep(10)  # Pausa breve prima di riprovare
    
    # Avvia task in background
    asyncio.create_task(periodic_tasks())
    if INDEXING_ENABLED:
        asyncio.create_task(indexing_task())
        logger.info(f"Periodic tasks started (cleanup every 6 hours, indexing ENABLED every {INDEXING_INTERVAL_SECONDS}s)")
    else:
        logger.info("Periodic tasks started (cleanup every 6 hours, indexing DISABLED - use /admin/index/sync)")
    
    # ============================================================
    # LOGGING DEFINITIVO: PATH ESATTI DEI FILE STATICI
    # ============================================================
    logger.info("=" * 80)
    logger.info("📁 STATIC FILES CONFIGURATION")
    logger.info("=" * 80)
    
    index_path = STATIC_DIR / "index.html"
    admin_path = STATIC_DIR / "admin.html"
    
    logger.info(f"STATIC_DIR (absolute): {STATIC_DIR.resolve()}")
    logger.info(f"STATIC_DIR exists: {STATIC_DIR.exists()}")
    logger.info("")
    logger.info(f"📄 index.html path: {index_path.resolve()}")
    logger.info(f"   index.html exists: {index_path.exists()}")
    if index_path.exists():
        logger.info(f"   index.html size: {index_path.stat().st_size} bytes")
        logger.info(f"   index.html modified: {datetime.fromtimestamp(index_path.stat().st_mtime).isoformat()}")
    else:
        logger.error(f"   ❌ index.html NOT FOUND!")
    logger.info("")
    logger.info(f"📄 admin.html path: {admin_path.resolve()}")
    logger.info(f"   admin.html exists: {admin_path.exists()}")
    if admin_path.exists():
        logger.info(f"   admin.html size: {admin_path.stat().st_size} bytes")
        logger.info(f"   admin.html modified: {datetime.fromtimestamp(admin_path.stat().st_mtime).isoformat()}")
    else:
        logger.error(f"   ❌ admin.html NOT FOUND!")
    logger.info("")
    logger.info(f"🌐 Serving static files from: /static -> {STATIC_DIR.resolve()}")
    logger.info(f"🏠 Home page will be served from: / -> {index_path.resolve()}")
    logger.info(f"🔐 Admin page will be served from: /admin -> {admin_path.resolve()}")
    logger.info("=" * 80)
    logger.info("✅ STARTUP COMPLETE")
    logger.info("=" * 80)

@app.on_event("shutdown")
async def shutdown():
    """Chiude il pool di connessioni PostgreSQL all'arresto"""
    global db_pool
    if db_pool:
        logger.info("Closing PostgreSQL connection pool...")
        await db_pool.close()
        logger.info("PostgreSQL connection pool closed")

# Handler HTTPException rimosso - già gestito sopra

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
    """Home page - serve index.html"""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        logger.error(f"❌ index.html not found at: {index_path.resolve()}")
        raise HTTPException(status_code=500, detail=f"index.html not found: {index_path}")
    logger.info(f"🏠 Serving index.html from: {index_path.resolve()}")
    return FileResponse(
        index_path,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.get("/album", response_class=HTMLResponse)
def album():
    """Pagina album con i risultati delle foto"""
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/test", response_class=HTMLResponse)
def test_page():
    """Pagina di test per verificare le funzionalità"""
    test_path = STATIC_DIR / "test.html"
    if not test_path.exists():
        raise HTTPException(status_code=404, detail="Test page not found")
    return FileResponse(
        test_path,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@app.get("/debug/watermark")
def debug_watermark():
    """Endpoint di debug per verificare quale testo viene usato nel watermark"""
    # Testa la funzione _create_watermark_overlay
    try:
        test_overlay = _create_watermark_overlay(1000, 1000)
        # Cerca il testo nel codice sorgente
        import inspect
        source = inspect.getsource(_create_watermark_overlay)
        watermark_text = None
        for line in source.split('\n'):
            if 'text =' in line and 'MetaProos' in line:
                watermark_text = 'MetaProos'
                break
            elif 'text =' in line and ('tenerife' in line.lower() or 'pictures' in line.lower()):
                watermark_text = 'ERRORE: testo watermark non valido!'
                break
        
        return {
            "watermark_text_in_code": watermark_text or "not found",
            "overlay_created": test_overlay is not None,
            "overlay_size": test_overlay.size if test_overlay else None,
            "function_file": str(Path(__file__).resolve()),
            "assertion": "MetaProos" if watermark_text == "MetaProos" else "ERROR: Wrong text!"
        }
    except Exception as e:
        return {"error": str(e), "traceback": str(e.__traceback__)}

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Ritorna 204 per favicon (i browser lo richiedono automaticamente)"""
    return Response(status_code=204)

@app.get("/apple-touch-icon.png", include_in_schema=False)
def apple_touch_icon():
    """Ritorna 204 per apple-touch-icon (i browser lo richiedono automaticamente)"""
    return Response(status_code=204)

@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
def apple_touch_icon_precomposed():
    """Ritorna 204 per apple-touch-icon-precomposed (i browser lo richiedono automaticamente)"""
    return Response(status_code=204)

@app.get("/health")
def health():
    """Endpoint di health check con informazioni R2 e index"""
    index_size = 0
    if faiss_index is not None:
        try:
            index_size = faiss_index.ntotal
        except Exception:
            pass
    
    r2_client_ready = r2_client is not None and USE_R2
    
    return {
        "status": "ok",
        "service": APP_NAME,
        "version": APP_VERSION,
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "r2_only_mode": R2_ONLY_MODE,
        "use_r2": USE_R2,
        "r2_client_ready": r2_client_ready,
        "index_size": index_size
    }

# Cache per hash delle keys (per check veloce)
_r2_keys_hash_cache = None

# Funzione globale per check veloce se R2 è cambiato (usata dal task periodico)
async def maybe_sync_index():
    """Check veloce se R2 è cambiato. Se sì, esegue sync incrementale. Se no, skip."""
    global faiss_index, meta_rows, indexing_lock, _r2_keys_hash_cache
    
    if not USE_R2 or not r2_client or not R2_ONLY_MODE:
        return
    
    # Anti-overlap: se sync già in corso, skip
    if indexing_lock.locked():
        logger.info("[INDEXING] Sync already running. Skipping this cycle.")
        return
    
    try:
        # ========== STEP 1: Lista veloce R2 keys (solo originali) con hash per check veloce ==========
        paginator = r2_client.get_paginator('list_objects_v2')
        
        # Se R2_PHOTOS_PREFIX è vuoto, non passare Prefix (lista tutto il bucket)
        paginate_kwargs = {"Bucket": R2_BUCKET}
        if R2_PHOTOS_PREFIX:
            paginate_kwargs["Prefix"] = R2_PHOTOS_PREFIX
        
        r2_originals_set = set()
        r2_keys_list = []  # Lista ordinata per hash
        
        for page in paginator.paginate(**paginate_kwargs):
            for obj in page.get('Contents', []):
                key = obj['Key']
                
                # Escludi file di sistema
                if key.startswith("faces.") or key.startswith("downloads_track"):
                    continue
                # Escludi preview (wm/ e thumbs/)
                if key.startswith("wm/") or key.startswith("thumbs/"):
                    continue
                # Include solo immagini originali
                # Se R2_PHOTOS_PREFIX è impostato, accetta solo keys che iniziano con quel prefix
                if R2_PHOTOS_PREFIX:
                    if not key.startswith(R2_PHOTOS_PREFIX):
                        continue
                
                # Filtro estensioni CASE-INSENSITIVE: .jpg .jpeg .png
                if not any(key.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                    continue
                
                # Estrai filename (basename) come r2_key
                from pathlib import Path
                filename = Path(key).name
                r2_originals_set.add(filename)
                r2_keys_list.append(filename)
        
        # Calcola hash veloce delle keys (solo per check, non per sync)
        import hashlib
        r2_keys_sorted = sorted(r2_keys_list)
        r2_keys_str = "|".join(r2_keys_sorted)
        r2_keys_hash = hashlib.md5(r2_keys_str.encode()).hexdigest()
        
        # ========== STEP 2: Check veloce con hash (evita sync se hash non cambia) ==========
        if _r2_keys_hash_cache == r2_keys_hash:
            # Hash identico = nessun cambiamento, skip sync
            logger.debug(f"[INDEXING] No changes detected (hash match). Skipping sync.")
            return
        
        # Hash diverso = cambiamenti rilevati, procedi con check dettagliato
        logger.info(f"[INDEXING] Hash changed (cache={_r2_keys_hash_cache[:8] if _r2_keys_hash_cache else 'None'} -> new={r2_keys_hash[:8]}). Checking differences...")
        
        # ========== STEP 3: Costruisci set delle foto già indicizzate ==========
        indexed_set = set()
        if meta_rows:
            for row in meta_rows:
                # Supporta sia r2_key (nuovo) che photo_id (retrocompatibilità)
                r2_key = row.get("r2_key") or row.get("photo_id")
                if r2_key:
                    indexed_set.add(r2_key)
        
        # ========== STEP 4: Calcola differenze ==========
        to_add = r2_originals_set - indexed_set
        to_remove = indexed_set - r2_originals_set
        
        # ========== STEP 5: Se nessun cambiamento, aggiorna cache e skip ==========
        if len(to_add) == 0 and len(to_remove) == 0:
            _r2_keys_hash_cache = r2_keys_hash
            logger.info(f"[INDEXING] No changes detected (r2={len(r2_originals_set)} indexed={len(indexed_set)}). Skipping sync.")
            return
        
        # ========== STEP 6: Cambiamenti rilevati, esegui sync incrementale ==========
        logger.info(f"[INDEXING] Changes detected: to_add={len(to_add)} to_remove={len(to_remove)}. Running incremental sync.")
        await sync_index_with_r2_incremental()
        
        # Aggiorna cache hash dopo sync completato
        _r2_keys_hash_cache = r2_keys_hash
        
    except Exception as e:
        logger.error(f"[INDEXING] Error in maybe_sync_index: {e}", exc_info=True)

# Funzione globale per sync incrementale (spostata da startup per accesso endpoint)
async def sync_index_with_r2_incremental():
    """Sync incrementale: add/remove foto dall'indice basandosi su R2 (source of truth)"""
    global faiss_index, meta_rows, _r2_keys_cache
    
    if not USE_R2 or not r2_client or not R2_ONLY_MODE:
        logger.warning(f"[INDEXING] SYNC skipped - conditions not met: USE_R2={USE_R2}, r2_client={r2_client is not None}, R2_ONLY_MODE={R2_ONLY_MODE}")
        return
    
    if indexing_lock.locked():
        logger.info("[INDEXING] SYNC already running, skipping")
        return
    
    async with indexing_lock:
        try:
            start_time = datetime.now(timezone.utc)
            logger.info("[INDEXING] SYNC start")
            
            # ========== STEP 1: Lista tutte le foto originali in R2 ==========
            paginator = r2_client.get_paginator('list_objects_v2')
            
            # Log prefix usato (R2_PHOTOS_PREFIX, default vuoto)
            logger.info(f"[INDEXING] list prefix={R2_PHOTOS_PREFIX!r}")
            
            # Raccogli tutte le keys prima del filtro
            # Se R2_PHOTOS_PREFIX è vuoto, non passare Prefix (lista tutto il bucket)
            all_keys = []
            paginate_kwargs = {"Bucket": R2_BUCKET}
            if R2_PHOTOS_PREFIX:
                paginate_kwargs["Prefix"] = R2_PHOTOS_PREFIX
            
            for page in paginator.paginate(**paginate_kwargs):
                for obj in page.get('Contents', []):
                    all_keys.append(obj['Key'])
            
            # Log totale keys e sample
            logger.info(f"[INDEXING] list total keys={len(all_keys)} sample={all_keys[:20]}")
            
            # Filtra per originali (solo filename, non path complessi)
            r2_originals_keys = set()
            for key in all_keys:
                # Escludi file di sistema
                if key.startswith("faces.") or key.startswith("downloads_track"):
                    continue
                # Escludi preview (wm/ e thumbs/)
                if key.startswith("wm/") or key.startswith("thumbs/"):
                    continue
                # Include solo immagini originali (estensioni case-insensitive)
                if not any(key.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.heic']):
                    continue
                
                # Estrai solo filename (basename) come r2_key
                from pathlib import Path
                filename = Path(key).name
                r2_originals_keys.add(filename)
            
            # Log keys dopo filtro originali
            r2_originals_list = sorted(list(r2_originals_keys))
            logger.info(f"[INDEXING] after filter originals: {len(r2_originals_list)} keys, sample={r2_originals_list[:20]}")
            
            # ========== STEP 2: Costruisci set delle foto già indicizzate ==========
            indexed_keys_set = set()
            if meta_rows:
                for row in meta_rows:
                    # Supporta sia r2_key (nuovo) che photo_id (retrocompatibilità)
                    r2_key = row.get("r2_key") or row.get("photo_id")
                    if r2_key:
                        indexed_keys_set.add(r2_key)
            
            # ========== STEP 3: Calcola differenze ==========
            to_add = r2_originals_keys - indexed_keys_set
            to_remove = indexed_keys_set - r2_originals_keys
            
            logger.info(f"[INDEXING] r2_originals={len(r2_originals_keys)} indexed={len(indexed_keys_set)} to_add={len(to_add)} to_remove={len(to_remove)}")
            
            # ========== STEP 4: Remove (senza ricalcolare embeddings) ==========
            # Filtra meta_rows mantenendo solo quelle NON in to_remove
            new_meta_rows = []
            old_idx_to_new_idx = {}  # Mappa vecchio indice -> nuovo indice
            new_idx = 0
            
            if faiss_index is not None and faiss_index.ntotal > 0:
                for old_idx, row in enumerate(meta_rows):
                    r2_key = row.get("r2_key") or row.get("photo_id")
                    if r2_key and r2_key not in to_remove:
                        new_meta_rows.append(row)
                        old_idx_to_new_idx[old_idx] = new_idx
                        new_idx += 1
                
                # Ricostruisci FAISS index solo con gli embedding delle righe mantenute
                new_faiss_index = faiss.IndexFlatIP(INDEX_DIM)
                new_embeddings_list = []
                
                for old_idx, row in enumerate(meta_rows):
                    r2_key = row.get("r2_key") or row.get("photo_id")
                    if r2_key and r2_key not in to_remove:
                        try:
                            # Ricostruisci embedding dal vecchio index
                            embedding = faiss_index.reconstruct(int(old_idx))
                            embedding = np.array(embedding, dtype=np.float32)
                            new_embeddings_list.append(embedding)
                        except Exception as e:
                            logger.warning(f"[INDEXING] Could not reconstruct embedding for old_idx={old_idx}, r2_key={r2_key}: {e}")
                
                # Aggiungi tutti gli embedding in batch
                if new_embeddings_list:
                    embeddings_array = np.array(new_embeddings_list, dtype=np.float32)
                    new_faiss_index.add(embeddings_array)
            else:
                new_faiss_index = faiss.IndexFlatIP(INDEX_DIM)
            
            # ========== STEP 5: Add (solo nuove foto) ==========
            faces_total = len(new_meta_rows)  # Conta già le facce mantenute
            new_faces_count = 0
            previews_generated_count = 0
            
            for filename in to_add:
                try:
                    # filename è già il basename (es. "IMG_1016.jpg")
                    # Cerca la foto in R2 (potrebbe essere in qualsiasi posizione nel bucket)
                    # Prima prova con filename diretto
                    r2_key_to_download = filename
                    photo_bytes = None
                    
                    # Se non trovata, cerca in tutte le keys
                    if filename not in all_keys:
                        # Cerca una key che finisce con questo filename
                        for key in all_keys:
                            if key.endswith(f"/{filename}") or key == filename:
                                r2_key_to_download = key
                                break
                    
                    # Scarica foto da R2
                    photo_bytes = await _r2_get_object_bytes(r2_key_to_download)
                    
                    # Decodifica immagine
                    nparr = np.frombuffer(photo_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    
                    # Estrai volti
                    faces = face_app.get(img)
                    
                    if not faces:
                        # Foto senza volti -> salta
                        continue
                    
                    # Per ogni volto, crea embedding e metadata
                    for face in faces:
                        embedding = face.embedding.astype(np.float32)
                        embedding = _normalize(embedding)
                        
                        # Aggiungi embedding al nuovo indice
                        embedding_reshaped = embedding.reshape(1, -1)
                        new_faiss_index.add(embedding_reshaped)
                        
                        # Metadata con r2_key = filename (semplice)
                        bbox = face.bbox
                        new_meta_rows.append({
                            "r2_key": filename,  # Solo filename (es. "IMG_1016.jpg")
                            "display_name": filename,  # Stesso del r2_key
                            "photo_id": filename,  # Compatibilità retroattiva
                            "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                            "det_score": float(face.det_score)
                        })
                        new_faces_count += 1
                        faces_total += 1
                    
                    # Genera preview per OGNI nuova foto (skip intelligente: controlla esistenza prima)
                    thumb_created, wm_created = await ensure_previews_for_photo(filename)
                    if thumb_created:
                        previews_generated_count += 1
                    if wm_created:
                        previews_generated_count += 1
                    
                except Exception as e:
                    logger.error(f"[INDEXING] Error processing new photo {filename}: {e}")
                    continue
            
            # ========== STEP 6: Swap atomico ==========
            faiss_index = new_faiss_index
            meta_rows = new_meta_rows
            
            vectors_in_index = faiss_index.ntotal
            meta_rows_count = len(meta_rows)
            
            logger.info(f"[INDEXING] faces_total={faces_total} vectors_in_index={vectors_in_index} meta_rows={meta_rows_count}")
            
            # ========== STEP 7: Salva su R2 (sovrascrivi sempre) ==========
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                faiss.write_index(faiss_index, tmp_path)
                with open(tmp_path, 'rb') as f:
                    index_bytes = f.read()
                r2_client.put_object(Bucket=R2_BUCKET, Key="faces.index", Body=index_bytes)
                
                # Salva metadata
                meta_lines = [json.dumps(m, ensure_ascii=False) for m in meta_rows]
                meta_bytes = '\n'.join(meta_lines).encode('utf-8')
                r2_client.put_object(Bucket=R2_BUCKET, Key="faces.meta.jsonl", Body=meta_bytes)
                
                import os as os_module
                os_module.unlink(tmp_path)
                
                # Invalida cache
                _r2_keys_cache = None
                
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                logger.info(f"[INDEXING] ✅ SYNC completed in {elapsed:.2f}s")
                logger.info(f"[INDEXING] 📊 Stats: +{len(to_add)} photos added, -{len(to_remove)} removed, {new_faces_count} new faces indexed")
                logger.info(f"[INDEXING] 🎯 Index ready: {vectors_in_index} faces in {meta_rows_count} photos")
                
            except Exception as e:
                logger.error(f"[INDEXING] Error saving synced index to R2: {e}")
                raise
            
        except Exception as e:
            logger.error(f"[INDEXING] Error in SYNC: {e}", exc_info=True)
            raise

@app.post("/admin/index/sync")
async def admin_index_sync(password: Optional[str] = Query(None)):
    """Endpoint admin per sync incrementale dell'indice con R2 (manual override)"""
    if not _check_admin_auth(password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    logger.info("[INDEXING] Manual sync requested")
    
    try:
        await sync_index_with_r2_incremental()
        return {
            "ok": True,
            "message": "Incremental sync completed",
            "index_size": faiss_index.ntotal if faiss_index else 0,
            "meta_rows": len(meta_rows)
        }
    except Exception as e:
        logger.error(f"Error in admin index sync: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")

@app.post("/admin/index/rebuild")
async def admin_index_rebuild(password: Optional[str] = Query(None)):
    """Endpoint admin per FULL REBUILD completo dell'indice (solo uso manuale)"""
    if not _check_admin_auth(password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        count = await index_new_r2_photos()
        return {
            "ok": True,
            "message": "Full rebuild completed",
            "photos_indexed": count,
            "index_size": faiss_index.ntotal if faiss_index else 0,
            "meta_rows": len(meta_rows)
        }
    except Exception as e:
        logger.error(f"Error in admin index rebuild: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {str(e)}")

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
    if R2_ONLY_MODE:
        photos_absolute = "DISABLED (R2_ONLY_MODE)"
        photos_exists = False
        photos_files = ["R2_ONLY_MODE: PHOTOS_DIR disabled"]
    else:
        photos_absolute = str(PHOTOS_DIR.resolve())
        photos_exists = PHOTOS_DIR.exists()
        photos_files = []
        if PHOTOS_DIR.exists():
            try:
                photos_files = [p.name for p in PHOTOS_DIR.iterdir() if p.is_file()][:20]
            except Exception as e:
                photos_files = [f"Error listing: {str(e)}"]
    
    static_files = []
    if STATIC_DIR.exists():
        try:
            static_files = [p.name for p in STATIC_DIR.iterdir() if p.is_file()]
        except Exception as e:
            static_files = [f"Error listing: {str(e)}"]
    
    index_exists = (STATIC_DIR / "index.html").exists()
    index_path = str((STATIC_DIR / "index.html").resolve()) if index_exists else "NOT FOUND"
    
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
        "static_files": static_files,
        "index_html_exists": index_exists,
        "index_html_path": index_path,
    }

@app.get("/debug/test-postgres")
async def debug_test_postgres():
    """Endpoint di test per verificare che PostgreSQL 18 funzioni correttamente"""
    if not USE_POSTGRES:
        return {
            "ok": False,
            "error": "PostgreSQL not in use",
            "database_type": "None"
        }
    
    results = {
        "ok": True,
        "database_type": "PostgreSQL",
        "tests": {}
    }
    
    try:
        # Test 1: Connessione base
        try:
            conn = await asyncpg.connect(DATABASE_URL)
            await conn.close()
            results["tests"]["connection"] = {"ok": True, "message": "Connection successful"}
        except Exception as e:
            results["tests"]["connection"] = {"ok": False, "error": str(e)}
            results["ok"] = False
        
        # Test 2: Query con NOW()
        try:
            rows = await _db_execute("SELECT NOW() as current_time", ())
            if rows and 'current_time' in rows[0]:
                results["tests"]["now_function"] = {"ok": True, "value": str(rows[0]['current_time'])}
            else:
                results["tests"]["now_function"] = {"ok": False, "error": "No result returned"}
                results["ok"] = False
        except Exception as e:
            results["tests"]["now_function"] = {"ok": False, "error": str(e)}
            results["ok"] = False
        
        # Test 3: Query con INTERVAL
        try:
            rows = await _db_execute("SELECT NOW() + INTERVAL '30 days' as future_date", ())
            if rows and 'future_date' in rows[0]:
                results["tests"]["interval_function"] = {"ok": True, "value": str(rows[0]['future_date'])}
            else:
                results["tests"]["interval_function"] = {"ok": False, "error": "No result returned"}
                results["ok"] = False
        except Exception as e:
            results["tests"]["interval_function"] = {"ok": False, "error": str(e)}
            results["ok"] = False
        
        # Test 4: Query con placeholder $1, $2
        try:
            test_email = "test@example.com"
            rows = await _db_execute("SELECT $1 as test_param, $2 as test_param2", (test_email, "test2"))
            if rows and rows[0]['test_param'] == test_email:
                results["tests"]["placeholder_query"] = {"ok": True, "message": "Placeholders work correctly"}
            else:
                results["tests"]["placeholder_query"] = {"ok": False, "error": "Placeholder values don't match"}
                results["ok"] = False
        except Exception as e:
            results["tests"]["placeholder_query"] = {"ok": False, "error": str(e)}
            results["ok"] = False
        
        # Test 5: INSERT e SELECT (test completo)
        try:
            test_photo_id = f"test_photo_{int(datetime.now(timezone.utc).timestamp())}"
            test_email = "test@example.com"
            
            # Inserisci record di test
            await _db_execute_write(f"""
                INSERT INTO user_photos (email, photo_id, found_at, status, expires_at)
                VALUES ($1, $2, NOW(), 'found', NOW() + INTERVAL '90 days')
                ON CONFLICT (email, photo_id) DO NOTHING
            """, (test_email, test_photo_id))
            
            # Leggi il record inserito
            rows = await _db_execute("""
                SELECT email, photo_id, status, expires_at FROM user_photos 
                WHERE email = $1 AND photo_id = $2
            """, (test_email, test_photo_id))
            
            if rows and len(rows) > 0:
                # Pulisci: elimina il record di test
                await _db_execute_write("""
                    DELETE FROM user_photos WHERE email = $1 AND photo_id = $2
                """, (test_email, test_photo_id))
                
                results["tests"]["insert_select"] = {
                    "ok": True, 
                    "message": "INSERT and SELECT work correctly",
                    "inserted_record": {
                        "email": rows[0]['email'],
                        "photo_id": rows[0]['photo_id'],
                        "status": rows[0]['status']
                    }
                }
            else:
                results["tests"]["insert_select"] = {"ok": False, "error": "Record not found after insert"}
                results["ok"] = False
        except Exception as e:
            results["tests"]["insert_select"] = {"ok": False, "error": str(e)}
            results["ok"] = False
        
        # Test 6: Verifica versione PostgreSQL
        try:
            rows = await _db_execute("SELECT version() as pg_version", ())
            if rows and 'pg_version' in rows[0]:
                pg_version = rows[0]['pg_version']
                results["tests"]["postgres_version"] = {"ok": True, "version": pg_version}
            else:
                results["tests"]["postgres_version"] = {"ok": False, "error": "Version not returned"}
        except Exception as e:
            results["tests"]["postgres_version"] = {"ok": False, "error": str(e)}
        
    except Exception as e:
        results["ok"] = False
        results["error"] = str(e)
    
    return results

@app.get("/debug/orders")
async def debug_orders(email: Optional[str] = Query(None, description="Filtra per email")):
    """Endpoint di debug per vedere ordini e foto nel database"""
    try:
        if email:
            email = _normalize_email(email)
        
        # Recupera tutti gli ordini
        if email:
            orders_rows = await _db_execute(
                "SELECT order_id, email, stripe_session_id, photo_ids, amount_cents, paid_at, download_token FROM orders WHERE email = $1 ORDER BY paid_at DESC LIMIT 50",
                (email,)
            )
        else:
            orders_rows = await _db_execute(
                "SELECT order_id, email, stripe_session_id, photo_ids, amount_cents, paid_at, download_token FROM orders ORDER BY paid_at DESC LIMIT 50"
            )
        
        orders = []
        for row in orders_rows:
            orders.append({
                "order_id": row['order_id'],
                "email": row['email'],
                "stripe_session_id": row['stripe_session_id'],
                "photo_ids": json.loads(row['photo_ids']) if row['photo_ids'] else [],
                "amount_cents": row['amount_cents'],
                "paid_at": str(row['paid_at']),
                "download_token": row['download_token'][:20] + "..." if row['download_token'] else None
            })
        
        # Recupera tutte le foto utente
        if email:
            photos_rows = await _db_execute(
                "SELECT email, photo_id, status, found_at, paid_at, expires_at FROM user_photos WHERE email = $1 ORDER BY paid_at DESC, found_at DESC LIMIT 100",
                (email,)
            )
        else:
            photos_rows = await _db_execute(
                "SELECT email, photo_id, status, found_at, paid_at, expires_at FROM user_photos ORDER BY paid_at DESC, found_at DESC LIMIT 100"
            )
        
        user_photos = []
        for row in photos_rows:
            user_photos.append({
                "email": row['email'],
                "photo_id": row['photo_id'],
                "status": row['status'],
                "found_at": str(row['found_at']),
                "paid_at": str(row['paid_at']) if row['paid_at'] else None,
                "expires_at": str(row['expires_at']) if row['expires_at'] else None
            })
        
        return {
            "ok": True,
            "email_filter": email,
            "orders_count": len(orders),
            "orders": orders,
            "user_photos_count": len(user_photos),
            "user_photos": user_photos
        }
    except Exception as e:
        logger.error(f"Error in debug_orders: {e}", exc_info=True)
        return {"error": str(e), "orders": [], "user_photos": []}

@app.post("/debug/fix-paid-photos")
async def fix_paid_photos(email: str = Query(..., description="Email utente")):
    """Endpoint per correggere manualmente le foto pagate basandosi sugli ordini"""
    try:
        email = _normalize_email(email)
        logger.info(f"Fixing paid photos for {email}")
        
        # Recupera tutti gli ordini per questa email
        orders_rows = await _db_execute(
            "SELECT order_id, photo_ids FROM orders WHERE email = $1",
            (email,)
        )
        
        if not orders_rows:
            return {"ok": False, "message": f"No orders found for {email}"}
        
        # Raccogli tutti i photo_ids dagli ordini
        all_paid_photo_ids = set()
        for row in orders_rows:
            photo_ids = json.loads(row['photo_ids']) if row['photo_ids'] else []
            all_paid_photo_ids.update(photo_ids)
        
        logger.info(f"Found {len(orders_rows)} orders with {len(all_paid_photo_ids)} unique paid photos")
        
        # Marca tutte queste foto come pagate
        fixed_count = 0
        for photo_id in all_paid_photo_ids:
            success = await _mark_photo_paid(email, photo_id)
            if success:
                fixed_count += 1
                logger.info(f"Fixed photo: {photo_id}")
            else:
                logger.warning(f"Failed to fix photo: {photo_id}")
        
        return {
            "ok": True,
            "message": f"Fixed {fixed_count} photos for {email}",
            "orders_count": len(orders_rows),
            "photos_fixed": fixed_count,
            "photo_ids": list(all_paid_photo_ids)
        }
    except Exception as e:
        logger.error(f"Error fixing paid photos: {e}", exc_info=True)
        return {"error": str(e)}

@app.get("/thumb/{photo_id:path}")
async def serve_thumb(photo_id: str):
    """Endpoint per servire thumbnail: redirect a R2 pubblico (deprecato, usa /photo?variant=thumb)"""
    # Redirect a /photo con variant=thumb
    from urllib.parse import quote
    photo_id_encoded = quote(photo_id, safe='')
    return RedirectResponse(url=f"/photo/{photo_id_encoded}?variant=thumb", status_code=302)

@app.get("/wm/{photo_id:path}")
async def serve_wm(photo_id: str):
    """Endpoint per servire preview watermarked: redirect a R2 pubblico (deprecato, usa /photo?variant=wm)"""
    # Redirect a /photo con variant=wm
    from urllib.parse import quote
    photo_id_encoded = quote(photo_id, safe='')
    return RedirectResponse(url=f"/photo/{photo_id_encoded}?variant=wm", status_code=302)

@app.get("/photo/{filename:path}")
async def serve_photo(
    filename: str, 
    request: Request,
    variant: Optional[str] = Query(None, description="variant=thumb per thumbnail, variant=wm per watermarked preview"),
    paid: bool = Query(False, description="Se true, serve foto senza watermark (solo se pagata)"),
    token: Optional[str] = Query(None, description="Download token per verificare pagamento"),
    email: Optional[str] = Query(None, description="Email utente per verificare pagamento"),
    download: bool = Query(False, description="Se true, forza il download con header Content-Disposition")
):
    """Endpoint per servire le foto: variant=thumb per thumbnail, variant=wm per watermarked, paid=true per originale"""
    logger.info(f"=== PHOTO REQUEST ===")
    logger.info(f"Request path: {request.url.path}")
    logger.info(f"Filename parameter: {filename}, variant: {variant}")
    logger.info(f"Paid: {paid}, Token: {token is not None}, Email: {email is not None}")
    
    # Decodifica il filename (potrebbe essere URL encoded)
    try:
        from urllib.parse import unquote
        decoded_filename = unquote(filename)
        logger.info(f"Decoded filename: {decoded_filename}")
    except Exception as e:
        logger.warning(f"Error decoding filename: {e}")
        decoded_filename = filename
    
    # Rimuovi prefissi wm/ e thumbs/ se presenti (per retrocompatibilità)
    original_key = decoded_filename.strip().lstrip("/")
    while True:
        if original_key.startswith("wm/"):
            original_key = original_key[3:]
            continue
        if original_key.startswith("thumbs/"):
            original_key = original_key[7:]
            continue
        # Rimuovi anche "photos/" se presente (retrocompatibilità)
        if original_key.startswith("photos/"):
            from pathlib import Path
            original_key = Path(original_key).name
            continue
        break
    
    # Validazione: original_key deve essere non vuoto
    if not original_key:
        logger.error(f"Invalid photo key: original={decoded_filename}, original_key={original_key}")
        raise HTTPException(status_code=400, detail="Invalid photo key")
    
    logger.info(f"[PHOTO] original_key={original_key} variant={variant}")
    
    # Verifica R2 configurato
    if not USE_R2 or r2_client is None:
        logger.error("R2 storage not configured. Cannot serve photos.")
        raise HTTPException(status_code=503, detail="Photo storage not available")

    # Determina se l'utente vuole l'originale (paid=true) oppure una preview (thumb/wm).
    # Nota: paid=true NON basta da solo; facciamo sempre verifica server-side via token/email.
    wants_original = bool(paid)
    filename_check = original_key
    
    # Verifica se la foto è pagata usando token o email
    is_paid = False
    
    if paid:
        logger.info("Ignoring client paid flag; server-side verification only")

    # Verifica con token (priorità)
    if token:
        order = await _get_order_by_token(token)
        if order and filename_check in order.get('photo_ids', []):
            expires_at = order.get('expires_at')
            if expires_at:
                try:
                    expires_date = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                    now = datetime.now(timezone.utc)
                    if now > expires_date:
                        logger.info(f"Token expired, forcing watermark: {token[:8]}... expires_at={expires_at}, original_key={original_key}")
                        is_paid = False
                    else:
                        is_paid = True
                        logger.info(f"Photo verified as paid via token: original_key={original_key}")
                except Exception as e:
                    logger.error(f"Error parsing expires_at for token validation: {e}")
                    is_paid = False
            else:
                is_paid = True
                logger.info(f"Photo verified as paid via token (no expiry): original_key={original_key}")

    # Fallback: verifica per email (solo se non già pagata via token)
    if (not is_paid) and email:
        try:
            paid_photos = await _get_user_paid_photos(email)
            if filename_check in paid_photos:
                is_paid = True
                logger.info(f"Photo verified as paid via email: original_key={original_key}")
        except Exception as e:
            logger.error(f"Error checking paid photos: {e}")

    # Costruisci object_key in modo deterministico:
    # - Se (paid=true e verificato) -> originals/<filename>
    # - Altrimenti -> variant default "thumb": thumbs/<filename> oppure wm/<filename>
    if wants_original and is_paid:
        object_key = f"originals/{original_key}"
    else:
        variant_effective = (variant or "thumb").lower()
        if variant_effective == "thumb":
            object_key = f"thumbs/{original_key}"
        elif variant_effective == "wm":
            object_key = f"wm/{original_key}"
        else:
            object_key = f"wm/{original_key}"
    
    # Render: deve essere SEMPRE settato per avere foto istantanee.
    is_render = (os.getenv("RENDER", "").lower() == "true") or bool(os.getenv("RENDER_SERVICE_ID"))

    # Se R2_PUBLIC_BASE_URL è configurato e NON serve download forzato, usa redirect (produzione).
    # Se serve download forzato, serviamo direttamente per controllare Content-Disposition.
    if R2_PUBLIC_BASE_URL and not download:
        public_url = _get_r2_public_url(object_key)

        # Track download se pagato (solo per originals/*)
        if is_paid and wants_original:
            from pathlib import Path
            _track_download(Path(original_key).name)

        headers = {"Cache-Control": "public, max-age=31536000, immutable"}
        logger.info(f"[PHOTO] Redirecting to R2 public URL: {object_key}")
        return RedirectResponse(url=public_url, status_code=302, headers=headers)

    # Se serve download forzato OPPURE manca R2_PUBLIC_BASE_URL, serviamo direttamente.
    # Questo ci permette di controllare Content-Disposition e Content-Type correttamente.
    if is_render and not R2_PUBLIC_BASE_URL:
        logger.error("[PHOTO] R2_PUBLIC_BASE_URL missing on Render (must be configured)")
        raise HTTPException(status_code=500, detail="R2_PUBLIC_BASE_URL not configured (required on Render)")

    try:
        photo_bytes = await _r2_get_object_bytes(object_key)

        if is_paid and wants_original:
            from pathlib import Path
            _track_download(Path(original_key).name)

        # Headers cross-platform corretti
        from pathlib import Path
        filename = Path(original_key).name
        
        headers = {
            "Cache-Control": "public, max-age=31536000, immutable",
            "Content-Type": "image/jpeg"  # Sempre esplicito per compatibilità cross-platform
        }
        
        # Content-Disposition: inline per visualizzazione, attachment per download
        if download:
            headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        else:
            headers["Content-Disposition"] = f'inline; filename="{filename}"'

        logger.info(f"[PHOTO] Streaming bytes: {object_key}, download={download}")
        return Response(content=photo_bytes, media_type="image/jpeg", headers=headers)
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code in ('404', 'NoSuchKey'):
            logger.warning(f"[PHOTO] Missing R2 object: {object_key}")
            # 404 pulito: restituisce solo status code senza body HTML/JSON
            # Questo evita che i browser tentino di renderizzare HTML/JSON come immagine
            return Response(status_code=404, content=b"", media_type="image/jpeg")
        logger.error(f"[PHOTO] R2 error for {object_key}: {error_code or type(e).__name__}")
        raise HTTPException(status_code=503, detail="R2 storage error")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PHOTO] Unexpected error serving photo: {type(e).__name__}: {e}, key={object_key}")
        raise HTTPException(status_code=500, detail=f"Error serving photo: {str(e)}")

# ========== ENDPOINT UTENTI ==========

@app.post("/register_user")
async def register_user(
    email: str = Query(..., description="Email utente"),
    selfie: UploadFile = File(..., description="Selfie per riconoscimento")
):
    """Registra un nuovo utente o aggiorna selfie esistente"""
    try:
        # Leggi e processa selfie
        file_bytes = await selfie.read()
        img = _read_image_from_bytes(file_bytes)
        
        # Rileva faccia e estrai embedding
        assert face_app is not None
        faces = face_app.get(img)
        
        if not faces:
            raise HTTPException(status_code=400, detail="No face detected in selfie")
        
        # Prendi il volto più grande
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )
        
        embedding = faces_sorted[0].embedding.astype("float32")
        embedding_bytes = embedding.tobytes()
        
        # Salva/aggiorna utente
        success = await _create_or_update_user(email, embedding_bytes)
        
        if not success:
            raise HTTPException(status_code=500, detail="Error saving user")
        
        return {
            "ok": True,
            "email": email,
            "message": "User registered successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/check_date")
async def check_date(
    tour_date: str = Query(..., description="Data del tour (YYYY-MM-DD)")
):
    """Verifica se ci sono foto disponibili per una data specifica"""
    _ensure_ready()
    
    try:
        # Normalizza formato data
        normalized_date = tour_date.replace("-", "") if "-" not in tour_date else tour_date
        if len(normalized_date) == 8:  # YYYYMMDD
            normalized_date = f"{normalized_date[:4]}-{normalized_date[4:6]}-{normalized_date[6:8]}"
        
        # Conta foto con facce per questa data
        photos_with_faces = 0
        for row in meta_rows:
            photo_tour_date = row.get("tour_date")
            if photo_tour_date and normalized_date in str(photo_tour_date):
                photos_with_faces += 1
        
        total_photos = photos_with_faces
        
        if total_photos == 0:
            # Verifica se ci sono foto in elaborazione (foto caricate di recente senza tour_date)
            # Per ora, assumiamo che se non ci sono foto, potrebbero essere in elaborazione
            # In futuro, possiamo aggiungere un flag "processing" nel metadata
            return {
                "ok": False,
                "status": "no_photos",
                "message": "Nessuna foto disponibile per questa data"
            }
        
        return {
            "ok": True,
            "status": "available",
            "photos_count": total_photos,
            "photos_with_faces": photos_with_faces
        }
    except Exception as e:
        logger.error(f"Error checking date: {e}")
        return {
            "ok": False,
            "status": "error",
            "message": f"Errore durante la verifica: {str(e)}"
        }

@app.post("/check_user")
async def check_user(
    email: str = Query(..., description="Email utente"),
    selfie: UploadFile = File(..., description="Selfie per verifica")
):
    """Verifica se utente esiste e matcha selfie, ritorna storico"""
    try:
        email = _normalize_email(email)
        # Leggi e processa selfie
        file_bytes = await selfie.read()
        img = _read_image_from_bytes(file_bytes)
        
        # Rileva faccia e estrai embedding
        assert face_app is not None
        faces = face_app.get(img)
        
        if not faces:
            raise HTTPException(status_code=400, detail="No face detected in selfie")
        
        # Prendi il volto più grande
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )
        
        embedding = faces_sorted[0].embedding.astype("float32")
        embedding_bytes = embedding.tobytes()
        
        # Verifica utente
        user = await _get_user_by_email(email)
        
        if not user:
            return {
                "ok": True,
                "exists": False,
                "message": "User not found"
            }
        
        # Verifica matching selfie
        if user.get('selfie_embedding'):
            saved_embedding = np.frombuffer(user['selfie_embedding'], dtype=np.float32)
            current_embedding = embedding
            saved_embedding = _normalize(saved_embedding)
            current_embedding = _normalize(current_embedding)
            
            similarity = np.dot(saved_embedding, current_embedding)
            
            if similarity < 0.7:
                return {
                    "ok": True,
                    "exists": True,
                    "match": False,
                    "similarity": float(similarity),
                    "message": "Email found but selfie doesn't match"
                }
        
        # Aggiorna last_login
        await _create_or_update_user(email, embedding_bytes)
        
        # Recupera foto trovate e pagate
        found_photos = await _get_user_found_photos(email)
        paid_photos = await _get_user_paid_photos(email)
        
        return {
            "ok": True,
            "exists": True,
            "match": True,
            "found_photos": found_photos,
            "paid_photos": paid_photos,
            "message": "User found and verified"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking user: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/user/register")
async def register_user_by_email(
    email: str = Query(..., description="Email utente")
):
    """Registra un utente (NO-OP in stateless mode)"""
    try:
        # Normalizza email
        email = _normalize_email(email)
        
        # Valida email
        import re
        if not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email):
            raise HTTPException(status_code=400, detail="Invalid email format")
        
        # Stateless mode: non salva nulla, solo valida
        logger.info(f"User register request (stateless): {email}")
        
        return {
            "ok": True,
            "email": email,
            "message": "User registered successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/user/photos")
async def get_user_photos(
    email: str = Query(..., description="Email utente")
):
    """Recupera tutte le foto di un utente (stateless: sempre vuoto)"""
    try:
        email = _normalize_email(email)
        logger.info(f"User photos request (stateless): {email}")
        
        # Stateless mode: sempre vuoto
        found_photos = []
        paid_photos = []
        
        # Response con Cache-Control: no-store
        from fastapi.responses import JSONResponse
        response = JSONResponse({
            "ok": True,
            "email": email,
            "found_photos": found_photos,
            "paid_photos": paid_photos
        })
        response.headers["Cache-Control"] = "no-store"
        return response
    except Exception as e:
        logger.error(f"Error getting user photos: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/my-photos")
async def my_photos_by_email(
    request: Request,
    email: Optional[str] = Query(None, description="Email utente")
):
    """Pagina download foto pagate usando solo email (per utenti che rientrano)"""
    try:
        if email:
            email = _normalize_email(email)
        if not email:
            return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Email richiesta - TenerifePictures</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body { font-family: Arial, sans-serif; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; background: linear-gradient(135deg, #7b74ff, #5f58ff); color: #fff; }
                    .container { text-align: center; padding: 40px; background: rgba(255,255,255,0.1); border-radius: 20px; }
                    h1 { font-size: 32px; margin: 0 0 20px; }
                    a { display: inline-block; margin-top: 20px; padding: 12px 24px; background: #fff; color: #5f58ff; text-decoration: none; border-radius: 8px; font-weight: 600; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📧 Email richiesta</h1>
                    <p>Per accedere alle tue foto, inserisci la tua email.</p>
                    <a href="/">Back to home</a>
                </div>
            </body>
            </html>
            """)
        
        # Recupera foto pagate per questa email
        paid_photos = []
        try:
            paid_photos = await _get_user_paid_photos(email)
        except Exception as e:
            logger.error(f"Error getting paid photos for {email}: {e}")
        
        if not paid_photos or len(paid_photos) == 0:
            return HTMLResponse(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Nessuna foto acquistata - TenerifePictures</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body {{ font-family: Arial, sans-serif; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; background: linear-gradient(135deg, #7b74ff, #5f58ff); color: #fff; }}
                    .container {{ text-align: center; padding: 40px; background: rgba(255,255,255,0.1); border-radius: 20px; }}
                    h1 {{ font-size: 32px; margin: 0 0 20px; }}
                    a {{ display: inline-block; margin-top: 20px; padding: 12px 24px; background: #fff; color: #5f58ff; text-decoration: none; border-radius: 8px; font-weight: 600; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📷 Nessuna foto acquistata</h1>
                    <p>Non hai ancora acquistato foto con questa email.</p>
                    <a href="/">Vai alla galleria</a>
                </div>
            </body>
            </html>
            """)
        
        # Genera HTML per pagina download (stesso stile di checkout/success)
        base_url = str(request.base_url).rstrip('/')
        photos_html = ""
        for photo_id in paid_photos:
            photo_url = f"{base_url}/photo/{photo_id}?email={email}"
            photo_id_escaped = photo_id.replace("'", "\\'").replace('"', '&quot;')
            email_escaped = email.replace("'", "\\'").replace('"', '&quot;')
            photos_html += f"""
    <div class="photo-item">
        <img
            src="{photo_url}"
            alt="Photo"
            loading="lazy"
            class="photo-img"
            data-photo-id="{photo_id_escaped}"
            data-photo-url="{photo_url}"
        >
        <button class="download-btn download-btn-desktop" onclick="downloadPhotoSuccess('{photo_id_escaped}', '{email_escaped}', this)">📥 Download</button>
    </div>
    """
        
        # Link intelligente: se ha email, porta direttamente all'album (con parametro view_album per forzare visualizzazione anche se ha foto pagate)
        if email:
            album_button_top = f'<a href="/?email={email}&view_album=true" class="main-button" style="margin-top: 0; margin-bottom: 30px;">📸 Back to album</a>'
            album_button_bottom = f'<a href="/?email={email}&view_album=true" class="main-button" style="margin-top: 30px; margin-bottom: 0;">📸 Back to album</a>'
        else:
            album_button_top = '<a href="/" class="main-button" style="margin-top: 0; margin-bottom: 30px;">📸 Back to album</a>'
            album_button_bottom = '<a href="/" class="main-button" style="margin-top: 30px; margin-bottom: 0;">📸 Back to album</a>'
        
        # Usa lo stesso HTML template di checkout/success
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Le mie foto - TenerifePictures</title>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 20px;
                    padding: 40px;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                }}
                .success-icon {{
                    font-size: 80px;
                    text-align: center;
                    margin-bottom: 20px;
                }}
                h1 {{
                    text-align: center;
                    color: #333;
                    margin-bottom: 20px;
                    font-size: 36px;
                }}
                .message {{
                    text-align: center;
                    color: #666;
                    font-size: 18px;
                    margin-bottom: 30px;
                }}
                .photos-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }}
                .photo-item {{
                    position: relative;
                    border-radius: 12px;
                    overflow: hidden;
                    background: #f5f5f5;
                }}
                .photo-img {{
                    width: 100%;
                    height: auto;
                    display: block;
                    cursor: pointer;
                    -webkit-touch-callout: default;
                    -webkit-user-select: none;
                    user-select: none;
                }}
                .photo-item {{ cursor: pointer; }}
                .photo-item:active {{ transform: scale(0.99); }}
                .download-btn {{
                    position: absolute;
                    bottom: 10px;
                    right: 10px;
                    background: #22c55e;
                    color: white;
                    border: none;
                    padding: 10px 16px;
                    border-radius: 10px;
                    cursor: pointer;
                    font-weight: 700;
                    font-size: 14px;
                    box-shadow: 0 8px 20px rgba(0,0,0,0.18);
                }}
                .download-btn:hover {{ background: #16a34a; }}
                .download-btn:disabled {{ opacity: 0.6; cursor: not-allowed; }}
                .main-button {{
                    display: block;
                    width: 100%;
                    max-width: 400px;
                    margin: 30px auto;
                    padding: 18px 30px;
                    background: linear-gradient(135deg, #7b74ff 0%, #5f58ff 100%);
                    color: #fff;
                    text-decoration: none;
                    border-radius: 12px;
                    font-weight: 700;
                    font-size: 18px;
                    border: none;
                    box-shadow: 0 4px 15px rgba(123, 116, 255, 0.4);
                    transition: transform 0.2s, box-shadow 0.2s;
                    text-align: center;
                    cursor: pointer;
                }}
                .main-button:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 6px 20px rgba(123, 116, 255, 0.6);
                }}
                .main-button:active {{
                    transform: translateY(0);
                }}
                @media (max-width: 600px) {{
                    .photos-grid {{
                        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                        gap: 15px;
                    }}
                    h1 {{
                        font-size: 28px;
                    }}
                    .message {{
                        font-size: 18px;
                    }}
                }}
                /* Fullscreen viewer (tap -> full, long-press -> Save to Photos on iOS) */
                .viewer {{
                    position: fixed;
                    inset: 0;
                    background: rgba(0,0,0,0.92);
                    display: none;
                    align-items: center;
                    justify-content: center;
                    padding: 16px;
                    z-index: 9999;
                }}
                .viewer.open {{ display: flex; }}
                .viewer-inner {{
                    width: 100%;
                    max-width: 900px;
                    display: flex;
                    flex-direction: column;
                    gap: 12px;
                }}
                .viewer-topbar {{
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    gap: 12px;
                    color: #fff;
                }}
                .viewer-badge {{
                    font-size: 14px;
                    font-weight: 700;
                    opacity: 0.95;
                }}
                .viewer-close {{
                    border: none;
                    background: rgba(255,255,255,0.14);
                    color: #fff;
                    padding: 10px 14px;
                    border-radius: 12px;
                    font-weight: 800;
                    cursor: pointer;
                }}
                .viewer-img-wrap {{
                    width: 100%;
                    border-radius: 14px;
                    overflow: hidden;
                    background: #111;
                    box-shadow: 0 18px 60px rgba(0,0,0,0.45);
                }}
                .viewer-img {{
                    width: 100%;
                    height: auto;
                    display: block;
                    -webkit-touch-callout: default;
                    -webkit-user-select: none;
                    user-select: none;
                }}
                .viewer-instructions {{
                    color: rgba(255,255,255,0.92);
                    background: rgba(255,255,255,0.10);
                    border: 1px solid rgba(255,255,255,0.18);
                    border-radius: 14px;
                    padding: 14px;
                    font-size: 15px;
                    line-height: 1.5;
                }}
                .viewer-instructions strong {{ color: #fff; }}
                /* Sticky CTA to sell more */
                .sticky-cta {{
                    position: fixed;
                    left: 16px;
                    right: 16px;
                    bottom: 16px;
                    z-index: 9000;
                    display: none;
                }}
                .sticky-cta-inner {{
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    gap: 12px;
                    background: rgba(255,255,255,0.92);
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(0,0,0,0.08);
                    border-radius: 16px;
                    padding: 12px 14px;
                    box-shadow: 0 14px 40px rgba(0,0,0,0.22);
                }}
                .sticky-cta-text {{
                    color: #111;
                    font-weight: 800;
                    font-size: 14px;
                    line-height: 1.2;
                }}
                .sticky-cta-sub {{
                    display: block;
                    color: rgba(0,0,0,0.65);
                    font-weight: 600;
                    font-size: 12px;
                    margin-top: 4px;
                }}
                .sticky-cta-btn {{
                    flex: 0 0 auto;
                    padding: 12px 14px;
                    border-radius: 14px;
                    border: none;
                    cursor: pointer;
                    font-weight: 900;
                    background: linear-gradient(135deg, #7b74ff 0%, #5f58ff 100%);
                    color: #fff;
                    box-shadow: 0 10px 24px rgba(123,116,255,0.35);
                    text-decoration: none;
                    white-space: nowrap;
                }}
                @media (max-width: 900px) {{
                    .sticky-cta {{ display: block; }}
                    body {{ padding-bottom: 90px; }}
                }}
                /* iOS detection: hide desktop download button */
                @supports (-webkit-touch-callout: none) {{
                    .download-btn-desktop {{ display: none !important; }}
                    #ios-instructions-top {{ display: block !important; }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="success-icon">✅</div>
                <h1>YOUR PHOTOS</h1>
                <p class="message">You purchased {len(paid_photos)} photos</p>
                <!-- iOS Instructions at top (if iPhone) -->
                <div id="ios-instructions-top" style="display: none; margin: 20px 0; padding: 20px; background: rgba(255, 255, 255, 0.2); border: 2px solid rgba(255, 255, 255, 0.4); border-radius: 12px; backdrop-filter: blur(10px); box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
                    <p style="margin: 0 0 12px 0; font-weight: bold; font-size: 20px; text-align: center;">📱 How to save your photos:</p>
                    <p style="margin: 0; font-size: 17px; line-height: 1.8; text-align: left;">1. Touch and hold on any photo below</p>
                    <p style="margin: 8px 0 0 0; font-size: 17px; line-height: 1.8; text-align: left;">2. Select "Save to Photos" from the menu</p>
                    <p style="margin: 12px 0 0 0; font-size: 15px; line-height: 1.6; text-align: center; opacity: 0.9; font-style: italic;">Repeat for each photo you want to save</p>
                </div>
                <p class="message" style="margin-top: 10px; margin-bottom: 10px; font-size: 16px;">Tap a photo to open it full screen. On iPhone, long-press the full photo and choose <strong>Save to Photos</strong>.</p>
                <!-- Buy more photos button at top -->
                {album_button_top}
                <!-- Foto -->
                <div class="photos-grid">
                    {photos_html}
                </div>
                <!-- Buy more photos button at bottom -->
                {album_button_bottom}
                <!-- Fullscreen viewer (tap photo -> fullscreen) -->
                <div id="viewer" class="viewer" aria-hidden="true">
                    <div class="viewer-inner">
                        <div class="viewer-topbar">
                            <div class="viewer-badge">📌 iPhone: long-press the photo → Save to Photos</div>
                            <button class="viewer-close" id="viewerClose" type="button">✕ Close</button>
                        </div>
                        <div class="viewer-img-wrap">
                            <img id="viewerImg" class="viewer-img" src="" alt="Photo">
                        </div>
                        <div class="viewer-instructions" id="viewerInstructions">
                            <strong>Quick save:</strong> long-press the photo above and tap <strong>Save to Photos</strong>.
                            <br>
                            <span style="opacity:0.9;">Tap outside the photo or press ESC to close.</span>
                        </div>
                    </div>
                </div>
                <!-- Sticky CTA (mobile) -->
                <div class="sticky-cta" id="stickyCta">
                    <div class="sticky-cta-inner">
                        <div class="sticky-cta-text">
                            Want more photos?
                            <span class="sticky-cta-sub">Go back to the album and buy the rest in 1 minute</span>
                        </div>
                        <a class="sticky-cta-btn" href="/?email={email}&view_album=true">📸 Back to album</a>
                    </div>
                </div>
            </div>
            <script>
                // Rileva se è iOS
                function isIOS() {{
                    return /iPad|iPhone|iPod/.test(navigator.userAgent) ||
                           (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
                }}
                // Rileva se è Android
                function isAndroid() {{
                    return /Android/i.test(navigator.userAgent);
                }}
                // Mostra/nascondi pulsante e istruzioni in base al dispositivo
                function setupIOSInstructions() {{
                    const iosInstructionsTop = document.getElementById('ios-instructions-top');
                    const downloadBtns = document.querySelectorAll('.download-btn-desktop');
                    if (isIOS()) {{
                        if (iosInstructionsTop) iosInstructionsTop.style.display = 'block';
                        downloadBtns.forEach(el => el.style.display = 'none');
                    }} else {{
                        if (iosInstructionsTop) iosInstructionsTop.style.display = 'none';
                        downloadBtns.forEach(el => el.style.display = 'block');
                    }}
                }}
                if (document.readyState === 'loading') {{
                    document.addEventListener('DOMContentLoaded', setupIOSInstructions);
                }} else {{
                    setupIOSInstructions();
                }}
                setTimeout(setupIOSInstructions, 100);
                setTimeout(setupIOSInstructions, 500);
                // Fullscreen viewer: tap photo -> open fullscreen, long-press on iOS -> Save to Photos
                const viewer = document.getElementById('viewer');
                const viewerImg = document.getElementById('viewerImg');
                const viewerClose = document.getElementById('viewerClose');
                function openViewer(url) {{
                    if (!viewer || !viewerImg) return;
                    viewerImg.src = url;
                    viewer.classList.add('open');
                    viewer.setAttribute('aria-hidden', 'false');
                }}
                function closeViewer() {{
                    if (!viewer) return;
                    viewer.classList.remove('open');
                    viewer.setAttribute('aria-hidden', 'true');
                    if (viewerImg) viewerImg.src = '';
                }}
                // Click any photo to open viewer
                document.querySelectorAll('.photo-img').forEach(img => {{
                    img.addEventListener('click', () => {{
                        const url = img.getAttribute('data-photo-url') || img.src;
                        openViewer(url);
                    }});
                }});
                // Close: ESC
                document.addEventListener('keydown', (e) => {{
                    if (e.key === 'Escape') closeViewer();
                }});
                // Close: button
                if (viewerClose) viewerClose.addEventListener('click', closeViewer);
                // Close: tap outside image (background only)
                if (viewer) {{
                    viewer.addEventListener('click', (e) => {{
                        if (e.target === viewer) closeViewer();
                    }});
                }}
                async function downloadPhotoSuccess(photoId, email, btnElement) {{
                    try {{
                        const btn = btnElement || event?.target || document.querySelector(`button[onclick*="${{photoId}}"]`);
                        if (!btn) {{
                            alert('Errore: pulsante non trovato');
                            return;
                        }}
                        btn.disabled = true;
                        btn.textContent = '⏳ Downloading...';
                        const filename = photoId.split('/').pop() || 'photo.jpg';
                        // Costruisci URL con email e download=true
                        let photoUrl = `/photo/${{encodeURIComponent(photoId)}}?download=true`;
                        if (email) {{
                            photoUrl += `&email=${{encodeURIComponent(email)}}`;
                        }}
                        if (isIOS()) {{
                            // On iOS we don’t force downloads: open fullscreen and let the user long-press -> Save to Photos
                            openViewer(photoUrl.replace('&download=true', ''));
                            btn.textContent = '✅ Opened';
                            setTimeout(() => {{
                                btn.disabled = false;
                                btn.textContent = '📥 Download';
                            }}, 1500);
                            return;
                        }}
                        else if (isAndroid()) {{
                            const link = document.createElement('a');
                            link.href = photoUrl;
                            link.download = filename;
                            link.style.display = 'none';
                            document.body.appendChild(link);
                            link.click();
                            setTimeout(() => {{
                                document.body.removeChild(link);
                            }}, 1000);
                            btn.textContent = '✅ Downloaded!';
                            setTimeout(() => {{
                                btn.disabled = false;
                                btn.textContent = '📥 Download';
                            }}, 2000);
                        }}
                        else {{
                            const response = await fetch(photoUrl);
                            if (!response.ok) {{
                                throw new Error('Download error');
                            }}
                            const blob = await response.blob();
                            const blobUrl = window.URL.createObjectURL(blob);
                            const link = document.createElement('a');
                            link.href = blobUrl;
                            link.download = filename;
                            document.body.appendChild(link);
                            link.click();
                            document.body.removeChild(link);
                            window.URL.revokeObjectURL(blobUrl);
                            btn.textContent = '✅ Downloaded!';
                            setTimeout(() => {{
                                btn.disabled = false;
                                btn.textContent = '📥 Download';
                            }}, 2000);
                        }}
                    }} catch (error) {{
                        alert('Download error. Please try again later.');
                        btn.disabled = false;
                        btn.textContent = '📥 Download';
                    }}
                }}
            </script>
        </body>
        </html>
        """
        
        return HTMLResponse(html_content)
    except Exception as e:
        logger.error(f"Error in my_photos_by_email: {e}", exc_info=True)
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <body style="font-family: Arial; padding: 50px; text-align: center;">
            <h1>❌ Errore</h1>
            <p>{str(e)}</p>
            <a href="/">Back to home</a>
        </body>
        </html>
        """)

@app.get("/download")
async def download_json(
    token: str = Query(..., description="Token di download"),
    request: Request = None
):
    """Endpoint JSON per recuperare foto pagate via token"""
    try:
        order = await _get_order_by_token(token)
        if not order:
            raise HTTPException(status_code=404, detail="Token not found or expired")
        
        # Verifica scadenza
        expires_at = order.get('expires_at')
        if expires_at:
            try:
                expires_date = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                now = datetime.now(timezone.utc)
                if now > expires_date:
                    raise HTTPException(status_code=410, detail="Token expired")
            except Exception as e:
                logger.error(f"Error parsing expires_at: {e}")
        
        return {
            "ok": True,
            "photo_ids": order.get('photo_ids', []),
            "email": order.get('email', ''),
            "expires_at": expires_at
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving order by token: {e}")
        raise HTTPException(status_code=500, detail="Internal error")

@app.get("/my-photos/{token}")
async def my_photos_page(
    token: str,
    request: Request
):
    """Pagina download foto dopo pagamento (con token)"""
    try:
        order = await _get_order_by_token(token)
        
        if not order:
            return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Link non valido - TenerifePictures</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body { font-family: Arial, sans-serif; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; background: linear-gradient(135deg, #7b74ff, #5f58ff); color: #fff; }
                    .container { text-align: center; padding: 40px; background: rgba(255,255,255,0.1); border-radius: 20px; }
                    h1 { font-size: 32px; margin: 0 0 20px; }
                    a { display: inline-block; margin-top: 20px; padding: 12px 24px; background: #fff; color: #5f58ff; text-decoration: none; border-radius: 8px; font-weight: 600; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>❌ Link non valido</h1>
                    <p>Il link che hai utilizzato non è valido o è scaduto.</p>
                    <a href="/">Back to home</a>
                </div>
            </body>
            </html>
            """)
        
        # Verifica scadenza token
        expires_at = order.get('expires_at')
        if expires_at:
            try:
                expires_date = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                now = datetime.now(timezone.utc)
                if now > expires_date:
                    logger.info(f"Token expired: {token[:8]}... expires_at={expires_at}")
                    return HTMLResponse("""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Link scaduto - TenerifePictures</title>
                        <meta charset="utf-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1">
                        <style>
                            body { font-family: Arial, sans-serif; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; background: linear-gradient(135deg, #7b74ff, #5f58ff); color: #fff; }
                            .container { text-align: center; padding: 40px; background: rgba(255,255,255,0.1); border-radius: 20px; }
                            h1 { font-size: 32px; margin: 0 0 20px; }
                            a { display: inline-block; margin-top: 20px; padding: 12px 24px; background: #fff; color: #5f58ff; text-decoration: none; border-radius: 8px; font-weight: 600; }
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <h1>⏰ Link scaduto</h1>
                            <p>Il link che hai utilizzato è scaduto.</p>
                            <a href="/">Back to home</a>
                        </div>
                    </body>
                    </html>
                    """)
                days_remaining = max(0, (expires_date - now).days)
            except Exception as e:
                logger.error(f"Error parsing expires_at: {e}")
                days_remaining = 30
        else:
            days_remaining = 30
        
        photo_ids = order['photo_ids']
        
        # Genera HTML per pagina download mobile-first
        email = order.get('email', '')
        photos_html = ""
        for idx, photo_id in enumerate(photo_ids):
            # URL foto senza download=true (per permettere long-press nativo)
            photo_url = f"/photo/{photo_id}?token={token}&paid=true"
            if email:
                photo_url += f"&email={email}"
            photo_id_escaped = photo_id.replace("'", "\\'").replace('"', '&quot;')
            photos_html += f"""
            <div class="photo-item" data-photo-id="{photo_id_escaped}" data-photo-url="{photo_url}" data-index="{idx}">
                <img src="{photo_url}" alt="Photo {idx + 1}" loading="lazy">
            </div>
            """
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="it">
        <head>
            <title>Le tue foto - TenerifePictures</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0, user-scalable=yes">
            <meta name="apple-mobile-web-app-capable" content="yes">
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
                    background: #0a0a0a; 
                    color: #fff; 
                    padding: 16px;
                    min-height: 100vh;
                }}
                .header {{ 
                    text-align: center; 
                    margin-bottom: 24px; 
                    padding: 0 8px;
                }}
                .header h1 {{ 
                    font-size: 28px; 
                    margin-bottom: 8px; 
                    font-weight: 700;
                }}
                .header p {{
                    font-size: 16px;
                    color: #aaa;
                }}
                .warning {{ 
                    background: rgba(255, 243, 205, 0.15); 
                    color: #ffc107; 
                    padding: 12px 16px; 
                    border-radius: 12px; 
                    margin: 0 0 24px 0; 
                    text-align: center;
                    font-size: 14px;
                    border: 1px solid rgba(255, 193, 7, 0.3);
                }}
                .warning strong {{ 
                    display: block; 
                    margin-bottom: 4px; 
                    font-size: 15px;
                }}
                .photos-grid {{ 
                    display: grid; 
                    grid-template-columns: repeat(2, 1fr); 
                    gap: 12px; 
                    margin: 0 0 24px 0;
                }}
                @media (min-width: 768px) {{
                    .photos-grid {{
                        grid-template-columns: repeat(3, 1fr);
                        gap: 16px;
                    }}
                }}
                @media (min-width: 1024px) {{
                    .photos-grid {{
                        grid-template-columns: repeat(4, 1fr);
                        gap: 20px;
                    }}
                }}
                .photo-item {{ 
                    position: relative; 
                    border-radius: 12px; 
                    overflow: hidden; 
                    background: #1a1a1a;
                    cursor: pointer;
                    aspect-ratio: 1;
                    transition: transform 0.2s;
                }}
                .photo-item:active {{
                    transform: scale(0.98);
                }}
                .photo-item img {{ 
                    width: 100%; 
                    height: 100%;
                    object-fit: cover;
                    display: block;
                    -webkit-touch-callout: default;
                    -webkit-user-select: none;
                    user-select: none;
                }}
                .back-link {{
                    display: block;
                    text-align: center;
                    margin-top: 24px;
                    color: #7b74ff;
                    text-decoration: none;
                    font-size: 16px;
                    padding: 12px;
                }}
                .back-link:active {{
                    opacity: 0.7;
                }}
                
                /* Full screen viewer */
                .viewer {{
                    display: none;
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0, 0, 0, 0.98);
                    z-index: 1000;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    padding: 20px;
                }}
                .viewer.active {{
                    display: flex;
                }}
                .viewer-close {{
                    position: absolute;
                    top: 20px;
                    right: 20px;
                    background: rgba(255, 255, 255, 0.2);
                    border: none;
                    color: white;
                    width: 44px;
                    height: 44px;
                    border-radius: 50%;
                    font-size: 24px;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    z-index: 1001;
                    backdrop-filter: blur(10px);
                }}
                .viewer-close:active {{
                    background: rgba(255, 255, 255, 0.3);
                }}
                .viewer-img {{
                    max-width: 100%;
                    max-height: calc(100vh - 120px);
                    object-fit: contain;
                    -webkit-touch-callout: default;
                    -webkit-user-select: none;
                    user-select: none;
                }}
                .viewer-download {{
                    position: absolute;
                    bottom: 70px;
                    left: 50%;
                    transform: translateX(-50%);
                    background: rgba(255,255,255,0.2);
                    border: 1px solid rgba(255,255,255,0.3);
                    color: #fff;
                    padding: 12px 18px;
                    border-radius: 14px;
                    font-size: 16px;
                    font-weight: 700;
                    cursor: pointer;
                    backdrop-filter: blur(10px);
                }}
                .viewer-download:active {{
                    background: rgba(255,255,255,0.3);
                }}
                .viewer-instruction {{
                    position: absolute;
                    bottom: 20px;
                    left: 50%;
                    transform: translateX(-50%);
                    background: rgba(0, 0, 0, 0.8);
                    color: white;
                    padding: 12px 20px;
                    border-radius: 20px;
                    font-size: 14px;
                    text-align: center;
                    max-width: 90%;
                    backdrop-filter: blur(10px);
                }}
                .viewer-nav {{
                    position: absolute;
                    top: 50%;
                    transform: translateY(-50%);
                    background: rgba(255, 255, 255, 0.2);
                    border: none;
                    color: white;
                    width: 44px;
                    height: 44px;
                    border-radius: 50%;
                    font-size: 20px;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    backdrop-filter: blur(10px);
                }}
                .viewer-nav:active {{
                    background: rgba(255, 255, 255, 0.3);
                }}
                .viewer-nav.prev {{
                    left: 20px;
                }}
                .viewer-nav.next {{
                    right: 20px;
                }}
                .viewer-nav.hidden {{
                    display: none;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>✅ Le tue foto</h1>
                <p>{len(photo_ids)} foto acquistate</p>
            </div>
            
            <div class="warning">
                <strong>⚠️ Disponibili per {days_remaining} giorni</strong>
                Scarica le foto nella tua galleria prima della scadenza
            </div>
            
            <div class="photos-grid">
                {photos_html}
            </div>
            
            <a href="/" class="back-link">← Torna alla home</a>
            
            <!-- Full screen viewer -->
            <div class="viewer" id="viewer">
                <button class="viewer-close" id="viewerClose">×</button>
                <button class="viewer-nav prev" id="viewerPrev">‹</button>
                <img class="viewer-img" id="viewerImg" src="" alt="">
                <button class="viewer-download" id="viewerDownload">⬇︎ Scarica</button>
                <button class="viewer-nav next" id="viewerNext">›</button>
                <div class="viewer-instruction" id="viewerInstruction"></div>
            </div>
            
            <script>
                const photoIds = {json.dumps(photo_ids)};
                const token = '{token}';
                const email = '{email}';
                
                // Salva in localStorage
                if(email) {{
                    localStorage.setItem('userEmail', email);
                }}
                if(token) {{
                    localStorage.setItem('downloadToken', token);
                }}
                
                // Rileva dispositivo
                const isIOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
                const isAndroid = /Android/i.test(navigator.userAgent);
                
                // Setup viewer
                const viewer = document.getElementById('viewer');
                const viewerImg = document.getElementById('viewerImg');
                const viewerClose = document.getElementById('viewerClose');
                const viewerPrev = document.getElementById('viewerPrev');
                const viewerNext = document.getElementById('viewerNext');
                const viewerInstruction = document.getElementById('viewerInstruction');
                const viewerDownload = document.getElementById('viewerDownload');
                
                let currentIndex = 0;
                
                // Mostra istruzioni in base al dispositivo
                function updateInstruction() {{
                    if (isIOS) {{
                        viewerInstruction.textContent = "Tieni premuto sulla foto e tocca 'Salva immagine'";
                        viewerDownload.style.display = 'none';
                    }} else if (isAndroid) {{
                        viewerInstruction.textContent = "Tieni premuto sulla foto e tocca 'Scarica'";
                        viewerDownload.style.display = 'none';
                    }} else {{
                        viewerInstruction.textContent = "Tieni premuto sulla foto per salvare";
                        viewerDownload.style.display = 'block';
                    }}
                }}
                
                function openViewer(index) {{
                    currentIndex = index;
                    const photoId = photoIds[index];
                    const photoUrl = `/photo/${{encodeURIComponent(photoId)}}?token=${{token}}&paid=true${{email ? '&email=' + encodeURIComponent(email) : ''}}`;
                    const downloadUrl = `/photo/${{encodeURIComponent(photoId)}}?token=${{token}}&paid=true&download=true${{email ? '&email=' + encodeURIComponent(email) : ''}}`;
                    viewerImg.src = photoUrl;
                    viewerDownload.setAttribute('data-download-url', downloadUrl);
                    viewer.classList.add('active');
                    updateInstruction();
                    updateNavButtons();
                }}
                
                function closeViewer() {{
                    viewer.classList.remove('active');
                }}
                
                function updateNavButtons() {{
                    viewerPrev.classList.toggle('hidden', currentIndex === 0);
                    viewerNext.classList.toggle('hidden', currentIndex === photoIds.length - 1);
                }}
                
                function showPrev() {{
                    if (currentIndex > 0) {{
                        openViewer(currentIndex - 1);
                    }}
                }}
                
                function showNext() {{
                    if (currentIndex < photoIds.length - 1) {{
                        openViewer(currentIndex + 1);
                    }}
                }}
                
                // Event listeners
                document.querySelectorAll('.photo-item').forEach((item, index) => {{
                    item.addEventListener('click', () => openViewer(index));
                }});
                
                viewerClose.addEventListener('click', closeViewer);
                viewerPrev.addEventListener('click', showPrev);
                viewerNext.addEventListener('click', showNext);
                
                // Download button handler
                viewerDownload.addEventListener('click', (e) => {{
                    e.preventDefault();
                    const url = viewerDownload.getAttribute('data-download-url');
                    if (!url) return;
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = '';
                    document.body.appendChild(a);
                    a.click();
                    a.remove();
                }});
                
                // Chiudi con ESC
                document.addEventListener('keydown', (e) => {{
                    if (e.key === 'Escape' && viewer.classList.contains('active')) {{
                        closeViewer();
                    }} else if (e.key === 'ArrowLeft' && viewer.classList.contains('active')) {{
                        showPrev();
                    }} else if (e.key === 'ArrowRight' && viewer.classList.contains('active')) {{
                        showNext();
                    }}
                }});
                
                // Chiudi cliccando fuori dall'immagine (solo sfondo, non istruzioni)
                viewer.addEventListener('click', (e) => {{
                    if (e.target === viewer) {{
                        closeViewer();
                    }}
                }});
            </script>
        </body>
        </html>
        """
        
        return HTMLResponse(html_content)
    except Exception as e:
        logger.error(f"Error loading my-photos page: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/test-checkout")
async def test_checkout(
    request: Request,
    session_id: str = Query(..., description="ID sessione"),
    email: Optional[str] = Query(None, description="Email utente")
):
    """Endpoint di test per simulare checkout senza Stripe (per test locali)"""
    try:
        logger.info(f"=== TEST CHECKOUT (no Stripe) ===")
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Email: {email}")
        
        photo_ids = await _get_cart(session_id)
        logger.info(f"Cart photo_ids: {photo_ids}")
        
        if not photo_ids:
            raise HTTPException(status_code=400, detail="Cart is empty")
        
        if not email:
            raise HTTPException(status_code=400, detail="Email is required")
        
        # Simula un ordine senza Stripe
        base_url = str(request.base_url).rstrip('/')
        fake_session_id = f"test_session_{int(datetime.now(timezone.utc).timestamp())}"
        
        # Crea ordine nel database
        amount_cents = calculate_price(len(photo_ids))
        logger.info(f"Creating test order: email={email}, order_id={fake_session_id}, photos={len(photo_ids)}, amount={amount_cents}")
        
        try:
            download_token = await _create_order(email, fake_session_id, fake_session_id, photo_ids, amount_cents)
            logger.info(f"Order created, download_token: {download_token}")
        except Exception as order_error:
            logger.error(f"Error creating order: {order_error}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error creating test order: {str(order_error)}")
        
        if download_token:
            # Redirecta direttamente alla pagina di successo
            success_url = f"{base_url}/checkout/success?session_id={fake_session_id}&cart_session={session_id}"
            logger.info(f"Test checkout success, redirecting to: {success_url}")
            return {
                "ok": True,
                "checkout_url": success_url,
                "session_id": fake_session_id,
                "test_mode": True
            }
        else:
            logger.error("_create_order returned None (no token)")
            raise HTTPException(status_code=500, detail="Error creating test order: download_token is None")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in test-checkout: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Test checkout error: {str(e)}")

@app.get("/test-album")
async def test_album(
    email: Optional[str] = Query("test@example.com", description="Email di test")
):
    """Endpoint di test per mostrare galleria con foto di prova senza passare per la camera"""
    if R2_ONLY_MODE:
        raise HTTPException(status_code=503, detail="Test album endpoint disabled in R2_ONLY_MODE. Use R2 for photos.")
    
    try:
        # Prendi le prime 20 foto dalla cartella photos
        photo_files = list(PHOTOS_DIR.glob("*.jpg")) + list(PHOTOS_DIR.glob("*.jpeg"))
        photo_files = [f for f in photo_files if not f.name.endswith('_watermarked.jpg')]  # Escludi watermark
        
        # Limita a 20 foto per test
        photo_files = photo_files[:20]
        
        if not photo_files:
            return {
                "ok": False,
                "error": "Nessuna foto trovata nella cartella photos"
            }
        
        # Crea risultati simulati (stesso formato di match_selfie)
        results = []
        for i, photo_file in enumerate(photo_files):
            photo_id = photo_file.name
            results.append({
                "photo_id": photo_id,
                "score": 0.85 - (i * 0.01),  # Score decrescente per simulare ranking
                "has_face": True,
                "is_back_photo": False
            })
        
        # Se email fornita, salva foto trovate nel database (simula comportamento normale)
        if email:
            for result in results:
                await _add_user_photo(email, result["photo_id"], "found")
            logger.info(f"Test album: saved {len(results)} photos for user {email}")
        
        return {
            "ok": True,
            "count": len(results),
            "matches": results,
            "results": results,
            "matched_count": len(results)
        }
    except Exception as e:
        logger.error(f"Error in test-album endpoint: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e)
        }


# ========== LOGICA RICONOSCIMENTO FACCIALE AVANZATA ==========


# Helper per normalizzare la data tour (accetta YYYY-MM-DD o YYYYMMDD)
def _normalize_tour_date(tour_date: Optional[str]) -> Optional[str]:
    """Normalizza data tour. Accetta YYYY-MM-DD o YYYYMMDD e ritorna YYYY-MM-DD."""
    if not tour_date:
        return None
    td = tour_date.strip()
    # Se arriva già nel formato YYYY-MM-DD, lascialo così
    if "-" in td and len(td) >= 10:
        return td[:10]
    # Se arriva come YYYYMMDD
    digits = "".join(ch for ch in td if ch.isdigit())
    if len(digits) == 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return td

def _is_tiny_or_weak_face(meta: dict) -> bool:
    """
    Verifica se una faccia è troppo piccola o debole (probabilmente falsa/background).
    
    Returns:
        True se la faccia deve essere ignorata (tiny/weak), False altrimenti.
    """
    try:
        score = meta.get("det_score", meta.get("score", None))
        if score is not None and float(score) < 0.60:
            return True

        bbox = meta.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
            if area < 6000:
                return True

            iw = meta.get("image_w", meta.get("w", None))
            ih = meta.get("image_h", meta.get("h", None))
            if iw and ih:
                denom = float(iw) * float(ih)
                if denom > 0:
                    area_ratio = area / denom
                    if area_ratio < 0.015:
                        return True
    except Exception:
        return False
    return False

async def _filter_photos_by_rules(
    selfie_embedding: np.ndarray,
    email: Optional[str] = None,
    min_score: float = 0.35,
    tour_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Filtra le foto usando pipeline STRICT:
    - Raggruppa FAISS results per photo_id e conserva best_score, second_best_score
    - Include una foto SOLO se:
      a) best_score >= min_score (default 0.35)
      b) det_score >= 0.80 (se disponibile)
      c) best_score - second_best_score >= 0.05 (opzionale, solo log)
    - Verifica esistenza su R2 (wm/<photo_id>)
    
    Returns:
        List[Dict]: Lista di foto filtrate con metadata
    """
    filtered_results: List[Dict[str, Any]] = []
    
    if faiss_index is None or len(meta_rows) == 0:
        return filtered_results
    
    # Normalizza embedding selfie
    selfie_emb = _normalize(selfie_embedding).reshape(1, -1)
    
    # Cerca tutte le facce che matchano il selfie
    D, I = faiss_index.search(selfie_emb, len(meta_rows))
    
    # Raggruppa per photo_id: photo_id -> (best_score, second_best_score, best_idx)
    photo_scores: Dict[str, Tuple[float, Optional[float], int]] = {}  # photo_id -> (best_score, second_best_score, best_idx)
    
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0 or idx >= len(meta_rows):
            continue
        if score < float(min_score):
            continue
        
        row = meta_rows[idx]
        # Usa r2_key se disponibile, altrimenti photo_id (retrocompatibilità)
        r2_key = row.get("r2_key") or row.get("photo_id")
        if not r2_key:
                continue

        # NON normalizzare: r2_key deve essere la chiave completa (es. "photos/2024-01-12/tour123/file.jpg")
        # Se contiene "/" è valido (path completo)
        if not r2_key:
            continue

        # Salva best_score e second_best_score per questa foto (usa r2_key come chiave)
        s = float(score)
        if r2_key not in photo_scores:
            photo_scores[r2_key] = (s, None, idx)
        else:
            best, second, best_idx = photo_scores[r2_key]
            if s > best:
                # Nuovo best, il vecchio best diventa second
                photo_scores[r2_key] = (s, best, idx)
            elif second is None or s > second:
                # Nuovo second
                photo_scores[r2_key] = (best, s, best_idx)
    
    # 2) Filtra per regole di matching robuste
    # Include una foto SOLO se:
    # a) best_score >= min_score
    # b) det_score >= 0.80 (se disponibile)
    # c) best_score - second_best_score >= 0.05 (se second esiste, opzionale ma consigliato)
    verified_photo_scores: Dict[str, Tuple[float, Optional[float], int]] = {}
    verified_tour_dates: Dict[str, str] = {}
    filtered_by_det_score = 0
    filtered_by_gap = 0
    
    for r2_key, (best_score, second_best_score, best_idx) in photo_scores.items():
        # a) Verifica best_score >= min_score (già fatto sopra, ma ricontrolla)
        if best_score < float(min_score):
            continue

        # b) Verifica det_score >= 0.85 (STRICT)
        det_score_strict = 0.85
        if best_idx >= 0 and best_idx < len(meta_rows):
            row = meta_rows[best_idx]
            det_score = row.get("det_score") or row.get("score")
            if det_score is not None:
                det_score_float = float(det_score)
                if det_score_float < det_score_strict:
                    filtered_by_det_score += 1
                    continue
            else:
                # Se det_score non disponibile, escludi per sicurezza in STRICT
                filtered_by_det_score += 1
                continue
        
        # c) Verifica gap best - second >= 0.06 (obbligatorio per STRICT)
        gap_strict = 0.06
        if second_best_score is not None:
            gap = best_score - second_best_score
            if gap < gap_strict:
                filtered_by_gap += 1
                continue  # STRICT: gap obbligatorio, escludi se non soddisfatto
        else:
            # Se non c'è second_best_score, escludi per sicurezza in STRICT
            filtered_by_gap += 1
            continue

        verified_photo_scores[r2_key] = (best_score, second_best_score, best_idx)
    
    # 3) Filtra per esistenza su R2 (wm/<filename>)
    r2_verified_scores: Dict[str, Tuple[float, Optional[float], int]] = {}
    
    if USE_R2 and r2_client:
        for r2_key, score_data in verified_photo_scores.items():
            # Costruisci wm_key semplice (r2_key è già filename)
            wm_key = f"{WM_PREFIX}{r2_key}"
            try:
                # Verifica esistenza senza scaricare (head_object è veloce)
                r2_client.head_object(Bucket=R2_BUCKET, Key=wm_key)
                r2_verified_scores[r2_key] = score_data
            except Exception as e:
                # Key non esiste su R2 -> filtra out
                logger.info(f"[FILTER] R2 missing wm for r2_key={r2_key}, filtered out")
                continue
    else:
        # Se R2 non disponibile, usa tutti i verified_photo_scores (fallback)
        r2_verified_scores = verified_photo_scores
    
    # 4) Costruisci i risultati: includi SOLO foto che passano tutti i filtri
    for r2_key, (best_score, second_best_score, best_idx) in r2_verified_scores.items():
        # Estrai display_name (basename) per retrocompatibilità
        from pathlib import Path
        display_name = Path(r2_key).name
        # URL encode r2_key per l'endpoint /photo
        from urllib.parse import quote
        r2_key_encoded = quote(r2_key, safe='')
        filtered_results.append({
            "r2_key": r2_key,  # Chiave R2 completa
            "display_name": display_name,  # Solo nome file
            "photo_id": r2_key,  # Retrocompatibilità
            "score": best_score,
            "has_face": True,
            "has_selfie": True,
            "wm_url": f"/photo/{r2_key_encoded}?variant=wm",  # URL per preview watermarked
            "thumb_url": f"/photo/{r2_key_encoded}?variant=thumb",  # URL per thumbnail
        })
    
    # Log: numero candidati FAISS sopra soglia
    candidates_above_threshold = sum(1 for score in D[0].tolist() if score >= float(min_score))
    logger.info(f"[FILTER] FAISS candidates above threshold ({min_score}): {candidates_above_threshold}")
    
    # Log: numero photo_id unici dopo grouping
    logger.info(f"[FILTER] Unique photo_ids after grouping: {len(photo_scores)}")
    
    # Log: numero finali dopo filtro det_score e gap
    logger.info(f"[STRICT] Filtered by det_score < 0.85: {filtered_by_det_score}, by gap < 0.06: {filtered_by_gap}")
    logger.info(f"[STRICT] Final photos after all filters (det_score + gap + R2): {len(filtered_results)}")
    
    # Ordina per score decrescente
    filtered_results.sort(key=lambda x: x["score"], reverse=True)
    
    return filtered_results

# ========== ENDPOINT ESISTENTI ==========

def _extract_frames_from_video(video_bytes: bytes, max_frames: int = 5) -> List[np.ndarray]:
    """
    Estrae N frame chiave da un video (mp4/mov).
    
    Args:
        video_bytes: Bytes del video
        max_frames: Numero massimo di frame da estrarre (default 5)
    
    Returns:
        Lista di frame (immagini OpenCV BGR) ridimensionate a max 720px lato lungo
    """
    import tempfile
    import os
    
    frames = []
    
    # Salva video in file temporaneo
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_path = tmp_file.name
        tmp_file.write(video_bytes)
    
    try:
        # Apri video con OpenCV
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            logger.warning("[VIDEO] Failed to open video file")
            return frames
        
        # Ottieni FPS e frame count
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count == 0 or fps <= 0:
            logger.warning("[VIDEO] Video has zero frames or invalid FPS")
            cap.release()
            return frames
        
        # Calcola frame target per percentuali: 10%, 30%, 50%, 70%, 90%
        percentages = [0.1, 0.3, 0.5, 0.7, 0.9]
        frame_targets = [int(frame_count * p) for p in percentages[:max_frames]]
        
        for target_frame in frame_targets:
            # Vai al frame corrispondente
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Ridimensiona a max 720px lato lungo
            h, w = frame.shape[:2]
            max_dim = max(h, w)
            if max_dim > 720:
                scale = 720.0 / max_dim
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            frames.append(frame)
        
        cap.release()
        duration = frame_count / fps if fps > 0 else 0
        logger.info(f"[VIDEO] Extracted {len(frames)} frames from video (frames={frame_count} fps={fps:.1f} duration={duration:.2f}s)")
        
    except Exception as e:
        logger.error(f"[VIDEO] Error extracting frames: {e}")
    finally:
        # Rimuovi file temporaneo
        try:
            os.unlink(tmp_path)
        except:
            pass
    
    return frames

def _extract_embeddings_from_frames(frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    Estrae embedding da una lista di frame.
    
    Args:
        frames: Lista di frame (immagini OpenCV BGR)
    
    Returns:
        Lista di embeddings normalizzati (uno per frame)
    """
    assert face_app is not None
    embeddings = []
    
    for idx, frame in enumerate(frames):
        faces = face_app.get(frame)
        if faces:
            # Prendi il volto più grande
            faces_sorted = sorted(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True
            )
            if faces_sorted:
                embedding = faces_sorted[0].embedding.astype(np.float32)
                embedding = _normalize(embedding)
                embeddings.append(embedding)
                logger.debug(f"[VIDEO] Frame {idx+1}: face detected")
    
    return embeddings

@app.post("/match_selfie")
async def match_selfie(
    selfie: UploadFile = File(...),
    email: Optional[str] = Query(None, description="Email utente (opzionale, per salvare foto trovate)"),
    top_k_faces: int = Query(600, description="Numero massimo di volti da cercare (default 600, aumentato per catturare foto difficili)"),
    min_score: float = Query(0.25, description="Soglia minima di similarità (default 0.25)"),
    # Parametri per two-pass query expansion
    strict_min_score: float = Query(0.35, description="Soglia strict per Pass 1 (default 0.35)"),
    soft_min_score: float = Query(0.30, description="Soglia soft per Pass 2 (default 0.30)"),
    det_min_score: float = Query(0.65, description="Soglia det_score per Pass 1 (default 0.65)"),
    det_min_score_soft: float = Query(0.55, description="Soglia det_score per Pass 2 (default 0.55, più permissiva)"),
    top_k_pass1: int = Query(200, description="Top K per Pass 1 (default 200)"),
    top_k_pass2: int = Query(400, description="Top K per Pass 2 (default 400)"),
    target_min_results: int = Query(15, description="Minimo risultati Pass 1 per saltare Pass 2 (default 15)")
):
    """
    Endpoint per face matching single-pass:
    - Accetta selfie (immagine) o mini-video (mp4/mov, 2-3 secondi)
    - Se video: estrae 5 frame chiave e crea embedding medio
    - Single-pass FAISS (top_k configurabile) + filtri dinamici (score/margin)
    - Nessun check R2 durante match (solo embedding + FAISS + metadata)
    - Ritorna fino a 80 foto ordinate per score
    - Logging dettagliato per debug
    """
    global meta_rows
    _ensure_ready()
    
    start_time = time.time()
    
    # Salva meta_rows originale per ripristino (prima di qualsiasi return)
    old_meta_rows = meta_rows
    
    try:
        # Gestisci indice vuoto o metadata mancanti
        if faiss_index is None or faiss_index.ntotal == 0 or len(meta_rows) == 0:
            logger.info("Index is empty or not loaded - returning empty result")
            return {
                "ok": True,
                "count": 0,
                "matches": [],
                "results": [],
                "matched_count": 0,
                "message": "Foto non trovate oppure ancora in elaborazione. Riprova più tardi."
            }
        
        # RICARICA SEMPRE I METADATA DAI FILE JSON
        current_meta_rows = []
        if R2_ONLY_MODE:
            current_meta_rows = meta_rows
        elif META_PATH.exists():
            current_meta_rows = _load_meta_jsonl(META_PATH)
        else:
            current_meta_rows = []
        
        if len(current_meta_rows) == 0:
            logger.info("No photos in metadata - returning empty result")
            return {
                "ok": True,
                "count": 0,
                "matches": [],
                "results": [],
                "matched_count": 0,
                "message": "Foto non trovate oppure ancora in elaborazione. Riprova più tardi."
            }
        
        # Leggi file (può essere immagine o video)
        file_bytes = await selfie.read()
        content_type = selfie.content_type or ""
        filename = selfie.filename or ""
        
        # Log upload
        import hashlib
        sha1_hash = hashlib.sha1(file_bytes).hexdigest()[:8]
        logger.info(f"[UPLOAD] content_type={content_type} size={len(file_bytes)} sha1={sha1_hash}")
        
        # Determina se è video o immagine
        is_video = (
            content_type.startswith("video/") or
            filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")) or
            (len(file_bytes) > 12 and file_bytes[4:8] in [b"ftyp", b"moov"])  # Magic bytes per MP4/MOV
        )
        
        ref_embeddings: List[np.ndarray] = []
        selfie_start = time.time()
        
        if is_video:
            # Estrai frame da video
            frames = _extract_frames_from_video(file_bytes, max_frames=5)
            if not frames:
                return {
                    "ok": True,
                    "count": 0,
                    "matches": [],
                    "results": [],
                    "matched_count": 0,
                    "message": "Nessun frame valido estratto dal video. Assicurati che il video contenga volti visibili."
                }
            
            # Estrai embedding da ogni frame
            frame_embeddings = _extract_embeddings_from_frames(frames)
            
            if len(frame_embeddings) < 2:
                return {
                    "ok": True,
                    "count": 0,
                    "matches": [],
                    "results": [],
                    "matched_count": 0,
                    "message": "Volti rilevati insufficienti nel video. Assicurati che almeno 2 frame contengano volti visibili."
                }
            
            # Usa tutti gli embeddings dei frame come reference embeddings
            ref_embeddings = frame_embeddings
            logger.info(f"[VIDEO] frames_ok={len(frames)} ref_embeddings={len(ref_embeddings)}")
        else:
            # Immagine: genera multipli embeddings con augmentazioni
            img = _read_selfie_image_with_resize(file_bytes, max_side=1024)
            assert face_app is not None
            faces = face_app.get(img)
            
            if not faces:
                return {
                    "ok": True,
                    "count": 0,
                    "matches": [],
                    "results": [],
                    "matched_count": 0,
                    "message": "Nessun volto rilevato nel selfie. Assicurati che il volto sia ben visibile."
                }
            
            # Genera multipli embeddings con augmentazioni leggere
            ref_embeddings = _generate_multi_embeddings_from_image(img, num_embeddings=2)
            
            if len(ref_embeddings) == 0:
                return {
                    "ok": True,
                    "count": 0,
                    "matches": [],
                    "results": [],
                    "matched_count": 0,
                    "message": "Impossibile estrarre embedding dal selfie."
                }
            
            logger.info(f"[IMAGE] ref_embeddings={len(ref_embeddings)}")
        
        selfie_elapsed_ms = (time.time() - selfie_start) * 1000
        
        if len(ref_embeddings) == 0:
            return {
                "ok": True,
                "count": 0,
                "matches": [],
                "results": [],
                "matched_count": 0,
                "message": "Impossibile estrarre embedding dal selfie."
            }
        
        # Aggiorna temporaneamente meta_rows
            meta_rows = current_meta_rows
        
        # ========== SINGLE-PASS MATCHING ==========
        n_faces_index = faiss_index.ntotal
        n_meta_rows = len(meta_rows)
        top_k = min(top_k_faces, n_faces_index)
        logger.info(f"[MATCH] Starting single-pass: n_faces_index={n_faces_index} meta_rows={n_meta_rows} ref_embeddings={len(ref_embeddings)} top_k={top_k}")

        def _compute_face_area(meta: Dict[str, Any]) -> float:
            bbox = meta.get("bbox")
            if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                try:
                    x1, y1, x2, y2 = bbox
                    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))
                except Exception:
                    return 0.0
            return 0.0

        def _dynamic_min_score(det_score_val: float, area: float) -> float:
            # Threshold adattivi per bilanciare recall e precision (protezione contro falsi positivi)
            # Ottimizzato per catturare foto difficili (profilo, lontane, parzialmente coperte)
            if det_score_val >= 0.85 and area >= 60000:
                return 0.28  # Foto "facili": soglia più bassa ma con conferma 2/2 se score < 0.40
            if det_score_val >= 0.75 and area >= 30000:
                return 0.31  # Foto "medie": soglia moderata
            # Foto con area MOLTO GRANDE (>150000): potrebbero essere molto lontane/profilo
            # Abbassa soglia a 0.10 per catturare meglio (foto difficili ma grandi, richiede sempre 2/2 hits)
            if area >= 150000:
                return 0.10  # Foto molto grandi: soglia molto bassa per lontane/profilo (protezione: 2/2 hits)
            # Foto con area GRANDE (>70000) anche con det_score basso: potrebbero essere lontane/profilo
            # Abbassa soglia a 0.20 per catturare meglio (se det_score alto, altrimenti 0.25 o 0.20 se det molto basso)
            if area >= 70000:
                if det_score_val >= 0.70:
                    return 0.20  # Foto grandi con det buono: soglia bassa
                elif det_score_val < 0.55:
                    return 0.20  # Foto grandi con det molto basso: soglia bassa per catturare foto difficili
                else:
                    return 0.25  # Foto grandi con det medio: soglia un po' più alta
            # Foto con area GRANDE (>150000) ma det_score medio (0.68-0.75): potrebbero essere profili/lontane
            # Abbassa soglia a 0.28 per catturare meglio (come foto "facili" ma con det medio)
            if 0.68 <= det_score_val < 0.75 and area >= 150000:
                return 0.28  # Foto grandi ma det medio: soglia molto bassa per profili/lontane
            # Foto difficili ma con det_score buono (>=0.70): potrebbero essere profili/lontane
            # Abbassa soglia a 0.30 ma richiede sempre 2/2 hits per protezione
            if det_score_val >= 0.70 and 15000 <= area < 20000:
                return 0.30  # Foto difficili (profilo/lontane) con det buono: soglia più bassa
            # Per foto "small" ma con det_score alto (>=0.75) e area decente (>=20000): usa 0.32
            if det_score_val >= 0.75 and 20000 <= area < 30000:
                return 0.32  # Soglia conservativa: richiede sempre 2/2 hits
            # Foto con det_score medio-alto (0.65-0.75) e area piccola: potrebbero essere profili
            if 0.65 <= det_score_val < 0.75 and 10000 <= area < 20000:
                return 0.32  # Soglia per foto difficili con det medio
            # Per bucket speciale (0.62 <= det < 0.75, 10000 <= area < 30000): usa 0.25 invece di 0.30
            # Foto piccole ma con det_score medio: potrebbero essere profili/angolate
            if 0.62 <= det_score_val < 0.75 and 10000 <= area < 30000:
                return 0.25  # Ridotto da 0.30 per catturare foto difficili piccole
            # Foto piccole (area < 30000) con det_score medio (0.60-0.75): potrebbero essere profili
            if 0.60 <= det_score_val < 0.75 and area < 30000:
                return 0.25  # Soglia più bassa per foto piccole con det medio
            # Foto piccole (area < 30000) con det_score basso (0.55-0.60): potrebbero essere profili molto difficili
            if 0.55 <= det_score_val < 0.60 and area < 30000:
                return 0.28  # Soglia bassa ma richiede sempre 2/2 hits
            return 0.35  # Default conservativo per tutte le altre foto "small"

        def _dynamic_margin_min(det_score_val: float, area: float) -> float:
            if area < 30000 or det_score_val < 0.75:
                return 0.03
            return 0.015
        
        try:
            candidates_by_photo: Dict[str, Dict[str, Any]] = {}
            total_candidates = 0

            # Single-pass FAISS search (top_k)
            for ref_idx, ref_emb in enumerate(ref_embeddings):
                D, I = faiss_index.search(ref_emb.reshape(1, -1), top_k)
                total_candidates += len(I[0])
                for score, idx in zip(D[0].tolist(), I[0].tolist()):
                    if idx < 0 or idx >= len(meta_rows):
                        continue
                    row = meta_rows[idx]
                    r2_key = row.get("r2_key") or row.get("photo_id")
                    if not r2_key:
                        continue
                    s = float(score)

                    entry = candidates_by_photo.get(r2_key)
                    if entry is None:
                        det = row.get("det_score") or row.get("score")
                        det_f = float(det) if det is not None else 0.0
                        area = _compute_face_area(row)
                        entry = {
                            "r2_key": r2_key,
                            "best_score": s,
                            "best_idx": idx,
                            "det_score": det_f,
                            "area": area,
                            # face_scores: best score per face idx (across ref embeddings)
                            "face_scores": {idx: s},
                            # ref_max: best score per ref embedding
                            "ref_max": [0.0] * len(ref_embeddings),
                        }
                        candidates_by_photo[r2_key] = entry
                    else:
                        face_scores = entry["face_scores"]
                        current_face_best = face_scores.get(idx)
                        if current_face_best is None or s > current_face_best:
                            face_scores[idx] = s
                        if s > entry["best_score"]:
                            entry["best_score"] = s
                            entry["best_idx"] = idx
                            det = row.get("det_score") or row.get("score")
                            entry["det_score"] = float(det) if det is not None else 0.0
                            entry["area"] = _compute_face_area(row)
                    # aggiorna ref_max
                    if ref_idx < len(entry["ref_max"]) and s > entry["ref_max"][ref_idx]:
                        entry["ref_max"][ref_idx] = s

            results: List[Dict[str, Any]] = []
            rejected: List[Dict[str, Any]] = []
            stats = {
                "filtered_by_score": 0,
                "filtered_by_margin": 0,
            }
            
            # Log versione protezioni (per verificare che i cambiamenti siano attivi su Render)
            logger.info("[PROTECTION_VERSION] det>=0.90: score>=0.50, det>=0.85: score>=0.50, det>=0.80: score>=0.30, det>=0.78: score>=0.25")

            for r2_key, c in candidates_by_photo.items():
                best_score = float(c["best_score"])
                det_score_val = float(c.get("det_score") or 0.0)
                area = float(c.get("area") or 0.0)
                face_scores = c.get("face_scores", {})
                scores_sorted = sorted(face_scores.values(), reverse=True)
                second_best = scores_sorted[1] if len(scores_sorted) > 1 else None
                margin = (best_score - second_best) if second_best is not None else None

                min_score_dyn = _dynamic_min_score(det_score_val, area)
                margin_min = _dynamic_margin_min(det_score_val, area)
                hits_count = sum(1 for v in c.get("ref_max", []) if v >= min_score_dyn)
                bucket = "large"
                if area < 30000 or det_score_val < 0.75:
                    bucket = "small"
                elif area < 60000 or det_score_val < 0.85:
                    bucket = "medium"

                reject_reason = None
                # PROTEZIONE CRITICA: det_score alto ma score molto basso = SEMPRE RIFIUTA (privacy)
                # Non applicare tolleranza per evitare falsi positivi (privacy violata)
                # Pattern tipico di falsi positivi: faccia ben visibile (det alto) ma match debole (score basso)
                # PROTEZIONE MOLTO AGGRESSIVA: det_score molto alto (>=0.85) = soglia più alta
                # VERSIONE: det>=0.90 richiede score>=0.50, det>=0.85 richiede score>=0.50 (ultra-aggressive)
                # Se det_score >= 0.90, richiedi score >= 0.50 per evitare falsi positivi critici
                if det_score_val >= 0.90 and best_score < 0.50:
                    stats["filtered_by_score"] += 1
                    reject_reason = f"score={best_score:.3f}<0.50 (det={det_score_val:.3f} molto molto alto, falso positivo)"
                # Se det_score >= 0.85, richiedi score >= 0.50 per evitare falsi positivi (es. MIT00045.jpg, MIT00044.jpg, MIT00062.jpg)
                elif det_score_val >= 0.85 and best_score < 0.50:
                    stats["filtered_by_score"] += 1
                    reject_reason = f"score={best_score:.3f}<0.50 (det={det_score_val:.3f} molto alto, falso positivo)"
                # PROTEZIONE SPECIALE: foto con area molto grande (>150000) e det_score medio-alto (0.70-0.85)
                # Queste foto hanno min_score=0.10 (troppo basso), ma se det_score è buono e score è basso = falso positivo
                # Es. MIT00044.jpg (det=0.728, area=216071, score=0.127) e MIT00062.jpg (det=0.727, area=243526, score=0.161)
                elif area >= 150000 and 0.70 <= det_score_val < 0.85 and best_score < 0.25:
                    stats["filtered_by_score"] += 1
                    reject_reason = f"score={best_score:.3f}<0.25 (det={det_score_val:.3f} area_grande={int(area)}, falso positivo)"
                elif det_score_val >= 0.80 and best_score < 0.30:
                    stats["filtered_by_score"] += 1
                    reject_reason = f"score={best_score:.3f}<0.30 (det={det_score_val:.3f} molto alto, falso positivo)"
                elif det_score_val >= 0.78 and best_score < 0.25:
                    # Protezione anche per det_score molto vicino a 0.80 (es. 0.797)
                    stats["filtered_by_score"] += 1
                    reject_reason = f"score={best_score:.3f}<0.25 (det={det_score_val:.3f} alto, falso positivo)"
                elif det_score_val >= 0.75 and best_score < 0.20:
                    stats["filtered_by_score"] += 1
                    reject_reason = f"score={best_score:.3f}<0.20 (det={det_score_val:.3f} alto, falso positivo)"
                else:
                    # Tolleranza per score molto vicini alla soglia
                    # Per foto molto grandi, tolleranza maggiore
                    score_diff = min_score_dyn - best_score
                    tolerance = 0.01  # Default
                    if area >= 150000:
                        tolerance = 0.08  # Foto molto grandi: tolleranza 0.08 (8%) per soglia 0.10
                    elif area >= 70000 or det_score_val >= 0.68:
                        tolerance = 0.03  # Foto grandi o det buono: tolleranza 0.03
                    elif 0.60 <= det_score_val < 0.75 and area < 30000:
                        tolerance = 0.08  # Foto piccole con det medio: tolleranza 0.08 per catturare profili difficili
                    elif 0.55 <= det_score_val < 0.60 and area < 30000:
                        tolerance = 0.10  # Foto piccole con det basso: tolleranza 0.10 per profili molto difficili
                    
                    if score_diff > tolerance:  # Differenza > tolerance = rifiuta
                        stats["filtered_by_score"] += 1
                        reject_reason = f"score={best_score:.3f}<{min_score_dyn:.2f}"
                    elif score_diff > 0:  # Differenza <= tolerance = accetta ma richiede 2/2 hits
                        # Score molto vicino alla soglia: accetta ma richiede conferma doppia
                        # Considera come se avesse superato la soglia, ma richiedi 2/2 hits
                        min_score_dyn = best_score  # Aggiusta min_score per permettere il check successivo
                        # Ricalcola hits_count con la nuova soglia
                        hits_count = sum(1 for v in c.get("ref_max", []) if v >= min_score_dyn)
                    # Se score_diff <= 0 (cioè best_score >= min_score_dyn), continua con la logica di conferma normale
                
                # Logica di conferma adattiva per bilanciare recall e precision (solo se non c'è reject_reason)
                if reject_reason is None:
                    # Logica di conferma adattiva per bilanciare recall e precision
                    # Ottimizzata per catturare foto difficili (profilo, lontane, parzialmente coperte)
                    required_hits = 1  # Default: almeno 1/2 ref_embeddings devono matchare
                    
                    # PROTEZIONE GENERALE: det_score alto ma score basso = possibile falso positivo
                    # PROTEZIONE MOLTO AGGRESSIVA: det_score molto alto (>=0.85) = richiedi sempre 2/2 hits se score < 0.45
                    # Se det_score >= 0.90, richiedi sempre 2/2 hits se score < 0.50
                    # Se det_score >= 0.80 (faccia molto ben visibile) ma score < 0.30, richiedi SEMPRE 2/2 hits
                    # Se det_score >= 0.78 (faccia molto ben visibile) ma score < 0.25, richiedi SEMPRE 2/2 hits
                    # Se det_score >= 0.75 (faccia ben visibile) ma score < 0.20, richiedi SEMPRE 2/2 hits
                    # PROTEZIONE CRITICA: anche se score è >= 0.25 ma < 0.35 con det molto alto, richiedi 2/2
                    if det_score_val >= 0.90 and best_score < 0.50:
                        required_hits = 2  # Det molto molto alto (>=0.90) + score basso = SEMPRE 2/2 hits (protezione critica)
                    elif det_score_val >= 0.85 and best_score < 0.50:
                        required_hits = 2  # Det molto alto (>=0.85) + score basso = SEMPRE 2/2 hits (protezione critica)
                    elif det_score_val >= 0.80 and best_score < 0.35:
                        required_hits = 2  # Det molto alto + score basso = SEMPRE 2/2 hits (protezione critica)
                    elif det_score_val >= 0.78 and best_score < 0.30:
                        required_hits = 2  # Det molto alto (vicino a 0.80) + score basso = SEMPRE 2/2 hits
                    elif det_score_val >= 0.75 and best_score < 0.25:
                        required_hits = 2  # Det alto + score basso = SEMPRE 2/2 hits (protezione falsi positivi)
                    # PROTEZIONE SPECIALE: foto con area molto grande (>150000) e det_score medio-alto (0.70-0.85)
                    # Richiedi sempre 2/2 hits se score < 0.30 (per bloccare MIT00044.jpg e MIT00062.jpg)
                    elif area >= 150000 and 0.70 <= det_score_val < 0.85 and best_score < 0.30:
                        required_hits = 2  # Area grande + det medio + score basso = SEMPRE 2/2 hits
                    # Foto "facili" (large): se score è molto alto (>=0.40), accetta anche con 1/2 hits
                    # Altrimenti richiedi 2/2 per evitare falsi positivi
                    elif bucket == "large":
                        if best_score >= 0.40:
                            required_hits = 1  # Score alto = match sicuro, accetta con 1/2
                        else:
                            required_hits = 2  # Score medio = richiedi conferma 2/2
                    # Foto con area MOLTO GRANDE (>150000): potrebbero essere molto lontane/profilo
                    # Se score è >= 0.15, accetta con 1/2 hits
                    # Se score è >= 0.10 (min_score), accetta con 2/2 hits (protezione)
                    elif area >= 150000:
                        if best_score >= 0.15:
                            required_hits = 1  # Score buono + area molto grande = accetta con 1/2
                        elif best_score >= min_score_dyn:
                            required_hits = 2  # Score basso ma >= min_score = richiedi 2/2 per protezione
                        else:
                            required_hits = 2  # Score molto basso, richiedi 2/2
                    # Foto con area GRANDE (>70000): potrebbero essere lontane/profilo
                    # PROTEZIONE: se score è molto basso (<0.20), richiedi SEMPRE 2/2 hits per evitare falsi positivi
                    # Se score è >= 0.25 (o min_score se più alto), accetta con 1/2 hits
                    # Se score è >= min_score ma < 0.25, accetta con 2/2 hits (protezione)
                    elif area >= 70000:
                        if best_score < 0.20:
                            required_hits = 2  # Score molto basso: SEMPRE 2/2 hits per evitare falsi positivi
                        elif best_score >= 0.25 or (best_score >= min_score_dyn and det_score_val >= 0.70 and best_score >= 0.22):
                            required_hits = 1  # Score buono + area grande = accetta con 1/2
                        elif best_score >= min_score_dyn:
                            required_hits = 2  # Score basso ma >= min_score = richiedi 2/2 per protezione
                        else:
                            required_hits = 2  # Score borderline, richiedi 2/2
                    # Foto difficili ma con det_score buono (>=0.70): potrebbero essere profili/lontane
                    # Accetta con 2/2 hits se score è borderline (0.28-0.32)
                    elif det_score_val >= 0.70 and 15000 <= area < 20000:
                        if best_score >= 0.33:
                            required_hits = 1  # Score buono, accetta con 1/2
                        else:
                            required_hits = 2  # Score borderline, richiedi 2/2 (ma accetta foto difficili)
                    # Foto con det_score medio-alto (0.65-0.75) e area piccola: potrebbero essere profili
                    elif 0.65 <= det_score_val < 0.75 and 10000 <= area < 20000:
                        if best_score >= 0.35:
                            required_hits = 1  # Score buono, accetta con 1/2
                        else:
                            required_hits = 2  # Score borderline, richiedi 2/2
                    # Bucket speciale: small/medium det -> richiede 2/2 hits solo se score borderline
                    # min_score è ora 0.25, quindi se score >= 0.28 accetta con 1/2 hits
                    elif 0.62 <= det_score_val < 0.75 and 10000 <= area < 30000:
                        if best_score >= 0.28:
                            required_hits = 1  # Score buono, accetta con 1/2
                        else:
                            required_hits = 2  # Score borderline, richiedi 2/2
                    # Foto piccole (area < 30000) con det_score medio (0.60-0.75): potrebbero essere profili
                    elif 0.60 <= det_score_val < 0.75 and area < 30000:
                        if best_score >= 0.28:
                            required_hits = 1  # Score buono, accetta con 1/2
                        elif best_score >= min_score_dyn:
                            # Score >= min_score ma < 0.28: accetta con 1/2 se score è almeno min_score
                            required_hits = 1  # Permetti 1/2 hits per foto difficili con score valido
                        else:
                            required_hits = 2  # Score borderline, richiedi 2/2
                    # Foto piccole (area < 30000) con det_score basso (0.55-0.60): potrebbero essere profili molto difficili
                    elif 0.55 <= det_score_val < 0.60 and area < 30000:
                        if best_score >= 0.30:
                            required_hits = 1  # Score buono, accetta con 1/2
                        else:
                            required_hits = 2  # Score borderline, richiedi 2/2
                    # Nuovo bucket: det_score >= 0.75 e 20000 <= area < 30000 (soglia 0.32)
                    # SEMPRE richiede 2/2 hits per evitare falsi positivi (anche se score è buono)
                    elif det_score_val >= 0.75 and 20000 <= area < 30000:
                        required_hits = 2  # Sempre 2/2 hits per questo bucket (protezione anti-falsi positivi)
                    # Foto "difficili" (small): più permissive, ma almeno 2/2 se score molto borderline
                    elif bucket == "small" and best_score < (min_score_dyn + 0.03):
                        required_hits = 2  # Se score è molto borderline, richiedi 2/2
                    
                    if hits_count < required_hits:
                        stats["filtered_by_score"] += 1
                        reject_reason = f"hits={hits_count}/{required_hits} (bucket={bucket}, score={best_score:.3f})"
                    
                    if reject_reason is None:
                        # Per foto "facili" (large), se margin è None (solo una faccia), 
                        # essere più permissivi: richiedi solo best_score >= min_score (già verificato sopra)
                        if margin is None:
                            # Per foto difficili con det_score buono (>=0.70), essere più permissivi
                            # Potrebbero essere profili/lontane con solo una faccia visibile
                            if det_score_val >= 0.70 and best_score >= min_score_dyn:
                                # Foto difficile ma valida: accetta anche senza margin
                                pass
                            # Per foto piccole con det_score medio (0.60-0.75), essere più permissivi
                            # Potrebbero essere profili con una sola faccia visibile
                            elif 0.60 <= det_score_val < 0.75 and area < 30000 and best_score >= min_score_dyn:
                                # Foto piccola con det medio: accetta anche senza margin
                                pass
                            # Per foto grandi (area >= 70000) con score buono, essere più permissivi
                            elif area >= 70000 and best_score >= min_score_dyn:
                                # Foto grande con score buono: accetta anche senza margin
                                pass
                            elif bucket != "large" and best_score < (min_score_dyn + 0.02):
                                stats["filtered_by_margin"] += 1
                                reject_reason = f"margin_missing best<{min_score_dyn + 0.02:.2f}"
                        else:
                            # Per foto difficili con det_score buono, riduci margin_min
                            # Ma AUMENTA margin_min per foto con score bassi (protezione falsi positivi)
                            effective_margin_min = margin_min
                            # PROTEZIONE CRITICA: det_score alto ma score basso = AUMENTA margin_min
                            # PROTEZIONE MOLTO AGGRESSIVA: det_score molto alto (>=0.85) = margin_min molto alto
                            if det_score_val >= 0.90 and best_score < 0.50:
                                # Det molto molto alto (>=0.90) + score basso: margin_min molto alto
                                effective_margin_min = max(margin_min, 0.15)  # Almeno 0.15
                            elif det_score_val >= 0.85 and best_score < 0.50:
                                # Det molto alto (>=0.85) + score basso: margin_min molto alto per evitare falsi positivi
                                effective_margin_min = max(margin_min, 0.15)  # Almeno 0.15 (aumentato per MIT00045.jpg, MIT00044.jpg, MIT00062.jpg)
                            elif det_score_val >= 0.80 and best_score < 0.35:
                                # Det molto alto + score basso: margin_min molto alto per evitare falsi positivi
                                effective_margin_min = max(margin_min, 0.08)  # Almeno 0.08
                            elif det_score_val >= 0.78 and best_score < 0.30:
                                # Det molto alto (vicino a 0.80) + score basso: margin_min alto
                                effective_margin_min = max(margin_min, 0.07)  # Almeno 0.07
                            elif det_score_val >= 0.75 and best_score < 0.25:
                                # Det alto + score basso: margin_min alto
                                effective_margin_min = max(margin_min, 0.06)  # Almeno 0.06
                            elif best_score < 0.20:
                                # Score molto basso: AUMENTA margin_min per evitare falsi positivi
                                effective_margin_min = max(margin_min, 0.05)  # Almeno 0.05 per score bassi
                            elif 0.60 <= det_score_val < 0.75 and area < 30000:
                                # Foto piccola con det medio: riduci margin_min per essere più permissivi (profili)
                                effective_margin_min = margin_min * 0.3  # Ridotto del 70% per catturare profili difficili
                            elif 0.55 <= det_score_val < 0.60 and area < 30000:
                                # Foto piccola con det basso: riduci margin_min ancora di più per profili molto difficili
                                effective_margin_min = margin_min * 0.2  # Ridotto dell'80%
                            elif det_score_val >= 0.70 and area < 20000:
                                # Foto difficile: riduci margin_min del 50% per essere più permissivi
                                effective_margin_min = margin_min * 0.5
                            elif area >= 150000 and best_score >= min_score_dyn:
                                # Foto molto grande con score valido: riduci margin_min per essere più permissivi
                                effective_margin_min = margin_min * 0.3  # Ridotto del 70% per foto molto grandi
                            
                            if margin < effective_margin_min:
                                stats["filtered_by_margin"] += 1
                                reject_reason = f"margin={margin:.3f}<{effective_margin_min:.3f}"

                if reject_reason is None:
                    from pathlib import Path
                    from urllib.parse import quote
                    r2_key_encoded = quote(r2_key, safe='')
                    
                    # Log dettagliato per foto accettate (per debug falsi positivi)
                    # Log sempre per MIT00044.jpg e MIT00062.jpg (falsi positivi noti)
                    if "MIT00044" in r2_key or "MIT00062" in r2_key or det_score_val >= 0.85:
                        margin_str = "None" if margin is None else f"{margin:.3f}"
                        effective_margin_str = f"{effective_margin_min:.3f}" if margin is not None else "N/A"
                        logger.warning(
                            f"[ACCEPTED] {r2_key}: score={best_score:.3f} det={det_score_val:.3f} "
                            f"area={int(area)} min_score={min_score_dyn:.2f} margin={margin_str} "
                            f"margin_min={effective_margin_str} hits={hits_count}/{required_hits} bucket={bucket}"
                        )
                    
                    results.append({
                        "r2_key": r2_key,
                        "display_name": Path(r2_key).name,
                        "photo_id": r2_key,
                        "score": best_score,
                        "has_face": True,
                        "has_selfie": True,
                        "wm_url": f"/photo/{r2_key_encoded}?variant=wm",
                        "thumb_url": f"/photo/{r2_key_encoded}?variant=thumb",
                    })
                else:
                    rejected.append({
                        "r2_key": r2_key,
                        "score": best_score,
                        "det_score": det_score_val,
                        "area": int(area),
                        "min_score": min_score_dyn,
                        "margin": margin,
                        "margin_min": margin_min,
                        "hits": hits_count,
                        "bucket": bucket,
                        "reason": reject_reason,
                    })

            # Deduplica risultati per r2_key (evita foto duplicate)
            seen_r2_keys = set()
            deduplicated_results = []
            for r in results:
                r2_key = r.get("r2_key") or r.get("photo_id")
                if r2_key and r2_key not in seen_r2_keys:
                    seen_r2_keys.add(r2_key)
                    deduplicated_results.append(r)
            results = deduplicated_results
            
            results.sort(key=lambda x: x["score"], reverse=True)
            if len(results) > 80:
                results = results[:80]

            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                f"[MATCH] elapsed_ms={elapsed_ms:.1f} selfie_ms={selfie_elapsed_ms:.1f} "
                f"photos={len(results)} candidates={len(candidates_by_photo)}"
            )
            logger.info(
                f"[MATCH_STATS] filtered_by_score={stats['filtered_by_score']} filtered_by_margin={stats['filtered_by_margin']}"
            )

            # Log top 10 rejected con dettagli richiesti
            rejected.sort(key=lambda x: x["score"], reverse=True)
            if rejected:
                top10 = []
                for r in rejected[:10]:
                    margin_str = "None" if r["margin"] is None else f"{r['margin']:.3f}"
                    top10.append(
                        (
                            r["r2_key"],
                            f"score={r['score']:.3f}",
                            f"det={r['det_score']:.3f}",
                            f"area={r['area']}",
                            f"min_score={r['min_score']:.2f}",
                            f"margin={margin_str}",
                            f"margin_min={r['margin_min']:.3f}",
                            f"hits={r['hits']}",
                            f"bucket={r['bucket']}",
                            r["reason"],
                        )
                    )
                logger.info(f"[MATCH_REJECTED] Top 10: {top10}")

            filtered_results = results
            
        finally:
            # Ripristina meta_rows
            meta_rows = old_meta_rows
        
        all_results = filtered_results
        
        # Estrai photo_ids (filename) da all_results
        matched_photo_ids = []
        for r in all_results:
            photo_id = r.get("photo_id") or r.get("r2_key") or r.get("display_name")
            if photo_id:
                # Normalizza: rimuovi prefissi originals/, thumbs/, wm/
                normalized = photo_id.replace("originals/", "").replace("thumbs/", "").replace("wm/", "").lstrip("/")
                matched_photo_ids.append(normalized)
        
        # Log per debug
        sample = matched_photo_ids[0] if matched_photo_ids else "none"
        logger.info(f"[MATCH_RESPONSE] count={len(matched_photo_ids)} sample={sample}")
        
        # Se non ci sono risultati
        if len(all_results) == 0:
            return {
                "ok": True,
                "count": 0,
                "matched_photo_ids": [],
                "photo_ids": [],  # Alias per compatibilità
                "matches": [],
                "results": [],
                "matched_count": 0,
                "message": "Nessuna foto trovata.",
                "debug_reason": "Nessun match sopra la soglia minima"
            }
        
        return {
            "ok": True,
            "count": len(all_results),
            "matched_photo_ids": matched_photo_ids,  # Lista di filename trovati
            "photo_ids": matched_photo_ids,  # Alias identico per compatibilità
            "matches": all_results,
            "results": all_results,
            "matched_count": len(all_results),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in match_selfie: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# ========== FAMILY MEMBERS (DETERMINISTIC) ==========

async def _match_member_photos(selfie_embedding: np.ndarray, email: str, member_id: int, min_score: float = 0.25) -> List[str]:
    """
    Match foto per un membro famiglia usando SOLO primary_face_indices (nessun linked).
    Salva le foto trovate in user_photos con source_member_id.
    """
    global meta_rows, faiss_index
    _ensure_ready()
    
    if faiss_index is None or faiss_index.ntotal == 0 or len(meta_rows) == 0:
        logger.info("Index is empty - returning empty result for member")
        return []
    
    # Normalizza embedding
    selfie_embedding = _normalize(selfie_embedding).reshape(1, -1)
    
    # Cerca volti simili (stesso motore di /match_selfie)
    top_k_faces = 120
    D, I = faiss_index.search(selfie_embedding, min(top_k_faces, faiss_index.ntotal))
    
    # Identifica primary_face_indices (volti che matchano il selfie)
    primary_face_indices: Set[int] = set()
    photo_faces: Dict[str, Set[int]] = defaultdict(set)
    
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0 or idx >= len(meta_rows):
            continue
        if float(score) < min_score:
                continue
            
        primary_face_indices.add(idx)
        row = meta_rows[idx]
        photo_id = row.get("photo_id") or row.get("filename") or row.get("id")
        if photo_id:
            photo_faces[photo_id].add(idx)
    
    # Filtra foto: SOLO quelle che contengono almeno un volto primary
    # E NON contengono facce esterne (regola "no estranei")
    matched_photo_ids: List[str] = []
    
    for photo_id, face_indices in photo_faces.items():
        has_primary = bool(face_indices.intersection(primary_face_indices))
        if not has_primary:
                    continue
        
        # Verifica che non ci siano facce "estranee"
        external_faces = face_indices - primary_face_indices
        if external_faces:
            # Contiene facce estranee -> escludi
                continue
        
        # Foto valida: contiene almeno primary e nessuna faccia esterna
        matched_photo_ids.append(photo_id)
    
    # Salva foto trovate in user_photos con source_member_id
    if matched_photo_ids:
        email = _normalize_email(email)
        for photo_id in matched_photo_ids:
            try:
                await _db_execute_write("""
                    INSERT INTO user_photos (email, photo_id, found_at, status, source_member_id, expires_at)
                    VALUES ($1, $2, NOW(), 'found', $3, NOW() + INTERVAL '90 days')
                    ON CONFLICT (email, photo_id) DO UPDATE
                    SET source_member_id = COALESCE(EXCLUDED.source_member_id, user_photos.source_member_id)
                """, (email, photo_id, member_id))
            except Exception as e:
                logger.error(f"Error saving member photo {photo_id} for member {member_id}: {e}")
    
    logger.info(f"Member {member_id} matched {len(matched_photo_ids)} photos")
    return matched_photo_ids

@app.post("/family/add_member")
async def add_family_member_disabled(
    email: str = Query(..., description="Email utente"),
    member_name: Optional[str] = Query(None, description="Nome membro (opzionale)"),
    selfie: UploadFile = File(..., description="Selfie del membro")
):
    """Aggiunge un membro famiglia (disabilitato in stateless mode)"""
    raise HTTPException(status_code=501, detail="Family members disabled in stateless mode")
    try:
        email = _normalize_email(email)
        
        # Verifica/crea utente
        try:
            await _db_execute_write("""
                INSERT INTO users (email, created_at)
                VALUES ($1, NOW())
                ON CONFLICT (email) DO NOTHING
            """, (email,))
        except Exception as e:
            logger.warning(f"Could not create user {email}: {e}")
        
        # Conta membri esistenti
        count_row = await _db_execute_one(
            "SELECT COUNT(*) as count FROM family_members WHERE email = $1",
            (email,)
        )
        member_count = count_row['count'] if count_row else 0
        
        if member_count >= MAX_FAMILY_MEMBERS:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {MAX_FAMILY_MEMBERS} family members allowed per email. You have {member_count} members."
            )
        
        # Leggi e processa selfie
        file_bytes = await selfie.read()
        img = _read_image_from_bytes(file_bytes)
        
        # Rileva faccia e estrai embedding
        assert face_app is not None
        faces = face_app.get(img)
        
        if not faces:
            raise HTTPException(status_code=400, detail="No face detected in selfie")
        
        # Prendi il volto più grande
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )
        member_embedding = faces_sorted[0].embedding.astype("float32")
        
        # Converti embedding in bytes per salvare nel database
        embedding_bytes = member_embedding.tobytes()
        
        # Salva membro nel database
        member_row = await _db_execute_one("""
            INSERT INTO family_members (email, member_name, selfie_embedding, created_at)
            VALUES ($1, $2, $3, NOW())
            RETURNING id
        """, (email, member_name, embedding_bytes))
        
        member_id = member_row['id']
        logger.info(f"Created family member {member_id} for {email}")
        
        # Match foto usando SOLO primary_face_indices (nessun linked)
        matched_photo_ids = await _match_member_photos(member_embedding, email, member_id)
        
        return {
            "ok": True,
            "member_id": member_id,
            "matched_count": len(matched_photo_ids),
            "photo_ids": matched_photo_ids
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding family member: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/family/members")
async def get_family_members_disabled(
    email: str = Query(..., description="Email utente")
):
    """Recupera lista membri famiglia (stateless: sempre vuoto)"""
    try:
        email = _normalize_email(email)
        
        # Stateless mode: sempre lista vuota
        return {
            "ok": True,
            "email": email,
            "members": [],
            "count": 0
        }
    except Exception as e:
        logger.error(f"Error getting family members: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.delete("/family/members/{member_id}")
async def delete_family_member_disabled(
    member_id: int,
    email: str = Query(..., description="Email utente")
):
    """Elimina un membro famiglia (disabilitato in stateless mode)"""
    raise HTTPException(status_code=501, detail="Family members disabled in stateless mode")
    try:
        email = _normalize_email(email)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting family member: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ========== CARRELLO E PAGAMENTI ==========

# Storage carrelli su database PostgreSQL
# Il carrello è persistente per session_id nella tabella carts

async def _get_cart(session_id: str) -> List[str]:
    """Ottiene il carrello per una sessione dal database"""
    try:
        row = await _db_execute_one(
            "SELECT photo_ids FROM carts WHERE session_id = $1",
            (session_id,)
        )
        
        if row and row.get('photo_ids'):
            photo_ids = json.loads(row['photo_ids'])
            return photo_ids if isinstance(photo_ids, list) else []
        return []
    except Exception as e:
        logger.error(f"Error getting cart for session {session_id}: {e}", exc_info=True)
        return []

async def _set_cart(session_id: str, photo_ids: List[str]) -> List[str]:
    """Imposta il carrello completo per una sessione (upsert) e restituisce la lista aggiornata"""
    try:
        # Dedup preservando ordine
        seen = set()
        unique_photo_ids = []
        for photo_id in photo_ids:
            if photo_id not in seen:
                seen.add(photo_id)
                unique_photo_ids.append(photo_id)
        
        photo_ids_json = json.dumps(unique_photo_ids, ensure_ascii=False)
        
        # Usa RETURNING per ottenere direttamente il risultato senza query separata
        query = """
            INSERT INTO carts (session_id, photo_ids, created_at, updated_at)
            VALUES ($1, $2::jsonb, NOW(), NOW())
            ON CONFLICT (session_id) DO UPDATE
            SET photo_ids = EXCLUDED.photo_ids, updated_at = NOW()
            RETURNING photo_ids
        """
        
        row = await _db_execute_one(query, (session_id, photo_ids_json))
        
        if row and row.get('photo_ids'):
            # Parse JSON se è stringa, altrimenti usa direttamente
            result = row['photo_ids']
            if isinstance(result, str):
                return json.loads(result)
            return result if isinstance(result, list) else []
        
        # Fallback: restituisci quello che abbiamo passato
        return unique_photo_ids
    except Exception as e:
        logger.error(f"Error setting cart for session {session_id}: {e}", exc_info=True)
        # Fallback: restituisci lista vuota in caso di errore
        return []

async def _add_to_cart(session_id: str, photo_id: str) -> List[str]:
    """Aggiunge foto al carrello e restituisce la lista aggiornata"""
    current_photo_ids = await _get_cart(session_id)
    if photo_id not in current_photo_ids:
        current_photo_ids.append(photo_id)
        updated = await _set_cart(session_id, current_photo_ids)
        logger.info(f"[CART] add session={session_id} photo={photo_id} size={len(updated)}")
        return updated
    # Foto già presente, restituisci carrello corrente
    return current_photo_ids

async def _remove_from_cart(session_id: str, photo_id: str) -> List[str]:
    """Rimuove foto dal carrello e restituisce la lista aggiornata"""
    current_photo_ids = await _get_cart(session_id)
    if photo_id in current_photo_ids:
        current_photo_ids = [p for p in current_photo_ids if p != photo_id]
        updated = await _set_cart(session_id, current_photo_ids)
        logger.info(f"[CART] remove session={session_id} photo={photo_id} size={len(updated)}")
        return updated
    # Foto non presente, restituisci carrello corrente
    return current_photo_ids

async def _clear_cart(session_id: str):
    """Svuota il carrello"""
    logger.info(f"_clear_cart: Starting for session_id={session_id}")
    try:
        query = "DELETE FROM carts WHERE session_id = $1"
        logger.info(f"_clear_cart: Executing PostgreSQL DELETE query")
        await _db_execute_write(query, (session_id,))
        logger.info(f"_clear_cart: Cart cleared successfully for session {session_id}")
    except Exception as e:
        logger.error(f"Error clearing cart for session {session_id}: {e}", exc_info=True)

@app.get("/cart")
async def get_cart(session_id: str = Query(..., description="ID sessione")):
    """Ottiene il contenuto del carrello - SEMPRE VUOTO (carrello non persistente tra sessioni)"""
    # Carrello sempre vuoto - non persiste tra sessioni
    # Il frontend genera sempre un nuovo sessionId, quindi non ci saranno mai carrelli vecchi
    photo_ids = []
    
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
    photo_id: str = Query(..., description="ID foto da aggiungere"),
    email: Optional[str] = Query(None, description="Email utente per verificare se foto è già pagata")
):
    """Aggiunge una foto al carrello - previene aggiunta di foto già pagate"""
    # Verifica se la foto è già pagata (se email fornita)
    if email:
        try:
            paid_photos = await _get_user_paid_photos(email)
            if photo_id in paid_photos:
                logger.warning(f"Attempt to add already paid photo to cart: {photo_id} for {email}")
                current_photo_ids = await _get_cart(session_id)
                return {
                    "ok": False,
                    "error": "Questa foto è già stata acquistata",
                    "photo_ids": current_photo_ids,
                    "count": len(current_photo_ids),
                    "price_cents": 0,
                    "price_euros": 0.0
                }
        except Exception as e:
            logger.error(f"Error checking paid photos in cart/add: {e}")
    
    # Usa il valore di ritorno di _add_to_cart invece di chiamare _get_cart dopo
    photo_ids = await _add_to_cart(session_id, photo_id)
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
    # Usa il valore di ritorno di _remove_from_cart invece di chiamare _get_cart dopo
    photo_ids = await _remove_from_cart(session_id, photo_id)
    price = calculate_price(len(photo_ids)) if photo_ids else 0
    
    return {
        "ok": True,
        "photo_ids": photo_ids,
        "count": len(photo_ids),
        "price_cents": price,
        "price_euros": price / 100.0
    }

@app.post("/create_checkout")
async def create_checkout_new(
    request: Request,
    body: dict = Body(...)
):
    """Crea una sessione di checkout Stripe basata su photo_ids (senza email)"""
    logger.info(f"=== CREATE_CHECKOUT REQUEST ===")
    
    photo_ids = body.get("photo_ids", [])
    price_cents = body.get("price_cents")
    currency = body.get("currency", "eur")
    
    if not photo_ids:
        logger.error("photo_ids is empty")
        raise HTTPException(status_code=400, detail="photo_ids is required and cannot be empty")
    
    # Calcola prezzo se non fornito
    if not price_cents:
        price_cents = calculate_price(len(photo_ids))
    
    # TEST MODE: simula Stripe senza chiamate reali
    if STRIPE_TEST_MODE:
        import time
        fake_session = f"test_{int(time.time())}"
        # Salva photo_ids per questo fake session_id (per /stripe/verify)
        test_sessions[fake_session] = {
            "photo_ids": photo_ids,
            "customer_email": None,  # In test mode non abbiamo email
            "payment_status": "paid",
            "amount_total": price_cents,
            "currency": currency
        }
        logger.info(f"[TEST_MODE] create_checkout -> saved session {fake_session} with {len(photo_ids)} photo_ids")
        logger.info(f"[TEST_MODE] Photo IDs: {photo_ids[:5] if len(photo_ids) > 5 else photo_ids}...")
        logger.info(f"[TEST_MODE] Total test sessions in memory: {len(test_sessions)}")
        # In test mode, reindirizza alla pagina success invece della home
        redirect_url = f"{PUBLIC_BASE_URL}/checkout/success?session_id={fake_session}"
        logger.info(f"[TEST_MODE] create_checkout -> redirecting to: {redirect_url}")
        return {
            "url": redirect_url,
            "mode": "test",
            "session_id": fake_session
        }
    
    # PRODUZIONE: Stripe reale
    if not USE_STRIPE:
        logger.error("Stripe not configured")
        raise HTTPException(status_code=503, detail="Stripe not configured")
    
    logger.info(f"Creating checkout for {len(photo_ids)} photos, price: {price_cents} cents")
    
    try:
        # Costruisci URL base
        base_url = str(request.base_url).rstrip('/')
        logger.info(f"Base URL: {base_url}")
        
        logger.info("Creating Stripe checkout session...")
        # Crea checkout session Stripe
        # Stripe chiederà l'email al cliente
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': currency,
                    'product_data': {
                        'name': f'{len(photo_ids)} foto da TenerifePictures',
                        'description': f'Download di {len(photo_ids)} foto in alta qualità',
                    },
                    'unit_amount': price_cents,
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=f'{base_url}/checkout/success?session_id={{CHECKOUT_SESSION_ID}}',
            cancel_url=f'{base_url}/',
            metadata=_build_checkout_metadata(None, None, photo_ids)
        )
        
        logger.info(f"Stripe checkout session created: {checkout_session.id}")
        logger.info(f"Checkout URL: {checkout_session.url}")
        
        return {
            "ok": True,
            "checkout_url": checkout_session.url,
            "session_id": checkout_session.id
        }
    except Exception as e:
        logger.error(f"Error creating Stripe checkout: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Checkout error: {str(e)}")

@app.get("/stripe/verify")
async def verify_stripe_session(
    session_id: str = Query(..., description="Stripe session ID")
):
    """Verifica sessione Stripe pagata e restituisce photo_ids (stateless)"""
    
    if not STRIPE_AVAILABLE or not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Stripe not configured")
    
    # IMPORTANTE: Anche in test mode, provare PRIMA a recuperare la sessione reale da Stripe
    # Questo permette di avere l'email anche quando si usa la carta test 4242
    # Solo se Stripe non ha la sessione, usare il fallback alla sessione test fake
    try:
        import stripe
        stripe.api_key = STRIPE_SECRET_KEY
        
        # Prova a recuperare la sessione da Stripe (funziona anche per sessioni test)
        try:
            session = stripe.checkout.Session.retrieve(session_id)
            logger.info(f"[STRIPE_VERIFY] Found session in Stripe: {session_id}, payment_status: {session.payment_status}")
            
            # Verifica che il pagamento sia completato
            if session.payment_status != 'paid':
                raise HTTPException(status_code=400, detail=f"Payment not completed. Status: {session.payment_status}")
            
            # Estrai photo_ids da metadata (gestisci token se presente)
            metadata = session.metadata if session.metadata else {}
            logger.info(f"[STRIPE_VERIFY] Session metadata: {metadata}")
            photo_ids_token = metadata.get('photo_ids_token')
            photo_ids_str = metadata.get('photo_ids', '')
            logger.info(f"[STRIPE_VERIFY] photo_ids_token: {photo_ids_token[:20] if photo_ids_token else None}..., photo_ids_str length: {len(photo_ids_str)}")
            
            if photo_ids_token:
                # Recupera photo_ids dal dizionario in memoria
                logger.info(f"[STRIPE_VERIFY] Looking for token in checkout_photo_ids (available tokens: {list(checkout_photo_ids.keys())[:5]})")
                if photo_ids_token in checkout_photo_ids:
                    photo_ids = checkout_photo_ids[photo_ids_token]
                    logger.info(f"[STRIPE_VERIFY] Photo IDs recovered from token: {len(photo_ids)} photos")
                else:
                    logger.warning(f"[STRIPE_VERIFY] Token not found in checkout_photo_ids (server restart?). Trying to recover from database...")
                    # FALLBACK: Se il token non è in memoria, prova a recuperare dal database
                    try:
                        if USE_POSTGRES:
                            # Cerca ordine con questo session_id
                            row = await _db_execute_one(
                                "SELECT photo_ids FROM orders WHERE stripe_session_id = $1",
                                (session_id,)
                            )
                            if row and row.get('photo_ids'):
                                photo_ids = json.loads(row['photo_ids']) if isinstance(row['photo_ids'], str) else row['photo_ids']
                                logger.info(f"[STRIPE_VERIFY] Recovered {len(photo_ids)} photo_ids from database")
                            else:
                                logger.error(f"[STRIPE_VERIFY] No order found in database for session_id: {session_id}")
                                photo_ids = []
                        else:
                            logger.warning(f"[STRIPE_VERIFY] Database not available, cannot recover photo_ids")
                            photo_ids = []
                    except Exception as e:
                        logger.error(f"[STRIPE_VERIFY] Error recovering photo_ids from database: {e}")
                        photo_ids = []
            elif photo_ids_str:
                # Usa lista diretta dai metadata (retrocompatibilità)
                photo_ids = [pid.strip() for pid in photo_ids_str.split(',') if pid.strip()]
                logger.info(f"[STRIPE_VERIFY] Photo IDs from metadata string: {len(photo_ids)} photos")
            else:
                logger.warning(f"[STRIPE_VERIFY] No photo_ids_token or photo_ids_str in metadata")
                photo_ids = []
            
            if not photo_ids:
                logger.error(f"[STRIPE_VERIFY] No photo_ids found! metadata={metadata}, token={photo_ids_token[:20] if photo_ids_token else None}")
                raise HTTPException(status_code=400, detail="No photo_ids found in session metadata or token")
            
            # Estrai email da customer_details o metadata
            customer_email = None
            if session.customer_details and session.customer_details.email:
                customer_email = session.customer_details.email
            elif session.metadata and session.metadata.get('email'):
                customer_email = session.metadata.get('email')
            
            logger.info(f"[STRIPE_VERIFY] Session verified: {len(photo_ids)} photos for {customer_email}")
            
            # IMPORTANTE: Salva le foto come pagate nel database (per velocità: verifica immediata in /photo)
            if customer_email and photo_ids:
                logger.info(f"[STRIPE_VERIFY] Marking {len(photo_ids)} photos as paid for {customer_email}")
                for photo_id in photo_ids:
                    try:
                        await _mark_photo_paid(customer_email, photo_id)
                    except Exception as e:
                        logger.error(f"[STRIPE_VERIFY] Error marking photo as paid: {photo_id} - {e}")
                        # Continua anche se una foto fallisce
            
            return {
                "ok": True,
                "session_id": session_id,
                "customer_email": customer_email,
                "photo_ids": photo_ids,
                "payment_status": session.payment_status,
                "amount_total": session.get('amount_total', 0),
                "currency": session.get('currency', 'eur')
            }
        except stripe.error.InvalidRequestError as e:
            # Sessione non trovata in Stripe, usa fallback test mode se disponibile
            if STRIPE_TEST_MODE and session_id.startswith("test_"):
                logger.info(f"[STRIPE_VERIFY] Session not found in Stripe, trying test mode fallback: {session_id}")
                if session_id in test_sessions:
                    session_data = test_sessions[session_id]
                    photo_ids = session_data.get("photo_ids", [])
                    customer_email = session_data.get("customer_email")
                    
                    logger.info(f"[TEST_MODE] Found session in test_sessions: {session_id}, photo_ids count: {len(photo_ids)}")
                    
                    if not photo_ids:
                        logger.warning(f"[TEST_MODE] Session {session_id} has no photo_ids")
                        raise HTTPException(status_code=400, detail="No photo_ids found in test session")
                    
                    # IMPORTANTE: Salva le foto come pagate nel database (per velocità: verifica immediata in /photo)
                    if customer_email and photo_ids:
                        logger.info(f"[TEST_MODE] Marking {len(photo_ids)} photos as paid for {customer_email}")
                        for photo_id in photo_ids:
                            try:
                                await _mark_photo_paid(customer_email, photo_id)
                            except Exception as e:
                                logger.error(f"[TEST_MODE] Error marking photo as paid: {photo_id} - {e}")
                                # Continua anche se una foto fallisce
                    
                    logger.info(f"[TEST_MODE] verify_stripe_session -> {session_id}, returning {len(photo_ids)} photo_ids")
                    return {
                        "ok": True,
                        "session_id": session_id,
                        "customer_email": customer_email,
                        "photo_ids": photo_ids,
                        "payment_status": session_data.get("payment_status", "paid"),
                        "amount_total": session_data.get("amount_total", 0),
                        "currency": session_data.get("currency", "eur")
                    }
                else:
                    logger.warning(f"[TEST_MODE] Session not found in test_sessions: {session_id}")
                    raise HTTPException(status_code=404, detail=f"Session not found in Stripe or test mode: {session_id}")
            else:
                # Non è test mode o non inizia con "test_", rilanciare l'errore
                raise HTTPException(status_code=404, detail=f"Session not found: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying Stripe session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/checkout/status")
async def api_checkout_status(
    session_id: Optional[str] = Query(None, description="Stripe session ID (opzionale se purchase_token fornito)"),
    purchase_token: Optional[str] = Query(None, description="Purchase token salvato in localStorage (opzionale)")
):
    """
    Endpoint per ottenere lo stato del checkout e le foto sbloccate.
    Ritorna purchased_photo_ids, unlocked_urls (full quality), e remaining_photo_ids.
    
    Args:
        session_id: Stripe session ID (se fornito, verifica pagamento)
        purchase_token: Token di acquisto salvato in localStorage (se fornito, recupera da DB)
    
    Returns:
        {
            "ok": true,
            "purchased_photo_ids": ["photo1.jpg", "photo2.jpg"],
            "unlocked_urls": {
                "photo1.jpg": "https://r2.../originals/photo1.jpg",
                "photo2.jpg": "https://r2.../originals/photo2.jpg"
            },
            "remaining_photo_ids": ["photo3.jpg", "photo4.jpg"],  # Se esistono
            "customer_email": "user@example.com"
        }
    """
    from urllib.parse import quote
    
    logger.info(f"[API_CHECKOUT_STATUS] Called with session_id={session_id}, purchase_token={'***' if purchase_token else None}")
    
    try:
        purchased_photo_ids = []
        customer_email = None
        
        # Verifica che almeno uno dei parametri sia fornito
        if not session_id and not purchase_token:
            logger.warning("[API_CHECKOUT_STATUS] Missing both session_id and purchase_token")
            raise HTTPException(status_code=400, detail="Either session_id or purchase_token must be provided")
        
        # Se purchase_token fornito, recupera da DB
        if purchase_token:
            logger.info(f"[API_CHECKOUT_STATUS] Looking up order by purchase_token...")
            try:
                if not USE_POSTGRES:
                    logger.warning("[API_CHECKOUT_STATUS] Database not available (USE_POSTGRES=False), cannot lookup by purchase_token")
                    raise HTTPException(status_code=503, detail="Database not available")
                
                row = await _db_execute_one(
                    "SELECT photo_ids, email FROM orders WHERE download_token = $1 OR stripe_session_id = $1",
                    (purchase_token,)
                )
                if row:
                    purchased_photo_ids = json.loads(row['photo_ids']) if row['photo_ids'] else []
                    customer_email = row['email']
                    logger.info(f"[API_CHECKOUT_STATUS] Found order by purchase_token: {len(purchased_photo_ids)} photos for {customer_email}")
                else:
                    logger.warning(f"[API_CHECKOUT_STATUS] No order found for purchase_token")
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"[API_CHECKOUT_STATUS] Error getting order by purchase_token: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        
        # Se session_id fornito, verifica con Stripe
        if session_id and not purchased_photo_ids:
            logger.info(f"[API_CHECKOUT_STATUS] Verifying session_id with Stripe...")
            try:
                verify_result = await verify_stripe_session(session_id)
                if verify_result.get("ok"):
                    purchased_photo_ids = verify_result.get("photo_ids", [])
                    customer_email = verify_result.get("customer_email")
                    logger.info(f"[API_CHECKOUT_STATUS] Verified session_id: {len(purchased_photo_ids)} photos for {customer_email}")
                else:
                    logger.warning(f"[API_CHECKOUT_STATUS] Stripe verification returned !ok: {verify_result}")
            except HTTPException as e:
                logger.error(f"[API_CHECKOUT_STATUS] HTTPException from verify_stripe_session: {e.status_code} - {e.detail}")
                raise
            except Exception as e:
                logger.error(f"[API_CHECKOUT_STATUS] Error verifying session_id: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Stripe verification error: {str(e)}")
        
        if not purchased_photo_ids:
            logger.warning(f"[API_CHECKOUT_STATUS] No purchased photos found (session_id={session_id}, purchase_token={'***' if purchase_token else None})")
            raise HTTPException(status_code=404, detail="No purchased photos found")
        
        # IMPORTANTE: Salva le foto come pagate PRIMA di costruire gli URL
        # Questo permette a /photo?paid=true&email=... di verificare correttamente il pagamento
        if customer_email and purchased_photo_ids:
            logger.info(f"[API_CHECKOUT_STATUS] Marking {len(purchased_photo_ids)} photos as paid for {customer_email}")
            for photo_id in purchased_photo_ids:
                try:
                    await _mark_photo_paid(customer_email, photo_id)
                except Exception as e:
                    logger.error(f"[API_CHECKOUT_STATUS] Error marking photo as paid: {photo_id} - {e}")
                    # Continua anche se una foto fallisce
        
        # Costruisci unlocked_urls (full quality, no watermark) per ogni foto acquistata
        # NOTA: Usiamo sempre /photo endpoint invece di URL diretti R2 per garantire verifica pagamento
        # e gestire correttamente CORS/permessi
        unlocked_urls = {}
        for photo_id in purchased_photo_ids:
            # Normalizza photo_id (rimuovi prefissi)
            normalized_id = photo_id.lstrip('/').replace('originals/', '').replace('thumbs/', '').replace('wm/', '')
            # Usa sempre endpoint /photo con email per verifica pagamento server-side
            if customer_email:
                photo_url = f"/photo/{quote(normalized_id, safe='')}?paid=true&email={quote(customer_email, safe='')}"
            else:
                photo_url = f"/photo/{quote(normalized_id, safe='')}?paid=true"
            unlocked_urls[photo_id] = photo_url
            logger.info(f"[API_CHECKOUT_STATUS] Built URL for {photo_id}: {photo_url[:100]}...")
        
        # Calcola remaining_photo_ids (tutte le foto disponibili meno quelle acquistate)
        remaining_photo_ids = []
        try:
            # Ottieni tutte le foto disponibili da R2
            all_available_photos = set()
            
            if USE_R2 and r2_client:
                try:
                    # Lista tutte le foto originali da R2
                    paginator = r2_client.get_paginator('list_objects_v2')
                    prefix = R2_PHOTOS_PREFIX if R2_PHOTOS_PREFIX else ""
                    
                    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=prefix):
                        for obj in page.get('Contents', []):
                            key = obj['Key']
                            # Filtra solo originals (non thumbs, wm, faces, ecc.)
                            if key.startswith("originals/") and not key.startswith("faces."):
                                # Estrai filename (rimuovi prefisso originals/)
                                filename = key.replace("originals/", "").lstrip('/')
                                if filename:
                                    all_available_photos.add(filename)
                    
                    logger.info(f"[API_CHECKOUT_STATUS] Found {len(all_available_photos)} available photos from R2")
                except Exception as e:
                    logger.warning(f"[API_CHECKOUT_STATUS] Error listing R2 photos: {e}")
            
            # Se non abbiamo foto da R2, prova dal database (meta_rows)
            if not all_available_photos:
                try:
                    # Carica meta_rows se disponibili
                    if 'meta_rows' in globals() and meta_rows:
                        for row in meta_rows:
                            photo_id = row.get('photo_id') or row.get('r2_key', '').replace('originals/', '')
                            if photo_id:
                                all_available_photos.add(photo_id)
                        logger.info(f"[API_CHECKOUT_STATUS] Found {len(all_available_photos)} available photos from meta_rows")
                except Exception as e:
                    logger.warning(f"[API_CHECKOUT_STATUS] Error getting photos from meta_rows: {e}")
            
            # Calcola remaining: tutte le foto disponibili meno quelle acquistate
            purchased_set = set(purchased_photo_ids)
            remaining_photo_ids = [pid for pid in all_available_photos if pid not in purchased_set]
            
            logger.info(f"[API_CHECKOUT_STATUS] Remaining photos: {len(remaining_photo_ids)}")
        except Exception as e:
            logger.warning(f"[API_CHECKOUT_STATUS] Error calculating remaining photos: {e}")
            # Se errore, non includere remaining_photo_ids (non critico)
            remaining_photo_ids = []
        
        return {
            "ok": True,
            "purchased_photo_ids": purchased_photo_ids,
            "unlocked_urls": unlocked_urls,
            "remaining_photo_ids": remaining_photo_ids,
            "customer_email": customer_email
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API_CHECKOUT_STATUS] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting checkout status: {str(e)}")

@app.post("/api/photos")
async def api_photos_post(
    request: Request,
    body: dict = Body(...)
):
    """
    Endpoint POST per ottenere URL delle foto (thumb/wm/full).
    Evita problemi con URL troppo lunghe in GET (Safari/Cloudflare limit).
    
    Body accetta MULTIPLI FORMATI:
    1) { "variant": "thumb", "photo_ids": ["a.jpg", ...] }
    2) { "variant": "thumb", "ids": ["a.jpg", ...] }
    3) ["a.jpg", ...]  (array diretto, variant default "thumb")
    4) GET query: /api/photos?ids=a.jpg&ids=b.jpg (opzionale)
    
    Returns: {
        "ok": true,
        "items": [  # o "photos" per retrocompatibilità
            {
                "id": "photo1.jpg",
                "photo_id": "photo1.jpg",  # retrocompatibilità
                "thumb_url": "...",  # SEMPRE presente
                "wm_url": "...",    # SEMPRE presente
                "direct_thumb_url": "...",
                "direct_wm_url": "..."
            },
            ...
        ]
    }
    """
    from urllib.parse import quote, unquote
    
    try:
        # DIAGNOSI: log completo della richiesta
        logger.info(f"[API_PHOTOS_POST] === REQUEST DIAGNOSTICS ===")
        logger.info(f"[API_PHOTOS_POST] Method: {request.method}")
        logger.info(f"[API_PHOTOS_POST] Content-Type: {request.headers.get('content-type', 'missing')}")
        logger.info(f"[API_PHOTOS_POST] Body keys: {list(body.keys()) if isinstance(body, dict) else 'not a dict'}")
        logger.info(f"[API_PHOTOS_POST] Body preview: {str(body)[:500]}")
        
        # Estrai variant (default "thumb")
        variant = body.get("variant", "thumb") if isinstance(body, dict) else "thumb"
        
        # Estrai photo_ids da MULTIPLI formati
        photo_ids = []
        if isinstance(body, list):
            # Formato 3: array diretto
            photo_ids = body
            logger.info(f"[API_PHOTOS_POST] Parsed: array format, {len(photo_ids)} items")
        elif isinstance(body, dict):
            # Formato 1 o 2: dict con photo_ids o ids
            photo_ids = body.get("ids") or body.get("photo_ids") or []
            if not isinstance(photo_ids, list):
                # Se è una stringa singola, converti in lista
                if isinstance(photo_ids, str):
                    photo_ids = [photo_ids]
                else:
                    photo_ids = []
            logger.info(f"[API_PHOTOS_POST] Parsed: dict format, variant={variant}, ids={len(photo_ids)}")
        else:
            logger.warning(f"[API_PHOTOS_POST] Unexpected body type: {type(body)}")
            photo_ids = []
        
        # Filtra null/undefined/empty strings
        photo_ids = [pid for pid in photo_ids if pid and isinstance(pid, str) and pid.strip()]
        
        # IMPORTANTE: se lista vuota, restituisci 200 con lista vuota (non 400)
        if len(photo_ids) == 0:
            logger.info(f"[API_PHOTOS_POST] Empty photo_ids list - returning empty response")
            return {
                "ok": True,
                "items": [],
                "photos": []  # retrocompatibilità
            }
        
        # Valida variant
        if variant not in ["thumb", "wm", "full"]:
            variant = "thumb"  # fallback invece di errore
            logger.warning(f"[API_PHOTOS_POST] Invalid variant, using 'thumb'")
        
        # Chunk se > 100 per evitare payload troppo grandi
        if len(photo_ids) > 100:
            logger.info(f"[API_PHOTOS_POST] Large request ({len(photo_ids)} ids), processing in chunks")
            # Per ora processiamo tutto, ma potremmo chunkare se necessario
        
        logger.info(f"[API_PHOTOS_POST] Processing {len(photo_ids)} photo_ids, variant={variant}")
        
        items = []
        for idx, photo_id in enumerate(photo_ids):
            try:
                # Normalizza photo_id: trim spazi, rimuovi prefissi, gestisci encoding
                original_id = str(photo_id).strip()
                # Decodifica se è già URL encoded
                try:
                    original_id = unquote(original_id)
                except:
                    pass
                
                # Rimuovi prefissi comuni
                normalized_id = original_id.lstrip('/')
                for prefix in ['originals/', 'thumbs/', 'wm/', 'photos/']:
                    if normalized_id.startswith(prefix):
                        normalized_id = normalized_id[len(prefix):]
                
                # Normalizza estensioni (.JPG -> .jpg, ma conserva il nome originale)
                # Non cambiamo l'estensione, solo la normalizziamo per il path
                normalized_id = normalized_id.strip()
                
                if not normalized_id:
                    logger.warning(f"[API_PHOTOS_POST] Skipping empty normalized_id for photo_id={photo_id}")
                    continue
                
                # Costruisci object_key per R2
                thumb_object_key = f"thumbs/{normalized_id}"
                wm_object_key = f"wm/{normalized_id}"
                
                # Costruisci URL relativi (per fallback)
                # IMPORTANTE: usa encodeURIComponent per gestire spazi e caratteri speciali
                encoded_id = quote(normalized_id, safe='')
                thumb_url = f"/photo/{encoded_id}?variant=thumb"
                wm_url = f"/photo/{encoded_id}?variant=wm"
                
                # Costruisci direct_url se R2_PUBLIC_BASE_URL disponibile
                direct_thumb_url = None
                direct_wm_url = None
                
                if R2_PUBLIC_BASE_URL:
                    try:
                        direct_thumb_url = _get_r2_public_url(thumb_object_key)
                        direct_wm_url = _get_r2_public_url(wm_object_key)
                    except Exception as e:
                        logger.warning(f"[API_PHOTOS_POST] Error building direct URLs for {normalized_id}: {e}")
                        # Fallback: usa URL relativi
                        direct_thumb_url = thumb_url
                        direct_wm_url = wm_url
                else:
                    # Se R2_PUBLIC_BASE_URL non configurato, usa URL relativi
                    direct_thumb_url = thumb_url
                    direct_wm_url = wm_url
                
                # Restituisci SEMPRE sia thumb_url che wm_url (indipendentemente da variant)
                item = {
                    "id": normalized_id,  # nuovo formato
                    "photo_id": normalized_id,  # retrocompatibilità
                    "filename": normalized_id,  # retrocompatibilità
                    "thumb_url": direct_thumb_url or thumb_url,
                    "wm_url": direct_wm_url or wm_url,
                    "direct_thumb_url": direct_thumb_url,
                    "direct_wm_url": direct_wm_url,
                    "url": thumb_url if variant == "thumb" else wm_url,  # retrocompatibilità
                    "direct_url": direct_thumb_url if variant == "thumb" else direct_wm_url,  # retrocompatibilità
                    "variant": variant  # retrocompatibilità
                }
                items.append(item)
                
            except Exception as e:
                logger.warning(f"[API_PHOTOS_POST] Error processing photo_id {photo_id} (index {idx}): {e}", exc_info=True)
                # Continua con le altre foto anche se una fallisce
                continue
        
        logger.info(f"[API_PHOTOS_POST] Success: returning {len(items)} items")
        return {
            "ok": True,
            "items": items,  # nuovo formato
            "photos": items  # retrocompatibilità
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API_PHOTOS_POST] Fatal error: {e}", exc_info=True)
        # Restituisci errore con dettaglio utile invece di 500 generico
        raise HTTPException(
            status_code=422 if "validation" in str(e).lower() else 500,
            detail=f"Error getting photos: {str(e)}"
        )

@app.post("/checkout")
async def create_checkout(
    request: Request,
    session_id: str = Query(..., description="ID sessione"),
    email: Optional[str] = Query(None, description="Email utente (obbligatoria per salvare ordine)")
):
    """Crea una sessione di checkout Stripe (legacy, usa /create_checkout)"""
    logger.info(f"=== CHECKOUT REQUEST (LEGACY) ===")
    logger.info(f"Session ID: {session_id}")
    logger.info(f"Email: {email}")
    logger.info(f"USE_STRIPE: {USE_STRIPE}")
    
    if not USE_STRIPE:
        logger.error("Stripe not configured")
        raise HTTPException(status_code=503, detail="Stripe not configured")
    
    photo_ids = await _get_cart(session_id)
    logger.info(f"Cart photo_ids: {photo_ids}")
    
    if not photo_ids:
        logger.error("Cart is empty")
        raise HTTPException(status_code=400, detail="Cart is empty")
    
    # Email non più obbligatoria: Stripe la chiederà
    price_cents = calculate_price(len(photo_ids))
    logger.info(f"Price calculated: {price_cents} cents ({price_cents/100} EUR) for {len(photo_ids)} photos")
    
    try:
        # Costruisci URL base
        base_url = str(request.base_url).rstrip('/')
        logger.info(f"Base URL: {base_url}")
        
        logger.info("Creating Stripe checkout session...")
        # Crea checkout session Stripe
        # Nota: Per disabilitare Stripe Link, devi disabilitarlo nella Dashboard Stripe:
        # Dashboard → Settings → Payment methods → Link → Disable
        # Il parametro payment_method_options[link][enabled] non è supportato da questa versione dell'API
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            # Non includere customer_email per ridurre probabilità che Stripe mostri Link
            # customer_email=email,  # Commentato per ridurre probabilità che Stripe mostri Link
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
            # customer_email rimosso: Stripe Link usa l'email per autofill, rimuovendola riduciamo la probabilità che Link appaia
            # L'email viene comunque salvata nel metadata per il webhook
            success_url=f'{base_url}/checkout/success?session_id={{CHECKOUT_SESSION_ID}}&cart_session={session_id}',
            cancel_url=f'{base_url}/checkout/cancel?session_id={session_id}',
            metadata=_build_checkout_metadata(session_id, email, photo_ids)
        )
        
        logger.info(f"Stripe checkout session created: {checkout_session.id}")
        logger.info(f"Checkout URL: {checkout_session.url}")
        
        return {
            "ok": True,
            "checkout_url": checkout_session.url,
            "session_id": checkout_session.id
        }
    except Exception as e:
        logger.error(f"Error creating Stripe checkout: {e}", exc_info=True)
        logger.error(f"Exception type: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=f"Checkout error: {str(e)}")

@app.get("/checkout/success")
async def checkout_success(
    request: Request,
    session_id: Optional[str] = Query(None, description="Stripe session ID (opzionale)"),
    purchase_token: Optional[str] = Query(None, description="Purchase token (opzionale)"),
    cart_session: Optional[str] = Query(None, description="Cart session ID (optional)")
):
    """Pagina di successo dopo pagamento - serve pagina separata success.html"""
    logger.info(f"[CHECKOUT_SUCCESS] Request: session_id={session_id}, purchase_token={purchase_token}, cart_session={cart_session}")
    # Servi la pagina success.html con i parametri nella query string
    # La pagina stessa chiamerà /api/checkout/status per ottenere le foto
    success_html_path = STATIC_DIR / "success.html"
    if success_html_path.exists():
        logger.info(f"[CHECKOUT_SUCCESS] Serving success.html from: {success_html_path.resolve()}")
        return FileResponse(success_html_path)
    else:
        # Fallback: genera HTML inline (compatibilità)
        try:
            download_token = None
            order_data = {}
            photo_ids = []
            email = None
            
            # Il session_id è lo Stripe session ID (order_id)
            stripe_session_id = session_id
            
            # Prova prima dal database (più affidabile)
            try:
                row = await _db_execute_one(
                    "SELECT download_token, photo_ids, email FROM orders WHERE stripe_session_id = $1",
                    (stripe_session_id,)
                )
                if row:
                    download_token = row['download_token']
                    photo_ids = json.loads(row['photo_ids']) if row['photo_ids'] else []
                    email = row['email']
                    logger.info(f"Order found in database: {stripe_session_id} - {len(photo_ids)} photos for {email}")
            except Exception as e:
                logger.error(f"Error getting order from database: {e}")
            
            # Se non trovato nel database, prova dal file JSON
            if not download_token:
                order_file = ORDERS_DIR / f"{stripe_session_id}.json"
                if order_file.exists():
                    try:
                        with open(order_file, 'r', encoding='utf-8') as f:
                            order_data = json.load(f)
                            download_token = order_data.get('download_token')
                            photo_ids = order_data.get('photo_ids', [])
                            email = order_data.get('email')
                            logger.info(f"Order found in file: {stripe_session_id} - {len(photo_ids)} photos for {email}")
                    except Exception as e:
                        logger.error(f"Error reading order file: {e}")
            
            # Se ancora non trovato, aspetta un po' (il webhook potrebbe essere in ritardo)
            if not download_token:
                logger.warning(f"Order not found yet: {stripe_session_id}. Webhook might be delayed.")
                # Prova a recuperare email direttamente da Stripe come fallback
                if not email and USE_STRIPE:
                    try:
                        stripe_session = stripe.checkout.Session.retrieve(stripe_session_id)
                        email = stripe_session.get('customer_email') or stripe_session.get('customer_details', {}).get('email') or stripe_session.get('metadata', {}).get('email')
                        photo_ids_str = stripe_session.get('metadata', {}).get('photo_ids', '')
                        if photo_ids_str:
                            photo_ids = photo_ids_str.split(',')
                        logger.info(f"Retrieved email from Stripe session: {email}")
                    except Exception as e:
                        logger.error(f"Error retrieving Stripe session: {e}")
                
                # Aspetta 2 secondi e riprova dal database
                import asyncio
                await asyncio.sleep(2)
                try:
                    row = await _db_execute_one(
                        "SELECT download_token, photo_ids, email FROM orders WHERE stripe_session_id = $1",
                        (stripe_session_id,)
                    )
                    if row:
                        download_token = row['download_token']
                        photo_ids = json.loads(row['photo_ids']) if row['photo_ids'] else []
                        if not email:  # Usa email dal database se non già recuperata
                            email = row['email']
                        logger.info(f"Order found after retry: {stripe_session_id}")
                except Exception as e:
                    logger.error(f"Error retrying order from database: {e}")
            
            # Se non abbiamo email ma abbiamo photo_ids, prova a recuperarla da Stripe
            if not email and photo_ids and USE_STRIPE:
                try:
                    stripe_session = stripe.checkout.Session.retrieve(stripe_session_id)
                    email = stripe_session.get('customer_email') or stripe_session.get('customer_details', {}).get('email') or stripe_session.get('metadata', {}).get('email')
                    if email:
                        logger.info(f"Recovered email from Stripe session in success page: {email}")
                        # Se abbiamo email e photo_ids ma non ordine nel DB, crealo ora
                        if not download_token:
                            logger.info(f"Creating order manually in success page: {email} - {len(photo_ids)} photos")
                            # Recupera amount da Stripe session se disponibile
                            amount_cents = 0
                            try:
                                if stripe_session.get('amount_total'):
                                    amount_cents = stripe_session.get('amount_total')
                            except:
                                pass
                            download_token = await _create_order(email, stripe_session_id, stripe_session_id, photo_ids, amount_cents)
                            if download_token:
                                logger.info(f"Order created successfully in success page: {download_token}")
                            else:
                                logger.error(f"Failed to create order in success page for {email}")
                except Exception as e:
                    logger.error(f"Error recovering email from Stripe in success page: {e}")
            
            base_url = str(request.base_url).rstrip('/')
            
            # Genera HTML per le foto (mostra direttamente)
            photos_html = ""
            if photo_ids:
                for photo_id in photo_ids:
                    # Usa il token e paid=true per assicurarsi che sia servita senza watermark
                    photo_url_params = []
                    if download_token:
                        photo_url_params.append(f"token={download_token}")
                    if email:
                        photo_url_params.append(f"email={email}")
                    photo_url_params.append("paid=true")  # Forza paid=true per foto pagate
                    photo_url = f"/photo/{photo_id}?{'&'.join(photo_url_params)}"
                    # Escape per JavaScript
                    photo_id_escaped = photo_id.replace("'", "\\'").replace('"', '\\"')
                    email_escaped = (email or "").replace("'", "\\'").replace('"', '\\"')
                    photos_html += f"""
                    <div class="photo-item">
                        <img src="{photo_url}" alt="Photo" loading="lazy" class="photo-img" style="cursor: pointer; -webkit-touch-callout: default; -webkit-user-select: none; user-select: none;">
                        <button class="download-btn download-btn-desktop" onclick="downloadPhotoSuccess('{photo_id_escaped}', '{email_escaped}', this)" style="display: none;">📥 Download</button>
                    </div>
                    """
            
            # Prepara le parti HTML che contengono backslash (non possono essere in f-string)
            if photos_html:
                photos_section = f'<div class="photos-grid">{photos_html}</div>'
            else:
                photos_section = '<p style="margin: 20px 0; opacity: 0.8; font-size: 18px;">Le foto verranno caricate a breve. Se non compaiono, clicca su "VAI ALL\'ALBUM COMPLETO" qui sotto.</p>'
            
            # Link intelligente: se ha email, porta direttamente all'album (con parametro view_album per forzare visualizzazione anche se ha foto pagate)
            if email:
                album_button_top = f'<a href="/?email={email}&view_album=true" class="main-button" style="margin-top: 0; margin-bottom: 30px;">📸 Back to album</a>'
                album_button_bottom = f'<a href="/?email={email}&view_album=true" class="main-button" style="margin-top: 30px; margin-bottom: 0;">📸 Back to album</a>'
            else:
                album_button_top = '<a href="/" class="main-button" style="margin-top: 0; margin-bottom: 30px;">📸 Back to album</a>'
                album_button_bottom = '<a href="/" class="main-button" style="margin-top: 30px; margin-bottom: 0;">📸 Back to album</a>'
            
            # Pagina con foto mostrate direttamente
            html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pagamento completato - TenerifePictures</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                    background: linear-gradient(135deg, #7b74ff, #5f58ff);
                    color: #fff;
                    padding: 20px;
                    min-height: 100vh;
                }}
                .container {{
                    max-width: 800px;
                    margin: 0 auto;
                    text-align: center;
                    padding: 40px 20px;
                }}
                .success-icon {{
                    font-size: 80px;
                    margin-bottom: 20px;
                }}
                h1 {{
                    font-size: 36px;
                    margin: 0 0 15px;
                    font-weight: 700;
                }}
                .message {{
                    font-size: 22px;
                    margin: 0 0 30px;
                    line-height: 1.5;
                }}
                .photos-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                    padding: 20px 0;
                }}
                .photo-item {{
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 12px;
                    padding: 15px;
                    backdrop-filter: blur(10px);
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 15px;
                }}
                .photo-img {{
                    width: 100%;
                    height: auto;
                    border-radius: 8px;
                    max-height: 300px;
                    object-fit: contain;
                    background: rgba(0, 0, 0, 0.2);
                }}
                .download-btn {{
                    width: 100%;
                    padding: 12px 20px;
                    background: #22c55e;
                    color: #fff;
                    border: none;
                    border-radius: 8px;
                    font-weight: 600;
                    font-size: 16px;
                    cursor: pointer;
                    transition: transform 0.2s, box-shadow 0.2s;
                    box-shadow: 0 4px 15px rgba(34, 197, 94, 0.3);
                }}
                .download-btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 6px 20px rgba(34, 197, 94, 0.4);
                }}
                .download-btn:active {{
                    transform: translateY(0);
                }}
                .download-btn:disabled {{
                    opacity: 0.6;
                    cursor: not-allowed;
                }}
                .main-button {{
                    display: block;
                    width: 100%;
                    max-width: 400px;
                    margin: 30px auto 0;
                    padding: 20px 40px;
                    background: rgba(255, 255, 255, 0.2);
                    color: #fff;
                    text-decoration: none;
                    border-radius: 12px;
                    font-weight: 700;
                    font-size: 18px;
                    border: 2px solid rgba(255, 255, 255, 0.3);
                    transition: transform 0.2s, background 0.2s;
                }}
                .main-button:hover {{
                    transform: translateY(-2px);
                    background: rgba(255, 255, 255, 0.3);
                }}
                .main-button:active {{
                    transform: translateY(0);
                }}
                @media (max-width: 600px) {{
                    .photos-grid {{
                        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                        gap: 15px;
                    }}
                    h1 {{
                        font-size: 28px;
                    }}
                    .message {{
                        font-size: 18px;
                    }}
                }}
                
                /* Fallback CSS per iOS - mostra istruzioni anche se JS non funziona */
                @supports (-webkit-touch-callout: none) {{
                    #ios-instructions-top {{
                        display: block !important;
                    }}
                    .download-btn-desktop {{
                        display: none !important;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="success-icon">✅</div>
                <h1>PAYMENT COMPLETED!</h1>
                <p class="message">Your photos are ready for download.</p>
                
                <!-- iOS Instructions at top (if iPhone) -->
                <div id="ios-instructions-top" style="display: none; margin: 20px 0; padding: 20px; background: rgba(255, 255, 255, 0.2); border: 2px solid rgba(255, 255, 255, 0.4); border-radius: 12px; backdrop-filter: blur(10px); box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
                    <p style="margin: 0 0 12px 0; font-weight: bold; font-size: 20px; text-align: center;">📱 How to save your photos:</p>
                    <p style="margin: 0; font-size: 17px; line-height: 1.8; text-align: left;">1. Touch and hold on any photo below</p>
                    <p style="margin: 8px 0 0 0; font-size: 17px; line-height: 1.8; text-align: left;">2. Select "Save to Photos" from the menu</p>
                    <p style="margin: 12px 0 0 0; font-size: 15px; line-height: 1.6; text-align: center; opacity: 0.9; font-style: italic;">Repeat for each photo you want to save</p>
                </div>
                
                <!-- Buy more photos button at top -->
                {album_button_top}
                
                <!-- Foto -->
                {photos_section}
                
                <!-- Buy more photos button at bottom -->
                {album_button_bottom}
            </div>
            <script>
                // Rileva se è iOS
                function isIOS() {{
                    return /iPad|iPhone|iPod/.test(navigator.userAgent) || 
                           (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
                }}
                
                // Rileva se è Android
                function isAndroid() {{
                    return /Android/i.test(navigator.userAgent);
                }}
                
                // Mostra/nascondi pulsante e istruzioni in base al dispositivo
                function setupIOSInstructions() {{
                    const iosInstructionsTop = document.getElementById('ios-instructions-top');
                    const downloadBtns = document.querySelectorAll('.download-btn-desktop');
                    
                    console.log('Setting up iOS instructions...');
                    console.log('isIOS():', isIOS());
                    console.log('iosInstructionsTop found:', !!iosInstructionsTop);
                    
                    if (isIOS()) {{
                        // Su iOS: mostra istruzioni in alto, nascondi pulsanti
                        if (iosInstructionsTop) {{
                            iosInstructionsTop.style.display = 'block';
                            console.log('iOS instructions top shown');
                        }}
                        downloadBtns.forEach(el => {{
                            el.style.display = 'none';
                            console.log('Download button hidden');
                        }});
                    }} else {{
                        // Su Android/Desktop: nascondi istruzioni, mostra pulsanti
                        if (iosInstructionsTop) iosInstructionsTop.style.display = 'none';
                        downloadBtns.forEach(el => el.style.display = 'block');
                    }}
                }}
                
                // Esegui subito e anche quando DOM è pronto
                if (document.readyState === 'loading') {{
                    document.addEventListener('DOMContentLoaded', setupIOSInstructions);
                }} else {{
                    setupIOSInstructions();
                }}
                
                // Esegui anche dopo un breve delay per sicurezza
                setTimeout(setupIOSInstructions, 100);
                setTimeout(setupIOSInstructions, 500);
                
                async function downloadPhotoSuccess(photoId, email, btnElement) {{
                    try {{
                        // Trova il pulsante se non passato come parametro
                        const btn = btnElement || event?.target || document.querySelector(`button[onclick*="${{photoId}}"]`);
                        if (!btn) {{
                            console.error('Pulsante non trovato');
                            alert('Errore: pulsante non trovato');
                            return;
                        }}
                        
                        btn.disabled = true;
                        btn.textContent = '⏳ Scaricamento...';
                        
                        const filename = photoId.split('/').pop() || 'foto.jpg';
                        
                        // Costruisci URL con paid=true, email e download=true per verifica backend e forzare download
                        let photoUrl = `/photo/${{encodeURIComponent(photoId)}}?paid=true&download=true`;
                        if (email) {{
                            photoUrl += `&email=${{encodeURIComponent(email)}}`;
                        }}
                        
                        // Su iOS: usa approccio semplice e diretto
                        if (isIOS()) {{
                            try {{
                                // Prova prima con Web Share API (salva direttamente nella galleria)
                                const response = await fetch(photoUrl);
                                if (!response.ok) {{
                                    throw new Error('Errore nel download: ' + response.status);
                                }}
                                
                                const blob = await response.blob();
                                const file = new File([blob], filename, {{ type: 'image/jpeg' }});
                                
                                if (navigator.share && navigator.canShare) {{
                                    try {{
                                        if (navigator.canShare({{ files: [file] }})) {{
                                            await navigator.share({{
                                                files: [file],
                                                title: 'Salva foto',
                                                text: 'Salva questa foto nella galleria'
                                            }});
                                            btn.textContent = '✅ Salvata!';
                                            setTimeout(() => {{
                                                btn.disabled = false;
                                                btn.textContent = '📥 Scarica';
                                            }}, 2000);
                                            return;
                                        }}
                                    }} catch (shareErr) {{
                                        console.log('Web Share error:', shareErr);
                                        // Continua con fallback
                                    }}
                                }}
                                
                                // Fallback: apri l'immagine direttamente usando l'URL
                                // Su iOS Safari, questo permette all'utente di fare long-press e salvare
                                const imgWindow = window.open(photoUrl, '_blank');
                                
                                if (imgWindow) {{
                                    // Mostra istruzioni dopo un breve delay
                                    setTimeout(() => {{
                                        alert('📱 Tocca e tieni premuto sull\'immagine, poi seleziona "Salva in Foto" per salvarla nella galleria.');
                                    }}, 800);
                                    
                                    btn.textContent = '✅ Aperta!';
                                    setTimeout(() => {{
                                        btn.disabled = false;
                                        btn.textContent = '📥 Scarica';
                                    }}, 2000);
                                }} else {{
                                    // Se popup bloccato, mostra istruzioni e riprova con link diretto
                                    alert('📱 Popup bloccato. Per salvare la foto:\n1. Tocca e tieni premuto sull\'immagine qui sotto\n2. Seleziona "Salva in Foto"\n\nOppure apri questa pagina in Safari.');
                                    
                                    // Crea un link visibile temporaneo
                                    const tempLink = document.createElement('a');
                                    tempLink.href = photoUrl;
                                    tempLink.target = '_blank';
                                    tempLink.style.display = 'block';
                                    tempLink.style.margin = '20px auto';
                                    tempLink.style.padding = '15px 30px';
                                    tempLink.style.background = '#22c55e';
                                    tempLink.style.color = '#fff';
                                    tempLink.style.borderRadius = '8px';
                                    tempLink.style.textDecoration = 'none';
                                    tempLink.style.fontWeight = 'bold';
                                    tempLink.textContent = '📱 Tocca qui per aprire la foto';
                                    tempLink.onclick = function(e) {{
                                        e.preventDefault();
                                        window.open(photoUrl, '_blank');
                                    }};
                                    
                                    const container = document.querySelector('.container');
                                    if (container) {{
                                        container.appendChild(tempLink);
                                    }}
                                    
                                    btn.disabled = false;
                                    btn.textContent = '📥 Scarica';
                                }}
                            }} catch (fetchError) {{
                                console.error('Errore fetch:', fetchError);
                                alert('Errore nel caricamento della foto. Riprova.');
                                btn.disabled = false;
                                btn.textContent = '📥 Scarica';
                                return;
                            }}
                        }}
                        // Su Android: download diretto (salva automaticamente nella galleria)
                        else if (isAndroid()) {{
                            // photoUrl già include download=true
                            const link = document.createElement('a');
                            link.href = photoUrl;
                            link.download = filename;
                            link.style.display = 'none';
                            link.target = '_self';
                            document.body.appendChild(link);
                            link.click();
                            
                            setTimeout(() => {{
                                document.body.removeChild(link);
                            }}, 1000);
                            
                            btn.textContent = '✅ Scaricata!';
                            setTimeout(() => {{
                                btn.disabled = false;
                                btn.textContent = '📥 Scarica';
                            }}, 2000);
                        }}
                        // Desktop: download normale
                        else {{
                            // photoUrl già include download=true
                            const response = await fetch(photoUrl);
                            if (!response.ok) {{
                                throw new Error('Errore nel download');
                            }}
                            
                            const blob = await response.blob();
                            const blobUrl = window.URL.createObjectURL(blob);
                            const link = document.createElement('a');
                            link.href = blobUrl;
                            link.download = filename;
                            document.body.appendChild(link);
                            link.click();
                            document.body.removeChild(link);
                            window.URL.revokeObjectURL(blobUrl);
                            
                            btn.textContent = '✅ Scaricata!';
                            setTimeout(() => {{
                                btn.disabled = false;
                                btn.textContent = '📥 Scarica';
                            }}, 2000);
                        }}
                    }} catch (error) {{
                        console.error('Errore download:', error);
                        alert('Errore durante il download. Riprova più tardi.');
                        btn.disabled = false;
                        btn.textContent = '📥 Scarica';
                    }}
                }}
            </script>
        </body>
        </html>
        """
            
            return HTMLResponse(html_content)
        except Exception as e:
            logger.error(f"Error in checkout success page: {e}")
        # Fallback semplice
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pagamento completato</title>
            <meta charset="utf-8">
        </head>
        <body style="font-family: Arial; text-align: center; padding: 50px;">
            <h1>✅ Pagamento completato!</h1>
            <p>Controlla la tua email per il link di download.</p>
            <a href="/">Back to home</a>
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
            <a href="/">Back to home</a>
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
        email = metadata.get('email') or session.get('customer_email') or session.get('customer_details', {}).get('email')
        
        # Recupera photo_ids: prima prova token, poi fallback a lista diretta
        photo_ids_token = metadata.get('photo_ids_token')
        photo_ids_str = metadata.get('photo_ids', '')
        
        if photo_ids_token:
            # Recupera photo_ids dal dizionario in memoria
            if photo_ids_token in checkout_photo_ids:
                photo_ids_list = checkout_photo_ids[photo_ids_token]
                photo_ids_str = ','.join(photo_ids_list)
                logger.info(f"Photo IDs recovered from token: {len(photo_ids_list)} photos")
                # Opzionale: rimuovi token dal dizionario dopo uso (cleanup)
                # del checkout_photo_ids[photo_ids_token]
            else:
                logger.error(f"Token not found in checkout_photo_ids: {photo_ids_token[:16]}...")
                photo_ids_str = ''
        # Se non c'è token, usa photo_ids_str direttamente (retrocompatibilità)
        
        # Log dettagliato per debug
        logger.info(f"=== WEBHOOK RECEIVED ===")
        logger.info(f"Event type: {event['type']}")
        logger.info(f"Session ID from metadata: {session_id}")
        logger.info(f"Session ID from session: {session.get('id')}")
        logger.info(f"Email from metadata: {metadata.get('email')}")
        logger.info(f"Email from session: {session.get('customer_email')}")
        logger.info(f"Email from customer_details: {session.get('customer_details', {}).get('email')}")
        logger.info(f"Final email: {email}")
        logger.info(f"Photo IDs token: {photo_ids_token[:16] if photo_ids_token else 'None'}...")
        logger.info(f"Photo IDs from metadata/token: {photo_ids_str[:100] if photo_ids_str else 'None'}...")
        logger.info(f"Full metadata: {metadata}")
        logger.info(f"Session payment_status: {session.get('payment_status')}")
        logger.info(f"Session status: {session.get('status')}")
        
        if photo_ids_str:
            # Email è obbligatoria per creare l'ordine
            if not email:
                logger.error("Email not found in Stripe session. Cannot create order.")
                logger.error(f"Session data: customer_email={session.get('customer_email')}, customer_details={session.get('customer_details')}, metadata={metadata}")
                # Non creare ordine senza email
                return {"status": "error", "message": "Email not found in payment session"}
            
            # session_id può essere nel metadata (legacy) o None (nuovo flusso)
            session_id = metadata.get('session_id')
            
            # Normalizza email
            original_email = email
            email = _normalize_email(email)
            logger.info(f"Email normalized: '{original_email}' -> '{email}'")
            photo_ids = [pid.strip() for pid in photo_ids_str.split(',') if pid.strip()]
            logger.info(f"Photo IDs parsed: {photo_ids} (count: {len(photo_ids)})")
            order_id = session.get('id')
            amount_cents = session.get('amount_total', 0)
            logger.info(f"Creating order: order_id={order_id}, email={email}, amount={amount_cents}, photos={len(photo_ids)}")
            
            # Stateless: non creare ordine nel database
            base_url = str(request.base_url).rstrip('/')
            # download_token non più necessario in stateless mode
            download_token = None
            
            logger.info(f"Payment confirmed for {email} - {len(photo_ids)} photos (stateless mode)")
            
            # Stateless: non salvare ordine in file JSON
            # Gli ordini sono tracciati solo tramite Stripe session_id
            logger.info(f"Order completed: {order_id} - {len(photo_ids)} photos for {email}")
            
            # Stateless: non svuotare carrello (non più usato)
        else:
            logger.error(f"Order failed: missing photo_ids in webhook")
    
    return {"ok": True}

@app.get("/test-email")
async def test_email(
    email: str = Query(..., description="Email di test")
):
    """Endpoint di test email - DISABLED (email system removed)"""
    return {
        "ok": False,
        "message": "Email system disabled - no emails will be sent",
        "email": email
    }

@app.get("/test-download")
async def test_download_page(
    request: Request,
    email: str = Query("test@example.com", description="Email di test"),
    photo_id: str = Query(None, description="ID foto di test (opzionale)")
):
    """Pagina di test per verificare il download su iPhone senza completare il flusso completo"""
    if R2_ONLY_MODE:
        raise HTTPException(status_code=503, detail="Test download endpoint disabled in R2_ONLY_MODE. Use R2 for photos.")
    
    try:
        # Se non specificato, prendi la prima foto disponibile
        if not photo_id:
            photos = list(PHOTOS_DIR.glob("*.jpg")) + list(PHOTOS_DIR.glob("*.jpeg"))
            if photos:
                photo_id = photos[0].name
            else:
                return HTMLResponse("""
                <html>
                <body style="font-family: Arial; padding: 50px; text-align: center;">
                    <h1>❌ Nessuna foto disponibile</h1>
                    <p>Carica almeno una foto per testare il download.</p>
                    <a href="/">Back to home</a>
                </body>
                </html>
                """)
        
        # Usa la stessa struttura HTML della pagina di successo checkout
        base_url = str(request.base_url).rstrip('/')
        photo_url = f"{base_url}/photo/{photo_id}?paid=true&email={email}"
        
        # Escape per sicurezza
        photo_id_escaped = photo_id.replace("'", "\\'").replace('"', '&quot;')
        email_escaped = email.replace("'", "\\'").replace('"', '&quot;')
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Test Download - iPhone</title>
            <style>
                body {{
                    margin: 0;
                    padding: 20px;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
                .container {{
                    background: white;
                    border-radius: 20px;
                    padding: 40px;
                    max-width: 600px;
                    width: 100%;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                    text-align: center;
                }}
                h1 {{
                    color: #333;
                    margin-bottom: 20px;
                }}
                .photo-item {{
                    margin: 30px 0;
                }}
                .photo-img {{
                    width: 100%;
                    max-width: 400px;
                    border-radius: 12px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                }}
                .download-btn {{
                    margin-top: 20px;
                    padding: 15px 30px;
                    background: #22c55e;
                    color: white;
                    border: none;
                    border-radius: 12px;
                    font-size: 18px;
                    font-weight: bold;
                    cursor: pointer;
                    transition: transform 0.2s, background 0.2s;
                }}
                .download-btn:hover {{
                    transform: translateY(-2px);
                    background: #16a34a;
                }}
                .download-btn:disabled {{
                    background: #ccc;
                    cursor: not-allowed;
                }}
                .info {{
                    background: #f0f9ff;
                    border: 1px solid #0ea5e9;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 20px 0;
                    color: #0369a1;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🧪 Test Download iPhone</h1>
                <div class="info">
                    <p><strong>Questa è una pagina di test</strong></p>
                    <p>Usa questa pagina per testare il download su iPhone senza completare il flusso completo.</p>
                </div>
                <div class="photo-item">
                    <img src="{photo_url}" alt="Foto test" class="photo-img" style="cursor: pointer;" onclick="if(isIOS()) {{ alert('📱 Per salvare: Tocca e tieni premuto sull\\'immagine, poi seleziona \\'Salva in Foto\\''); }}">
                    <div id="ios-instructions" style="display: none; margin-top: 20px; padding: 15px; background: #f0f9ff; border: 2px solid #0ea5e9; border-radius: 12px; color: #0369a1;">
                        <p style="margin: 0; font-weight: bold; font-size: 16px;">📱 Come salvare la foto:</p>
                        <p style="margin: 10px 0 0 0;">1. Tocca e tieni premuto sull'immagine qui sopra</p>
                        <p style="margin: 5px 0 0 0;">2. Seleziona "Salva in Foto"</p>
                    </div>
                    <button id="download-btn-desktop" onclick="downloadPhotoSuccess('{photo_id_escaped}', '{email_escaped}', this)" class="download-btn" style="display: none;">📥 Scarica</button>
                </div>
                <p style="margin-top: 30px;">
                    <a href="/" style="color: #667eea; text-decoration: none;">← Back to home</a>
                </p>
            </div>
            <script>
                // Rileva se è iOS
                function isIOS() {{
                    return /iPad|iPhone|iPod/.test(navigator.userAgent) || 
                           (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
                }}
                
                // Rileva se è Android
                function isAndroid() {{
                    return /Android/i.test(navigator.userAgent);
                }}
                
                // Mostra/nascondi pulsante e istruzioni in base al dispositivo
                document.addEventListener('DOMContentLoaded', function() {{
                    const iosInstructions = document.getElementById('ios-instructions');
                    const downloadBtn = document.getElementById('download-btn-desktop');
                    
                    if (isIOS()) {{
                        // Su iOS: mostra istruzioni, nascondi pulsante
                        if (iosInstructions) iosInstructions.style.display = 'block';
                        if (downloadBtn) downloadBtn.style.display = 'none';
                    }} else {{
                        // Su Android/Desktop: mostra pulsante, nascondi istruzioni
                        if (iosInstructions) iosInstructions.style.display = 'none';
                        if (downloadBtn) downloadBtn.style.display = 'block';
                    }}
                }});
                
                async function downloadPhotoSuccess(photoId, email, btnElement) {{
                    console.log('downloadPhotoSuccess chiamata:', photoId, email, btnElement);
                    try {{
                        const btn = btnElement || (typeof event !== 'undefined' ? event.target : null);
                        if (!btn) {{
                            console.error('Pulsante non trovato');
                            alert('Errore: pulsante non trovato');
                            return;
                        }}
                        
                        console.log('Pulsante trovato, disabilito...');
                        btn.disabled = true;
                        btn.textContent = '⏳ Scaricamento...';
                        
                        const filename = photoId.split('/').pop() || 'foto.jpg';
                        console.log('Filename:', filename);
                        
                        // Costruisci URL con paid=true, email e download=true
                        let photoUrl = '/photo/' + encodeURIComponent(photoId) + '?paid=true&download=true';
                        if (email) {{
                            photoUrl += '&email=' + encodeURIComponent(email);
                        }}
                        console.log('Photo URL:', photoUrl);
                        console.log('isIOS():', isIOS());
                        console.log('isAndroid():', isAndroid());
                        
                        // Su iOS: usa approccio semplice e diretto
                        if (isIOS()) {{
                            console.log('iOS rilevato, uso Web Share API o fallback');
                            try {{
                                // Prova prima con Web Share API
                                console.log('Fetch foto...');
                                const response = await fetch(photoUrl);
                                if (!response.ok) {{
                                    throw new Error('Errore nel download: ' + response.status);
                                }}
                                
                                console.log('Foto scaricata, creo blob...');
                                const blob = await response.blob();
                                const file = new File([blob], filename, {{ type: 'image/jpeg' }});
                                
                                if (navigator.share && navigator.canShare) {{
                                    try {{
                                        console.log('Provo Web Share API...');
                                        if (navigator.canShare({{ files: [file] }})) {{
                                            await navigator.share({{
                                                files: [file],
                                                title: 'Salva foto',
                                                text: 'Salva questa foto nella galleria'
                                            }});
                                            console.log('Web Share completato');
                                            btn.textContent = '✅ Salvata!';
                                            setTimeout(() => {{
                                                btn.disabled = false;
                                                btn.textContent = '📥 Scarica';
                                            }}, 2000);
                                            return;
                                        }}
                                    }} catch (shareErr) {{
                                        console.log('Web Share error:', shareErr);
                                    }}
                                }}
                                
                                // Fallback: apri l'immagine
                                console.log('Web Share non disponibile, uso fallback...');
                                const imgWindow = window.open(photoUrl, '_blank');
                                
                                if (imgWindow) {{
                                    setTimeout(() => {{
                                        alert('📱 Tocca e tieni premuto sull\\'immagine, poi seleziona "Salva in Foto" per salvarla nella galleria.');
                                    }}, 800);
                                    
                                    btn.textContent = '✅ Aperta!';
                                    setTimeout(() => {{
                                        btn.disabled = false;
                                        btn.textContent = '📥 Scarica';
                                    }}, 2000);
                                }} else {{
                                    alert('📱 Popup bloccato. Per salvare la foto:\\n1. Tocca e tieni premuto sull\\'immagine qui sotto\\n2. Seleziona "Salva in Foto"');
                                    btn.disabled = false;
                                    btn.textContent = '📥 Scarica';
                                }}
                            }} catch (fetchError) {{
                                console.error('Errore fetch:', fetchError);
                                alert('Errore nel caricamento della foto. Riprova.\\nErrore: ' + fetchError.message);
                                btn.disabled = false;
                                btn.textContent = '📥 Scarica';
                                return;
                            }}
                        }}
                        // Su Android: download diretto
                        else if (isAndroid()) {{
                            const link = document.createElement('a');
                            link.href = photoUrl;
                            link.download = filename;
                            link.style.display = 'none';
                            document.body.appendChild(link);
                            link.click();
                            setTimeout(() => {{
                                document.body.removeChild(link);
                            }}, 1000);
                            
                            btn.textContent = '✅ Scaricata!';
                            setTimeout(() => {{
                                btn.disabled = false;
                                btn.textContent = '📥 Scarica';
                            }}, 2000);
                        }}
                        // Desktop: download normale
                        else {{
                            console.log('Desktop rilevato, uso download normale');
                            try {{
                                console.log('Fetch foto per download...');
                                const response = await fetch(photoUrl);
                                if (!response.ok) {{
                                    throw new Error('Errore nel download: ' + response.status);
                                }}
                                
                                console.log('Foto scaricata, creo blob URL...');
                                const blob = await response.blob();
                                const blobUrl = window.URL.createObjectURL(blob);
                                const link = document.createElement('a');
                                link.href = blobUrl;
                                link.download = filename;
                                link.style.display = 'none';
                                document.body.appendChild(link);
                                console.log('Click su link download...');
                                link.click();
                                setTimeout(() => {{
                                    document.body.removeChild(link);
                                    window.URL.revokeObjectURL(blobUrl);
                                }}, 100);
                                
                                btn.textContent = '✅ Scaricata!';
                                setTimeout(() => {{
                                    btn.disabled = false;
                                    btn.textContent = '📥 Scarica';
                                }}, 2000);
                            }} catch (desktopError) {{
                                console.error('Errore download desktop:', desktopError);
                                alert('Errore durante il download: ' + desktopError.message);
                                btn.disabled = false;
                                btn.textContent = '📥 Scarica';
                            }}
                        }}
                    }} catch (error) {{
                        console.error('Errore download:', error);
                        alert('Errore durante il download. Riprova più tardi.');
                        btn.disabled = false;
                        btn.textContent = '📥 Scarica';
                    }}
                }}
            </script>
        </body>
        </html>
        """
        
        return HTMLResponse(html_content)
    except Exception as e:
        logger.error(f"Error in test download page: {e}", exc_info=True)
        return HTMLResponse(f"""
        <html>
        <body style="font-family: Arial; padding: 50px; text-align: center;">
            <h1>❌ Errore</h1>
            <p>{str(e)}</p>
            <a href="/">Back to home</a>
        </body>
        </html>
        """)

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
        
        # Servi foto originale (senza watermark) da R2
        if not USE_R2 or r2_client is None:
            raise HTTPException(status_code=503, detail="R2 storage not configured")
        
        try:
            photo_bytes = await _r2_get_object_bytes(photo_id)
            logger.info(f"Serving photo from R2: key={photo_id}, bucket={R2_BUCKET}")
            
            # Traccia download
            _track_download(photo_id)
            
            # Headers cross-platform corretti per download
            from pathlib import Path
            filename = Path(photo_id).name
            headers = {
                "Content-Type": "image/jpeg",
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "public, max-age=31536000, immutable"
            }
            
            logger.info(f"PHOTO SERVE: source=R2, filename={photo_id}")
            
            return Response(content=photo_bytes, media_type="image/jpeg", headers=headers)
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code in ('404', 'NoSuchKey'):
                logger.warning(f"[DOWNLOAD] Missing R2 object: {photo_id}")
                # 404 pulito per immagini
                return Response(status_code=404, content=b"", media_type="image/jpeg")
            logger.error(f"[DOWNLOAD] R2 error for {photo_id}: {error_code or type(e).__name__}")
            raise HTTPException(status_code=503, detail="R2 storage error")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error serving photo from R2: {e}")
            raise HTTPException(status_code=500, detail=f"Error serving photo: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading photo: {e}")
        raise HTTPException(status_code=500, detail=f"Download error: {str(e)}")

# ========== ADMIN PANEL ==========

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")  # Cambia in produzione!
ADMIN_RESET_SECRET = os.getenv("ADMIN_RESET_SECRET", "")  # Secret per reset database

def _check_admin_auth(password: Optional[str] = None) -> bool:
    """Verifica autenticazione admin"""
    if not password:
        return False
    return password == ADMIN_PASSWORD

def _check_admin_reset_secret(request: Request) -> bool:
    """Verifica header X-Admin-Secret per operazioni critiche"""
    secret = request.headers.get("X-Admin-Secret", "")
    if not ADMIN_RESET_SECRET:
        logger.warning("ADMIN_RESET_SECRET not configured - reset endpoint disabled")
        return False
    return secret == ADMIN_RESET_SECRET

@app.get("/check-version")
async def check_version():
    """Endpoint PUBBLICO per verificare quale versione è deployata - NO AUTH RICHIESTO"""
    try:
        file_path = Path(__file__).resolve()
        with open(file_path, 'r', encoding='utf-8') as f:
            first_10_lines = [f.readline() for _ in range(10)]
        
        version_line = None
        for line in first_10_lines:
            if "BUILD_VERSION" in line:
                version_line = line.strip()
                break
        
        # Verifica se l'endpoint /admin/debug esiste
        import inspect
        has_debug_endpoint = False
        try:
            # Cerca la funzione admin_debug nel modulo corrente
            if 'admin_debug' in globals():
                source = inspect.getsource(admin_debug)
                has_debug_endpoint = True
        except:
            pass
        
        return {
            "status": "ok",
            "python_file": str(file_path),
            "build_version": version_line or "NON TROVATO - VERSIONE VECCHIA",
            "has_debug_endpoint": has_debug_endpoint,
            "has_admin_panel_logging": "🔍 ADMIN PANEL v2.2" in inspect.getsource(admin_panel),
            "message": "Se build_version è 'NON TROVATO', Render sta servendo codice vecchio"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}

@app.get("/admin/version")
async def admin_version():
    """Endpoint per verificare quale versione del codice è in esecuzione"""
    import inspect
    try:
        # Leggi la prima riga del file per vedere la versione
        file_path = Path(__file__).resolve()
        with open(file_path, 'r', encoding='utf-8') as f:
            first_lines = [f.readline() for _ in range(5)]
        
        version_line = None
        for line in first_lines:
            if "BUILD_VERSION" in line:
                version_line = line.strip()
                break
        
        return {
            "file_path": str(file_path),
            "version": version_line or "Versione non trovata",
            "has_admin_panel_logging": "🔍 ADMIN PANEL v2.2" in inspect.getsource(admin_panel),
            "has_cursor_fix": "await cursor.fetchall()" not in inspect.getsource(admin_orders) if 'admin_orders' in globals() else "N/A"
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/test-admin-file")
async def test_admin_file():
    """Endpoint PUBBLICO per testare se il file admin.html viene letto correttamente"""
    admin_path = STATIC_DIR / "admin.html"
    
    result = {
        "file_path": str(admin_path.resolve()),
        "file_exists": admin_path.exists(),
        "can_read": False,
        "file_size": 0,
        "has_version_2_2": False,
        "has_date_selector": False,
        "first_200_chars": "",
        "error": None
    }
    
    if admin_path.exists():
        try:
            with open(admin_path, 'r', encoding='utf-8') as f:
                content = f.read()
                result["can_read"] = True
                result["file_size"] = len(content)
                result["has_version_2_2"] = "VERSIONE 2.2" in content
                result["has_date_selector"] = "dateSelector" in content
                result["first_200_chars"] = content[:200]
        except Exception as e:
            result["error"] = str(e)
    
    return result

@app.get("/admin", response_class=HTMLResponse)
def admin_panel():
    """Admin page - serve admin.html"""
    admin_path = STATIC_DIR / "admin.html"
    if not admin_path.exists():
        logger.error(f"❌ admin.html not found at: {admin_path.resolve()}")
        raise HTTPException(status_code=500, detail=f"admin.html not found: {admin_path}")
    logger.info(f"🔐 Serving admin.html from: {admin_path.resolve()}")
    return FileResponse(
        admin_path,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.get("/admin/debug")
async def admin_debug(password: str = Query(..., description="Password admin")):
    """Debug endpoint per verificare quale versione di admin.html è deployata"""
    if not _check_admin_auth(password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    admin_path = STATIC_DIR / "admin.html"
    
    # Verifica anche il codice Python
    import inspect
    python_code = inspect.getsource(admin_panel)
    has_logging = "=== ADMIN PANEL REQUEST ===" in python_code
    has_cursor_fix = "await cursor.fetchall()" not in inspect.getsource(admin_orders)
    
    result = {
        "python_code_version": "BUILD_VERSION: 2026-01-05-00-25-FORCE-REBUILD" if "BUILD_VERSION" in open(__file__).read()[:500] else "Vecchia versione",
        "python_has_logging": has_logging,
        "python_has_cursor_fix": has_cursor_fix,
        "static_dir": str(STATIC_DIR.resolve()),
        "admin_path": str(admin_path.resolve()),
        "file_exists": admin_path.exists(),
    }
    
    if not admin_path.exists():
        result["error"] = "File not found"
        return result
    
    try:
        with open(admin_path, 'r', encoding='utf-8') as f:
            content = f.read()
            version_2_2 = "VERSIONE 2.2" in content
            version_2_1 = "VERSIONE 2.1" in content
            version_2_0 = "VERSIONE 2.0" in content
            has_date_selector = "dateSelector" in content
            has_selettore_data = "Selettore Data" in content
            has_timestamp = "TIMESTAMP: 2026-01-04-23:45" in content
            
            # Leggi anche il file dal repository per confronto
            try:
                import subprocess
                git_content = subprocess.check_output(
                    ["git", "show", "HEAD:static/admin.html"],
                    cwd=BASE_DIR.parent,
                    stderr=subprocess.DEVNULL
                ).decode('utf-8')
                git_has_2_2 = "VERSIONE 2.2" in git_content
                result["git_repo_has_2.2"] = git_has_2_2
            except:
                result["git_repo_check"] = "Non disponibile"
            
            result.update({
                "file_size": len(content),
                "version_2.2": version_2_2,
                "version_2.1": version_2_1,
                "version_2.0": version_2_0,
                "has_date_selector": has_date_selector,
                "has_selettore_data": has_selettore_data,
                "has_timestamp": has_timestamp,
                "first_100_chars": content[:100],
                "title_line": [line for line in content.split('\n')[:15] if 'title' in line.lower()][:1]
            })
    except Exception as e:
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()
    
    return result

@app.get("/admin/stats")
async def admin_stats(password: str = Query(..., description="Password admin")):
    """Statistiche dashboard admin"""
    if not _check_admin_auth(password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        total_orders = 0
        total_revenue = 0
        total_users = 0
        recent_orders = []
        
        # Ordini totali e ricavi
        row = await _db_execute_one("SELECT COUNT(*) as count, SUM(amount_cents) as total FROM orders")
        if row:
            total_orders = row['count'] or 0
            total_revenue = row['total'] or 0
        
        # Utenti totali
            row = await _db_execute_one("SELECT COUNT(DISTINCT email) as count FROM users")
            if row:
                total_users = row['count'] or 0
            
            # Ordini recenti (ultimi 10)
            rows = await _db_execute("""
                SELECT email, photo_ids, amount_cents, paid_at, download_token
                FROM orders
                ORDER BY paid_at DESC
                LIMIT 10
            """)
            for row in rows:
                photo_ids = json.loads(row['photo_ids']) if row['photo_ids'] else []
                recent_orders.append({
                    'email': row['email'],
                    'photo_count': len(photo_ids),
                    'amount_cents': row['amount_cents'],
                    'paid_at': str(row['paid_at']),
                    'download_token': row['download_token']
                })
        
        # Foto totali (disabilitato in R2_ONLY_MODE)
        if R2_ONLY_MODE:
            total_photos = 0  # In R2_ONLY_MODE non contiamo foto da filesystem
        else:
            total_photos = len(list(PHOTOS_DIR.glob("*.jpg"))) + len(list(PHOTOS_DIR.glob("*.jpeg"))) + len(list(PHOTOS_DIR.glob("*.png")))
        
        return {
            "ok": True,
            "total_orders": total_orders,
            "total_revenue": total_revenue,
            "total_users": total_users,
            "total_photos": total_photos,
            "recent_orders": recent_orders
        }
    except Exception as e:
        logger.error(f"Error getting admin stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/orders")
async def admin_orders(password: str = Query(..., description="Password admin")):
    """Lista ordini admin"""
    if not _check_admin_auth(password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        orders = []
        rows = await _db_execute("""
            SELECT email, photo_ids, amount_cents, paid_at, download_token, order_id
            FROM orders
            ORDER BY paid_at DESC
        """)
        for row in rows:
                photo_ids = json.loads(row['photo_ids']) if row['photo_ids'] else []
                orders.append({
                    'email': row['email'],
                    'photo_ids': photo_ids,
                    'photo_count': len(photo_ids),
                    'amount_cents': row['amount_cents'],
                    'paid_at': str(row['paid_at']),
                    'download_token': row['download_token'],
                    'order_id': row['order_id']
                })
        
        return {"ok": True, "orders": orders}
    except Exception as e:
        logger.error(f"Error getting admin orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/upload")
async def admin_upload(
    photo: UploadFile = File(...),
    password: str = Form(...),
    tour_date: Optional[str] = Form(None)
):
    """Upload e indicizzazione foto admin - converte automaticamente in JPEG"""
    if not _check_admin_auth(password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        global meta_rows, faiss_index
        
        # Leggi il contenuto del file
        content = await photo.read()
        
        # Determina nome file finale (sempre JPEG)
        original_ext = Path(photo.filename).suffix.lower()
        if original_ext in ['.jpg', '.jpeg']:
            # È già JPEG, usa nome originale
            jpeg_filename = photo.filename
            photo_bytes = content  # Usa contenuto originale senza riconversione
        else:
            # Converti l'immagine in JPEG
            img = _read_image_from_bytes(content)
            original_name = Path(photo.filename).stem
            jpeg_filename = f"{original_name}.jpg"
            
            # Converti e salva in bytes
            from io import BytesIO
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # Salva come JPEG con qualità massima (100) e subsampling=0
            output = BytesIO()
            pil_img.save(output, 'JPEG', quality=100, optimize=False, subsampling=0)
            photo_bytes = output.getvalue()
            logger.info(f"Photo converted to JPEG (max quality, subsampling=0): {jpeg_filename} (original: {photo.filename})")
        
        # Verifica se R2 è configurato
        if not USE_R2 or r2_client is None:
            raise HTTPException(status_code=500, detail="R2 storage not configured. Cannot upload photos.")
        
        # Evita duplicati: verifica se esiste già su R2
        counter = 1
        original_name = Path(jpeg_filename).stem
        final_filename = jpeg_filename
        while True:
            try:
                r2_client.head_object(Bucket=R2_BUCKET, Key=final_filename)
                # Esiste già, prova con numero
                final_filename = f"{original_name}_{counter}.jpg"
                counter += 1
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    # Non esiste, usa questo nome
                    break
                else:
                    raise
        
        # Salva su R2
        try:
            r2_client.put_object(
                Bucket=R2_BUCKET,
                Key=final_filename,
                Body=photo_bytes,
                ContentType='image/jpeg'
            )
            logger.info(f"PHOTO STORAGE: target=R2, filename={final_filename}")
            logger.info(f"Photo saved to R2: {final_filename} ({len(photo_bytes) / 1024:.1f} KB)")
        except Exception as e:
            logger.error(f"Error saving photo to R2: {e}")
            raise HTTPException(status_code=500, detail=f"Error saving photo to R2: {str(e)}")
        
        # Usa final_filename per indicizzazione
        jpeg_filename = final_filename
        
        # Leggi l'immagine per l'indicizzazione (usa il contenuto originale)
        img = _read_image_from_bytes(content)
        
        # Indicizza foto (se face_app è disponibile)
        if face_app is not None:
            faces = face_app.get(img)
            
            if faces:
                # Foto con volti - aggiungi all'indice
                for f in faces:
                    embedding = _normalize(f.embedding.astype("float32"))
                    if faiss_index is not None:
                        faiss_index.add(embedding.reshape(1, -1))
                    
                    # Aggiungi a meta (con tour_date se fornita)
                    record = {
                        "face_idx": len(meta_rows),
                        "photo_id": jpeg_filename,  # Usa il nome JPEG convertito
                        "has_face": True,
                        "det_score": float(getattr(f, "det_score", 0.0)),
                        "bbox": [float(x) for x in f.bbox.tolist()],
                    }
                    if tour_date:
                        record["tour_date"] = tour_date
                    meta_rows.append(record)
                    
                    # Salva meta su file (disabilitato in R2_ONLY_MODE)
                    if not R2_ONLY_MODE:
                        with open(META_PATH, 'a', encoding='utf-8') as meta_f:
                            meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    else:
                        logger.info("R2_ONLY_MODE: Skipping metadata write to filesystem")
                
                # Salva indice aggiornato
                if faiss_index is not None:
                    # In R2_ONLY_MODE non salviamo l'indice su filesystem
                    if not R2_ONLY_MODE:
                        faiss.write_index(faiss_index, str(INDEX_PATH))
                    else:
                        logger.info("R2_ONLY_MODE: Skipping FAISS index write to filesystem")
                
                logger.info(f"Photo indexed: {photo.filename} - {len(faces)} faces (tour_date: {tour_date})")
            else:
                # Foto senza volti -> salta (non gestiamo foto senza volti)
                logger.info(f"Photo without faces skipped: {jpeg_filename}")
                
                
        
        return {"ok": True, "filename": jpeg_filename, "original_filename": photo.filename}
    except Exception as e:
        logger.error(f"Error uploading photo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/purge")
async def admin_purge(request: Request):
    """Endpoint admin per pulizia completa del DB (hard delete di foto non esistenti in R2)"""
    # Verifica token admin
    admin_token = request.headers.get("X-Admin-Token", "")
    if not ADMIN_TOKEN or admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized: invalid or missing admin token")
    
    if not USE_R2 or not r2_client:
        raise HTTPException(status_code=503, detail="R2 not configured - cannot perform purge")
    
    logger.info("[ADMIN PURGE] Starting manual purge...")
    purge_stats = {
        "user_photos_deleted": 0,
        "indexed_photos_deleted": 0,
        "photo_assets_deleted": 0,
        "orders_cleaned": 0,
        "status_deleted_removed": 0
    }
    
    try:
        # 1. Recupera tutte le chiavi R2 (foto)
        logger.info("[ADMIN PURGE] Fetching R2 keys...")
        r2_keys_set = set()
        paginator = r2_client.get_paginator('list_objects_v2')
        prefix = R2_PHOTOS_PREFIX if R2_PHOTOS_PREFIX else ""
        
        for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key not in ['faces.index', 'faces.meta.jsonl']:
                    if any(key.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.heic']):
                        r2_keys_set.add(key)
        
        logger.info(f"[ADMIN PURGE] Found {len(r2_keys_set)} photos in R2")
        
        if not db_pool:
            raise HTTPException(status_code=503, detail="Database not available")
        
        async with db_pool.acquire() as conn:
            # 2. Elimina tutti i record con status='deleted' da user_photos
            deleted_status_count = await conn.execute("""
                DELETE FROM user_photos WHERE status = 'deleted'
            """)
            purge_stats["status_deleted_removed"] = int(str(deleted_status_count).split()[-1]) if deleted_status_count else 0
            logger.info(f"[ADMIN PURGE] Removed {purge_stats['status_deleted_removed']} records with status='deleted' from user_photos")
            
            # 3. Trova photo_ids nel DB che non esistono in R2
            if r2_keys_set:
                # Recupera tutti i photo_id dal DB
                db_photo_ids = await conn.fetch("SELECT DISTINCT photo_id FROM indexed_photos")
                db_photo_ids_set = {row['photo_id'] for row in db_photo_ids}
                
                missing_in_r2 = db_photo_ids_set - r2_keys_set
                logger.info(f"[ADMIN PURGE] Found {len(missing_in_r2)} photo_ids in DB that don't exist in R2")
                
                if missing_in_r2:
                    missing_list = list(missing_in_r2)
                    
                    # 4. Hard delete in cascata
                    # a) user_photos
                    deleted_user = await conn.execute("""
                        DELETE FROM user_photos WHERE photo_id = ANY($1::text[])
                    """, (missing_list,))
                    purge_stats["user_photos_deleted"] = int(str(deleted_user).split()[-1]) if deleted_user else 0
                    
                    # b) photo_assets (prima di indexed_photos per evitare constraint)
                    deleted_assets = await conn.execute("""
                        DELETE FROM photo_assets WHERE photo_id = ANY($1::text[])
                    """, (missing_list,))
                    purge_stats["photo_assets_deleted"] = int(str(deleted_assets).split()[-1]) if deleted_assets else 0
                    
                    # c) indexed_photos
                    deleted_indexed = await conn.execute("""
                        DELETE FROM indexed_photos WHERE photo_id = ANY($1::text[])
                    """, (missing_list,))
                    purge_stats["indexed_photos_deleted"] = int(str(deleted_indexed).split()[-1]) if deleted_indexed else 0
                    
                    logger.info(f"[ADMIN PURGE] Hard deleted {len(missing_list)} photos: user_photos={purge_stats['user_photos_deleted']}, indexed_photos={purge_stats['indexed_photos_deleted']}, photo_assets={purge_stats['photo_assets_deleted']}")
                    
                    # 5. Pulisci ordini: rimuovi photo_ids non esistenti
                    orders_rows = await conn.fetch("""
                        SELECT order_id, photo_ids FROM orders WHERE photo_ids IS NOT NULL
                    """)
                    
                    for order_row in orders_rows:
                        order_id = order_row['order_id']
                        photo_ids_json = order_row['photo_ids']
                        if not photo_ids_json:
                            continue
                        
                        try:
                            photo_ids_list = json.loads(photo_ids_json) if isinstance(photo_ids_json, str) else photo_ids_json
                            original_count = len(photo_ids_list)
                            # Filtra solo photo_ids che esistono in R2
                            cleaned_list = [pid for pid in photo_ids_list if pid in r2_keys_set]
                            
                            if len(cleaned_list) < original_count:
                                await conn.execute("""
                                    UPDATE orders 
                                    SET photo_ids = $1::jsonb
                                    WHERE order_id = $2
                                """, (json.dumps(cleaned_list), order_id))
                                purge_stats["orders_cleaned"] += 1
                                logger.info(f"[ADMIN PURGE] Cleaned order {order_id}: removed {original_count - len(cleaned_list)} missing photo_ids")
                        except Exception as e:
                            logger.warning(f"[ADMIN PURGE] Error cleaning order {order_id}: {e}")
            else:
                logger.warning("[ADMIN PURGE] R2 keys set is empty - skipping photo_id cleanup")
        
        logger.info(f"[ADMIN PURGE] Purge completed: {purge_stats}")
        return {
            "ok": True,
            "message": "Purge completed successfully",
            "stats": purge_stats
        }
    except Exception as e:
        logger.error(f"[ADMIN PURGE] Error during purge: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Purge error: {str(e)}")

@app.get("/admin/photos-by-date")
async def admin_photos_by_date(password: str = Query(..., description="Password admin")):
    """Ottieni tutte le foto organizzate per data del tour"""
    if not _check_admin_auth(password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        photos_by_date = defaultdict(list)
        
        # Ricarica sempre i dati dai file JSON per avere dati aggiornati
        # Foto con volti (da META_PATH)
        current_meta_rows = []
        if META_PATH.exists():
            current_meta_rows = _load_meta_jsonl(META_PATH)
        else:
            # Fallback a meta_rows in memoria se il file non esiste
            current_meta_rows = meta_rows
        
        for record in current_meta_rows:
            photo_id = record.get("photo_id")
            tour_date = record.get("tour_date", "Senza data")
            if photo_id:
                # Evita duplicati
                if not any(p["photo_id"] == photo_id for p in photos_by_date[tour_date]):
                    photos_by_date[tour_date].append({
                        "photo_id": photo_id,
                        "has_face": True,
                        "det_score": record.get("det_score", 0.0)
                    })
        
        # Converti in lista ordinata per data (più recente prima)
        result = []
        for date in sorted(photos_by_date.keys(), reverse=True):
            result.append({
                "date": date,
                "photos": photos_by_date[date],
                "count": len(photos_by_date[date])
            })
        
        return {"ok": True, "photos_by_date": result}
    except Exception as e:
        logger.error(f"Error getting photos by date: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/packages")
async def admin_packages(
    password: str = Query(..., description="Password admin"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """Analisi pacchetti venduti: distribuzione, trend, confronto con offerte"""
    if not _check_admin_auth(password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        from datetime import datetime, timedelta
        from collections import defaultdict
        
        # Prezzi offerti (da main.py calculate_price)
        OFFERED_PRICES = {
            1: 2000,      # €20.00
            2: 4000,      # €40.00
            3: 3500,      # €35.00
            4: 4000,      # €40.00
            5: 4500,      # €45.00
            (6, 11): 5000,  # €50.00 (6-11 foto)
            (12, 999): 6000  # €60.00 (12+ foto)
        }
        
        # Query base
        query = "SELECT email, photo_ids, amount_cents, paid_at FROM orders WHERE 1=1"
        params = []
        
        # Filtro per data se specificato
        if start_date:
            query += " AND paid_at >= $1"
            params.append(start_date)
            if end_date:
                query += " AND paid_at <= $2"
                params.append(end_date)
        elif end_date:
            query += " AND paid_at <= $1"
            params.append(end_date)
        
        query += " ORDER BY paid_at DESC"
        
        rows = await _db_execute(query, tuple(params) if params else ())
        
        # Raggruppa per pacchetto
        package_stats = defaultdict(lambda: {
            'count': 0,
            'total_revenue': 0,
            'total_photos': 0,
            'dates': []
        })
        
        # Raggruppa per mese per trend
        monthly_stats = defaultdict(lambda: {
            'orders': 0,
            'revenue': 0,
            'photos': 0,
            'packages': defaultdict(int)
        })
        
        total_orders = 0
        total_revenue = 0
        
        for row in rows:
            photo_ids = json.loads(row['photo_ids']) if row['photo_ids'] else []
            photo_count = len(photo_ids)
            amount_cents = row['amount_cents']
            paid_at = row['paid_at']
            
            # Determina pacchetto
            if photo_count == 1:
                package_key = "1 photo"
            elif photo_count == 2:
                package_key = "2 photos"
            elif photo_count == 3:
                package_key = "3 photos"
            elif photo_count == 4:
                package_key = "4 photos"
            elif photo_count == 5:
                package_key = "5 photos"
            elif 6 <= photo_count <= 11:
                package_key = "6-11 photos"
            else:
                package_key = "12+ photos"
            
            # Aggiorna statistiche pacchetto
            package_stats[package_key]['count'] += 1
            package_stats[package_key]['total_revenue'] += amount_cents
            package_stats[package_key]['total_photos'] += photo_count
            package_stats[package_key]['dates'].append(paid_at)
            
            # Aggiorna statistiche mensili
            if isinstance(paid_at, str):
                try:
                    paid_date = datetime.fromisoformat(paid_at.replace('Z', '+00:00'))
                except:
                    paid_date = datetime.now()
            else:
                paid_date = paid_at if hasattr(paid_at, 'year') else datetime.now()
            
            month_key = paid_date.strftime('%Y-%m')
            monthly_stats[month_key]['orders'] += 1
            monthly_stats[month_key]['revenue'] += amount_cents
            monthly_stats[month_key]['photos'] += photo_count
            monthly_stats[month_key]['packages'][package_key] += 1
            
            total_orders += 1
            total_revenue += amount_cents
        
        # Prepara dati pacchetti
        packages_data = []
        for package_key in ["1 photo", "2 photos", "3 photos", "4 photos", "5 photos", "6-11 photos", "12+ photos"]:
            if package_key in package_stats:
                stats = package_stats[package_key]
                avg_price = stats['total_revenue'] / stats['count'] if stats['count'] > 0 else 0
                percentage = (stats['count'] / total_orders * 100) if total_orders > 0 else 0
                
                # Prezzo offerto per questo pacchetto
                photo_count_num = int(package_key.split()[0]) if package_key.split()[0].isdigit() else None
                if photo_count_num:
                    if photo_count_num == 1:
                        offered_price = OFFERED_PRICES[1]
                    elif photo_count_num == 2:
                        offered_price = OFFERED_PRICES[2]
                    elif photo_count_num == 3:
                        offered_price = OFFERED_PRICES[3]
                    elif photo_count_num == 4:
                        offered_price = OFFERED_PRICES[4]
                    elif photo_count_num == 5:
                        offered_price = OFFERED_PRICES[5]
                    else:
                        offered_price = None
                elif "6-11" in package_key:
                    offered_price = OFFERED_PRICES[(6, 11)]
                else:
                    offered_price = OFFERED_PRICES[(12, 999)]
                
                packages_data.append({
                    'package': package_key,
                    'sales': stats['count'],
                    'revenue_cents': stats['total_revenue'],
                    'revenue_euros': stats['total_revenue'] / 100,
                    'avg_price_cents': avg_price,
                    'avg_price_euros': avg_price / 100,
                    'total_photos': stats['total_photos'],
                    'percentage': round(percentage, 1),
                    'offered_price_cents': offered_price,
                    'offered_price_euros': offered_price / 100 if offered_price else None,
                    'price_match': abs(avg_price - offered_price) < 10 if offered_price else None  # Tolleranza 10 centesimi
                })
        
        # Ordina per vendite (decrescente)
        packages_data.sort(key=lambda x: x['sales'], reverse=True)
        
        # Prepara dati mensili (ultimi 12 mesi)
        monthly_data = []
        current_date = datetime.now()
        # Traduzione mesi in italiano
        months_it = {
            'January': 'Gennaio', 'February': 'Febbraio', 'March': 'Marzo', 'April': 'Aprile',
            'May': 'Maggio', 'June': 'Giugno', 'July': 'Luglio', 'August': 'Agosto',
            'September': 'Settembre', 'October': 'Ottobre', 'November': 'Novembre', 'December': 'Dicembre'
        }
        for i in range(11, -1, -1):  # Ultimi 12 mesi
            month_date = current_date - timedelta(days=30 * i)
            month_key = month_date.strftime('%Y-%m')
            month_name_en = month_date.strftime('%B %Y')
            # Traduci il nome del mese
            for en, it in months_it.items():
                if month_name_en.startswith(en):
                    month_name = month_name_en.replace(en, it)
                    break
            else:
                month_name = month_name_en  # Fallback se non trovato
            
            if month_key in monthly_stats:
                stats = monthly_stats[month_key]
                monthly_data.append({
                    'month': month_key,
                    'month_name': month_name,
                    'orders': stats['orders'],
                    'revenue_cents': stats['revenue'],
                    'revenue_euros': stats['revenue'] / 100,
                    'photos': stats['photos'],
                    'packages': dict(stats['packages'])
                })
            else:
                monthly_data.append({
                    'month': month_key,
                    'month_name': month_name,
                    'orders': 0,
                    'revenue_cents': 0,
                    'revenue_euros': 0,
                    'photos': 0,
                    'packages': {}
                })
        
        # Suggerimenti
        suggestions = []
        if packages_data:
            most_sold = packages_data[0]
            least_sold = min(packages_data, key=lambda x: x['sales'])
            
            if most_sold['sales'] > 0:
                suggestions.append({
                    'type': 'success',
                    'message': f"Pacchetto più venduto: {most_sold['package']} ({most_sold['sales']} vendite, {most_sold['percentage']}%) - Considera di promuoverlo di più!"
                })
            
            if least_sold['sales'] < 3 and total_orders > 20:
                suggestions.append({
                    'type': 'warning',
                    'message': f"Pacchetto sottoperformante: {least_sold['package']} (solo {least_sold['sales']} vendite) - Considera di modificare il prezzo o rimuoverlo"
                })
            
            # Controlla se i prezzi corrispondono
            for pkg in packages_data:
                if pkg['price_match'] is False:
                    suggestions.append({
                        'type': 'info',
                        'message': f"{pkg['package']}: Prezzo medio €{pkg['avg_price_euros']:.2f} differisce da quello offerto €{pkg['offered_price_euros']:.2f} - Controlla la logica dei prezzi"
                    })
        
        return {
            "ok": True,
            "total_orders": total_orders,
            "total_revenue_cents": total_revenue,
            "total_revenue_euros": total_revenue / 100,
            "packages": packages_data,
            "monthly_trend": monthly_data,
            "suggestions": suggestions,
            "period": {
                "start_date": start_date,
                "end_date": end_date
            }
        }
    except Exception as e:
        logger.error(f"Error getting package analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/fix-all-orders")
async def admin_fix_all_orders(password: str = Query(..., description="Password admin")):
    """Aggiorna tutti gli ordini esistenti: marca tutte le foto come pagate nella tabella user_photos"""
    if not _check_admin_auth(password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        # Recupera tutti gli ordini
        orders_rows = await _db_execute("""
            SELECT order_id, email, photo_ids, amount_cents, paid_at
            FROM orders
            ORDER BY paid_at DESC
        """)
        
        results = {
            "ok": True,
            "total_orders": len(orders_rows),
            "orders_processed": 0,
            "photos_marked_paid": 0,
            "photos_already_paid": 0,
            "photos_not_found": 0,
            "errors": [],
            "details": []
        }
        
        for row in orders_rows:
            order_id = row['order_id']
            email = _normalize_email(row['email'])
            photo_ids = json.loads(row['photo_ids']) if row['photo_ids'] else []
            amount_cents = row['amount_cents']
            paid_at = row['paid_at']
            
            order_detail = {
                "order_id": order_id,
                "email": email,
                "photo_count": len(photo_ids),
                "amount_cents": amount_cents,
                "paid_at": str(paid_at),
                "photos_marked": 0,
                "photos_already_paid": 0,
                "photos_not_found": 0,
                "errors": []
            }
            
            # Per ogni foto nell'ordine, marca come pagata
            for photo_id in photo_ids:
                try:
                    # Verifica se la foto esiste in user_photos
                    existing = await _db_execute_one(
                        "SELECT status, paid_at FROM user_photos WHERE email = $1 AND photo_id = $2",
                        (email, photo_id)
                    )
                    
                    if existing:
                        # Verifica se è già pagata
                        if existing.get('status') == 'paid' and existing.get('paid_at'):
                            order_detail["photos_already_paid"] += 1
                            results["photos_already_paid"] += 1
                        else:
                            # Marca come pagata (usa la funzione esistente)
                            success = await _mark_photo_paid(email, photo_id)
                            if success:
                                order_detail["photos_marked"] += 1
                                results["photos_marked_paid"] += 1
                            else:
                                error_msg = f"Failed to mark {photo_id} as paid"
                                order_detail["errors"].append(error_msg)
                                results["errors"].append(f"Order {order_id}: {error_msg}")
                    else:
                        # La foto non esiste in user_photos, crea un record pagato
                        # Questo può succedere se l'ordine è stato creato ma la foto non è stata trovata
                        order_detail["photos_not_found"] += 1
                        results["photos_not_found"] += 1
                        # Prova comunque a marcarla come pagata (la funzione crea il record se non esiste)
                        success = await _mark_photo_paid(email, photo_id)
                        if success:
                            order_detail["photos_marked"] += 1
                            results["photos_marked_paid"] += 1
                except Exception as e:
                    error_msg = f"Error marking {photo_id} as paid: {str(e)}"
                    order_detail["errors"].append(error_msg)
                    results["errors"].append(f"Order {order_id}: {error_msg}")
                    logger.error(error_msg, exc_info=True)
            
            results["orders_processed"] += 1
            results["details"].append(order_detail)
        
        logger.info(f"Fixed all orders: {results['orders_processed']} orders, {results['photos_marked_paid']} photos marked as paid, {results['photos_already_paid']} already paid, {results['photos_not_found']} not found")
        
        return results
    except Exception as e:
        logger.error(f"Error fixing all orders: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/reset-database")
async def admin_reset_database(request: Request):
    """Reset totale del database: TRUNCATE di tutte le tabelle con RESTART IDENTITY CASCADE.
    
    Richiede header X-Admin-Secret con valore corrispondente a ADMIN_RESET_SECRET env var.
    """
    if not _check_admin_reset_secret(request):
        raise HTTPException(status_code=401, detail="Unauthorized: X-Admin-Secret header required and must match ADMIN_RESET_SECRET")
    
    try:
        # Tabelle da resettare (in ordine per evitare problemi di foreign key)
        tables = [
            "orders",
            "carts",
            "user_photos",
            "users"
        ]
        
        # Esegui TRUNCATE con RESTART IDENTITY CASCADE per ogni tabella
        for table in tables:
            await _db_execute_write(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE")
            logger.info(f"Truncated table: {table}")
        
        # Verifica conteggi finali (tutti devono essere 0)
        counts = {}
        for table in tables:
            result = await _db_execute_one(f"SELECT COUNT(*) as count FROM {table}")
            counts[table] = result['count'] if result else 0
        
        logger.warning(f"Database reset completed by admin. All tables truncated. Final counts: {counts}")
        
        return {
            "ok": True,
            "message": "Database reset completed successfully",
            "tables_truncated": tables,
            "final_counts": counts
        }
    except Exception as e:
        logger.error(f"Error resetting database: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error resetting database: {str(e)}")

# SPA Fallback: catch-all route per tutte le route frontend GET (eccetto /api/*, /static/*, /photo/*, ecc.)
# IMPORTANTE: Questa route deve essere l'ULTIMA definita, dopo tutte le altre route specifiche
@app.get("/{path:path}", response_class=HTMLResponse, include_in_schema=False)
async def spa_fallback(request: Request, path: str):
    """
    Catch-all route per SPA: serve index.html per tutte le route frontend GET.
    Esclude route API, static files, e endpoint specifici.
    Questa route deve essere l'ultima definita per non interferire con altre route.
    """
    # Route da escludere (servite da endpoint specifici)
    excluded_prefixes = [
        "/api/",
        "/static/",
        "/photo/",
        "/thumb/",
        "/wm/",
        "/download/",
        "/health",
        "/debug/",
        "/dev/",
        "/admin/",
        "/checkout/",
        "/stripe/",
        "/match_selfie",
        "/register_user",
        "/check_user",
        "/check_date",
        "/user/",
        "/my-photos",
        "/family/",
        "/cart",
        "/favicon.ico",
        "/apple-touch-icon",
        "/test",
        "/album",
    ]
    
    # Verifica se la route è esclusa
    full_path = f"/{path}" if not path.startswith("/") else path
    for excluded in excluded_prefixes:
        if full_path.startswith(excluded):
            # Route esclusa: ritorna 404
            raise HTTPException(status_code=404, detail=f"Route not found: {full_path}")
    
    # Route frontend: serve index.html (SPA fallback)
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        logger.error(f"[SPA_FALLBACK] index.html not found at: {index_path.resolve()}")
        raise HTTPException(status_code=500, detail=f"index.html not found: {index_path}")
    
    logger.info(f"[SPA_FALLBACK] Serving index.html for route: {full_path}")
    return FileResponse(
        index_path,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

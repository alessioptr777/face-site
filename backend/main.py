# File principale dell'API FaceSite
import json
import logging
import os
import hashlib
import secrets
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import cv2
import faiss

from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Request, Form
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

# SendGrid per email
try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail, Email, To, Content
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False

# Database: PostgreSQL (se DATABASE_URL presente) o SQLite (fallback)
try:
    import sqlite3
    import aiosqlite
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

# PostgreSQL support
try:
    import asyncpg
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

# Verifica se usare PostgreSQL o SQLite
DATABASE_URL = os.getenv("DATABASE_URL")
USE_POSTGRES = POSTGRES_AVAILABLE and DATABASE_URL is not None and DATABASE_URL.startswith("postgresql://")

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

# Directory per ordini e download tokens
ORDERS_DIR = DATA_DIR / "orders"
ORDERS_DIR.mkdir(parents=True, exist_ok=True)

# Database SQLite per utenti
DB_PATH = DATA_DIR / "users.db"

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

# Configurazione SendGrid (opzionale)
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
SENDGRID_FROM_EMAIL = os.getenv("SENDGRID_FROM_EMAIL", "tenerifestars.photo@gmail.com")
USE_SENDGRID = SENDGRID_AVAILABLE and bool(SENDGRID_API_KEY)

if USE_SENDGRID:
    try:
        sg_client = SendGridAPIClient(SENDGRID_API_KEY)
        logger.info("SendGrid configured - email sending enabled")
        logger.info(f"SendGrid FROM email: {SENDGRID_FROM_EMAIL}")
        # Test connessione SendGrid (opzionale, solo per debug)
        # logger.info(f"SendGrid API Key length: {len(SENDGRID_API_KEY)} characters")
    except Exception as e:
        logger.error(f"Error initializing SendGrid client: {e}")
        sg_client = None
        USE_SENDGRID = False
else:
    sg_client = None
    if not SENDGRID_AVAILABLE:
        logger.warning("SendGrid not configured - sendgrid package not available")
    elif not SENDGRID_API_KEY:
        logger.warning("SendGrid not configured - SENDGRID_API_KEY not set")
        logger.warning("Please set SENDGRID_API_KEY environment variable on Render")
    else:
        logger.warning("SendGrid not configured - email features disabled")

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
# ========== DATABASE (PostgreSQL o SQLite) ==========

def _normalize_email(email: str) -> str:
    """Normalizza l'email: minuscolo, senza spazi"""
    if not email:
        return email
    return email.strip().lower()

async def _init_database():
    """Inizializza il database (PostgreSQL o SQLite) con le tabelle necessarie"""
    if USE_POSTGRES:
        # PostgreSQL
        try:
            logger.info(f"Initializing PostgreSQL database: {DATABASE_URL[:30]}...")
            conn = await asyncpg.connect(DATABASE_URL)
            
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
            
            # Tabella email follow-up
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS email_followups (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255) NOT NULL,
                    photo_id VARCHAR(255) NOT NULL,
                    followup_type VARCHAR(50) NOT NULL,
                    sent_at TIMESTAMP,
                    FOREIGN KEY (email) REFERENCES users(email),
                    UNIQUE(email, photo_id, followup_type)
                )
            """)
            
            # Indici per performance
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_user_photos_email ON user_photos(email)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_user_photos_status ON user_photos(status)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_user_photos_expires ON user_photos(expires_at)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_email ON orders(email)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_token ON orders(download_token)")
            
            await conn.close()
            logger.info("PostgreSQL database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing PostgreSQL database: {e}", exc_info=True)
            raise
    elif SQLITE_AVAILABLE:
        # SQLite (fallback)
        try:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initializing SQLite database: {DB_PATH}")
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Tabella utenti
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    email TEXT PRIMARY KEY,
                    selfie_embedding BLOB,
                    created_at TEXT NOT NULL,
                    last_login_at TEXT,
                    last_selfie_at TEXT
                )
            """)
            
            # Tabella foto utente
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_photos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL,
                    photo_id TEXT NOT NULL,
                    found_at TEXT NOT NULL,
                    paid_at TEXT,
                    expires_at TEXT,
                    status TEXT NOT NULL DEFAULT 'found',
                    FOREIGN KEY (email) REFERENCES users(email),
                    UNIQUE(email, photo_id)
                )
            """)
            
            # Tabella ordini
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    email TEXT NOT NULL,
                    stripe_session_id TEXT,
                    photo_ids TEXT NOT NULL,
                    amount_cents INTEGER NOT NULL,
                    paid_at TEXT NOT NULL,
                    expires_at TEXT,
                    download_token TEXT UNIQUE,
                    FOREIGN KEY (email) REFERENCES users(email)
                )
            """)
            
            # Tabella email follow-up
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS email_followups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL,
                    photo_id TEXT NOT NULL,
                    followup_type TEXT NOT NULL,
                    sent_at TEXT,
                    FOREIGN KEY (email) REFERENCES users(email),
                    UNIQUE(email, photo_id, followup_type)
                )
            """)
            
            # Indici per performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_photos_email ON user_photos(email)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_photos_status ON user_photos(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_photos_expires ON user_photos(expires_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_email ON orders(email)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_token ON orders(download_token)")
            
            conn.commit()
            conn.close()
            logger.info(f"SQLite database initialized: {DB_PATH}")
        except Exception as e:
            logger.error(f"Error initializing SQLite database: {e}")
    else:
        logger.warning("No database available - user tracking disabled")

# Helper functions per database (PostgreSQL o SQLite)
async def _db_execute(query: str, params: tuple = ()):
    """Esegue una query e restituisce i risultati (adatta per PostgreSQL o SQLite)"""
    if USE_POSTGRES:
        # Se la query contiene già $1, $2, etc., non convertire (è già formattata per PostgreSQL)
        # Altrimenti, converti ? in $1, $2, ... per PostgreSQL
        if '$' in query and any(f'${i}' in query for i in range(1, 10)):
            # Query già formattata per PostgreSQL, usa direttamente
            pg_query = query
        else:
            # Converti ? in $1, $2, ... per PostgreSQL
            pg_query = query
            param_count = query.count('?')
            for i in range(1, param_count + 1):
                pg_query = pg_query.replace('?', f'${i}', 1)
        
        conn = await asyncpg.connect(DATABASE_URL)
        try:
            rows = await conn.fetch(pg_query, *params)
            return [dict(row) for row in rows]
        finally:
            await conn.close()
    elif SQLITE_AVAILABLE:
        async with aiosqlite.connect(DB_PATH) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
    return []

async def _db_execute_one(query: str, params: tuple = ()):
    """Esegue una query e restituisce un solo risultato"""
    rows = await _db_execute(query, params)
    return rows[0] if rows else None

async def _db_execute_write(query: str, params: tuple = ()):
    """Esegue una query di scrittura (INSERT, UPDATE, DELETE)"""
    if USE_POSTGRES:
        # Se la query contiene già $1, $2, etc., non convertire (è già formattata per PostgreSQL)
        # Altrimenti, converti ? in $1, $2, ... per PostgreSQL
        if '$' in query and any(f'${i}' in query for i in range(1, 10)):
            # Query già formattata per PostgreSQL, usa direttamente
            pg_query = query
        else:
            # Converti ? in $1, $2, ... per PostgreSQL
            pg_query = query
            param_count = query.count('?')
            for i in range(1, param_count + 1):
                pg_query = pg_query.replace('?', f'${i}', 1)
        
        conn = await asyncpg.connect(DATABASE_URL)
        try:
            await conn.execute(pg_query, *params)
        finally:
            await conn.close()
    elif SQLITE_AVAILABLE:
        async with aiosqlite.connect(DB_PATH) as conn:
            await conn.execute(query, params)
            await conn.commit()

async def _get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Recupera un utente per email"""
    if not USE_POSTGRES and not SQLITE_AVAILABLE:
        return None
    
    try:
        email = _normalize_email(email)
        row = await _db_execute_one(
            "SELECT * FROM users WHERE email = ?",
            (email,)
        )
        return row
    except Exception as e:
        logger.error(f"Error getting user: {e}")
    return None

async def _create_or_update_user(email: str, selfie_embedding: Optional[bytes] = None) -> bool:
    """Crea o aggiorna un utente (salva solo con email, selfie opzionale)"""
    if not USE_POSTGRES and not SQLITE_AVAILABLE:
        return False
    
    try:
        email = _normalize_email(email)
        
        # Verifica se esiste
        exists = await _db_execute_one(
            "SELECT email FROM users WHERE email = ?",
            (email,)
        )
        
        if exists:
            # Aggiorna
            if USE_POSTGRES:
                # PostgreSQL: usa NOW() per evitare problemi con timezone
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
                # SQLite: usa parametro
                now = datetime.now(timezone.utc).isoformat()
                if selfie_embedding:
                    await _db_execute_write("""
                        UPDATE users 
                        SET selfie_embedding = ?, last_login_at = ?, last_selfie_at = ?
                        WHERE email = ?
                    """, (selfie_embedding, now, now, email))
                else:
                    # Aggiorna solo last_login_at (non sovrascrivere selfie_embedding esistente)
                    await _db_execute_write("""
                        UPDATE users 
                        SET last_login_at = ?
                        WHERE email = ?
                    """, (now, email))
        else:
            # Crea nuovo utente (solo con email, selfie opzionale)
            if USE_POSTGRES:
                # PostgreSQL: usa NOW() per evitare problemi con timezone
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
            else:
                # SQLite: usa parametro
                now = datetime.now(timezone.utc).isoformat()
                await _db_execute_write("""
                    INSERT INTO users (email, selfie_embedding, created_at, last_login_at, last_selfie_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (email, selfie_embedding, now, now, now if selfie_embedding else None))
        
        return True
    except Exception as e:
        logger.error(f"Error creating/updating user: {e}")
    return False

async def _add_user_photo(email: str, photo_id: str, status: str = "found") -> bool:
    """Aggiunge una foto trovata per un utente"""
    if not USE_POSTGRES and not SQLITE_AVAILABLE:
        return False
    
    try:
        email = _normalize_email(email)
        
        # Verifica se esiste già
        if USE_POSTGRES:
            exists = await _db_execute_one(
                "SELECT id FROM user_photos WHERE email = $1 AND photo_id = $2",
                (email, photo_id)
            )
        else:
            exists = await _db_execute_one(
                "SELECT id FROM user_photos WHERE email = ? AND photo_id = ?",
                (email, photo_id)
            )
        
        if USE_POSTGRES:
            # PostgreSQL: usa NOW() e INTERVAL per evitare problemi con timezone
            days = 30 if status == "paid" else 90
            if exists:
                # Aggiorna - usa f-string per INTERVAL
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
        else:
            # SQLite: usa parametro
            now = datetime.now(timezone.utc).isoformat()
            expires_at = (datetime.now(timezone.utc) + timedelta(days=30 if status == "paid" else 90)).isoformat()
            if exists:
                # Aggiorna
                await _db_execute_write("""
                    UPDATE user_photos 
                    SET found_at = ?, status = ?, expires_at = ?
                    WHERE email = ? AND photo_id = ?
                """, (now, status, expires_at, email, photo_id))
            else:
                # Inserisci nuovo
                await _db_execute_write("""
                    INSERT INTO user_photos (email, photo_id, found_at, status, expires_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (email, photo_id, now, status, expires_at))
        
        return True
    except Exception as e:
        logger.error(f"Error adding user photo: {e}")
    return False

async def _mark_photo_paid(email: str, photo_id: str) -> bool:
    """Marca una foto come pagata (crea record se non esiste)"""
    if not USE_POSTGRES and not SQLITE_AVAILABLE:
        return False
    
    try:
        email = _normalize_email(email)
        
        # Verifica se esiste già
        if USE_POSTGRES:
            exists = await _db_execute_one("""
                SELECT id FROM user_photos 
                WHERE email = $1 AND photo_id = $2
            """, (email, photo_id))
        else:
            exists = await _db_execute_one("""
                SELECT id FROM user_photos 
                WHERE email = ? AND photo_id = ?
            """, (email, photo_id))
        
        if USE_POSTGRES:
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
        else:
            # SQLite: usa parametro
            now = datetime.now(timezone.utc).isoformat()
            expires_at = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
            if exists:
                # Aggiorna record esistente
                await _db_execute_write("""
                    UPDATE user_photos 
                    SET paid_at = ?, status = 'paid', expires_at = ?
                    WHERE email = ? AND photo_id = ?
                """, (now, expires_at, email, photo_id))
            else:
                # Crea nuovo record (foto pagata senza essere stata trovata prima)
                await _db_execute_write("""
                    INSERT INTO user_photos (email, photo_id, found_at, paid_at, status, expires_at)
                    VALUES (?, ?, ?, ?, 'paid', ?)
                """, (email, photo_id, now, now, expires_at))
        
        logger.info(f"✅ Photo marked as paid: {email} - {photo_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Error marking photo paid: {e}", exc_info=True)
        logger.error(f"Exception type: {type(e).__name__}")
    return False

async def _get_user_paid_photos(email: str) -> List[str]:
    """Recupera lista foto pagate per un utente (non scadute)"""
    if not USE_POSTGRES and not SQLITE_AVAILABLE:
        logger.warning(f"Database not available, cannot get paid photos for {email}")
        return []
    
    try:
        email = _normalize_email(email)
        if USE_POSTGRES:
            now = datetime.now(timezone.utc)
        else:
            now = datetime.now(timezone.utc).isoformat()
        
        # Prima verifica tutte le foto per questo utente (per debug)
        all_rows = await _db_execute("""
            SELECT photo_id, status, expires_at FROM user_photos 
            WHERE email = ?
        """, (email,))
        logger.info(f"All photos for {email}: {len(all_rows)} total")
        for row in all_rows:
            logger.info(f"  - {row['photo_id']}: status={row['status']}, expires_at={row['expires_at']}")
        
        # Poi recupera solo quelle pagate e non scadute
        if USE_POSTGRES:
            # PostgreSQL: usa NOW() per evitare problemi con timezone
            rows = await _db_execute("""
                SELECT photo_id FROM user_photos 
                WHERE email = $1 AND status = 'paid' AND expires_at > NOW()
            """, (email,))
        else:
            # SQLite: usa parametro
            rows = await _db_execute("""
                SELECT photo_id FROM user_photos 
                WHERE email = ? AND status = 'paid' AND expires_at > ?
            """, (email, now))
        photo_ids = [row['photo_id'] for row in rows]
        logger.info(f"Paid photos (not expired) for {email}: {len(photo_ids)} photos - {photo_ids}")
        return photo_ids
    except Exception as e:
        logger.error(f"Error getting paid photos for {email}: {e}", exc_info=True)
    return []

async def _get_user_found_photos(email: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Recupera tutte le foto trovate per un utente (con limite opzionale per performance)"""
    if not USE_POSTGRES and not SQLITE_AVAILABLE:
        return []
    
    try:
        email = _normalize_email(email)
        # Limite default di 200 foto per evitare problemi di performance
        # Se serve di più, si può aumentare o implementare paginazione
        limit_clause = f"LIMIT {limit}" if limit else "LIMIT 200"
        
        if USE_POSTGRES:
            rows = await _db_execute(f"""
                SELECT photo_id, found_at, paid_at, expires_at, status 
                FROM user_photos 
                WHERE email = $1
                ORDER BY found_at DESC
                {limit_clause}
            """, (email,))
        else:
            rows = await _db_execute(f"""
                SELECT photo_id, found_at, paid_at, expires_at, status 
                FROM user_photos 
                WHERE email = ?
                ORDER BY found_at DESC
                {limit_clause}
            """, (email,))
        return rows
    except Exception as e:
        logger.error(f"Error getting found photos: {e}")
    return []

async def _match_selfie_embedding(selfie_embedding: bytes, threshold: float = 0.7) -> Optional[str]:
    """Confronta selfie embedding con quelli salvati, ritorna email se match"""
    if not USE_POSTGRES and not SQLITE_AVAILABLE:
        return None
    
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
    if not USE_POSTGRES and not SQLITE_AVAILABLE:
        logger.error("Database not available, cannot create order")
        return None
    
    try:
        email = _normalize_email(email)
        logger.info(f"_create_order called: email={email}, order_id={order_id}, photo_count={len(photo_ids)}")
        download_token = secrets.token_urlsafe(32)
        logger.info(f"Generated download_token: {download_token[:20]}...")
        
        if USE_POSTGRES:
            # PostgreSQL: usa NOW() per evitare problemi con timezone
            # paid_at ha DEFAULT CURRENT_TIMESTAMP, ma lo passiamo esplicitamente per chiarezza
            await _db_execute_write("""
                INSERT INTO orders (order_id, email, stripe_session_id, photo_ids, amount_cents, paid_at, download_token)
                VALUES ($1, $2, $3, $4, $5, NOW(), $6)
            """, (order_id, email, stripe_session_id, json.dumps(photo_ids), amount_cents, download_token))
        else:
            # SQLite: usa parametro
            now = datetime.now(timezone.utc).isoformat()
            await _db_execute_write("""
                INSERT INTO orders (order_id, email, stripe_session_id, photo_ids, amount_cents, paid_at, download_token)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (order_id, email, stripe_session_id, json.dumps(photo_ids), amount_cents, now, download_token))
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
    if not USE_POSTGRES and not SQLITE_AVAILABLE:
        return None
    
    try:
        row = await _db_execute_one(
            "SELECT * FROM orders WHERE download_token = ?",
            (token,)
        )
        if row:
            order = dict(row)
            order['photo_ids'] = json.loads(order['photo_ids'])
            return order
    except Exception as e:
        logger.error(f"Error getting order: {e}")
    return None

async def _get_photos_for_followup() -> List[Dict[str, Any]]:
    """Recupera foto che necessitano follow-up email"""
    if not USE_POSTGRES and not SQLITE_AVAILABLE:
        return []
    
    try:
        now = datetime.now(timezone.utc)
        if USE_POSTGRES:
            # PostgreSQL usa EXTRACT(EPOCH FROM ...) per differenze di date
            query = """
                SELECT email, photo_id, found_at, 
                       CASE 
                           WHEN EXTRACT(EPOCH FROM (? - found_at)) / 86400 BETWEEN 6 AND 8 THEN '7days'
                           WHEN EXTRACT(EPOCH FROM (? - found_at)) / 86400 BETWEEN 29 AND 31 THEN '30days'
                           WHEN EXTRACT(EPOCH FROM (? - found_at)) / 86400 BETWEEN 59 AND 61 THEN '60days'
                       END as followup_type
                FROM user_photos
                WHERE status = 'found'
                AND (EXTRACT(EPOCH FROM (? - found_at)) / 86400 BETWEEN 6 AND 8
                     OR EXTRACT(EPOCH FROM (? - found_at)) / 86400 BETWEEN 29 AND 31
                     OR EXTRACT(EPOCH FROM (? - found_at)) / 86400 BETWEEN 59 AND 61)
                AND NOT EXISTS (
                    SELECT 1 FROM email_followups 
                    WHERE email_followups.email = user_photos.email 
                    AND email_followups.photo_id = user_photos.photo_id
                    AND email_followups.followup_type = 
                        CASE 
                            WHEN EXTRACT(EPOCH FROM (? - user_photos.found_at)) / 86400 BETWEEN 6 AND 8 THEN '7days'
                            WHEN EXTRACT(EPOCH FROM (? - user_photos.found_at)) / 86400 BETWEEN 29 AND 31 THEN '30days'
                            WHEN EXTRACT(EPOCH FROM (? - user_photos.found_at)) / 86400 BETWEEN 59 AND 61 THEN '60days'
                        END
                )
            """
            params = (now,) * 8
        else:
            # SQLite usa julianday
            query = """
                SELECT email, photo_id, found_at, 
                       CASE 
                           WHEN julianday(?) - julianday(found_at) BETWEEN 6 AND 8 THEN '7days'
                           WHEN julianday(?) - julianday(found_at) BETWEEN 29 AND 31 THEN '30days'
                           WHEN julianday(?) - julianday(found_at) BETWEEN 59 AND 61 THEN '60days'
                       END as followup_type
                FROM user_photos
                WHERE status = 'found'
                AND (julianday(?) - julianday(found_at) BETWEEN 6 AND 8
                     OR julianday(?) - julianday(found_at) BETWEEN 29 AND 31
                     OR julianday(?) - julianday(found_at) BETWEEN 59 AND 61)
                AND NOT EXISTS (
                    SELECT 1 FROM email_followups 
                    WHERE email_followups.email = user_photos.email 
                    AND email_followups.photo_id = user_photos.photo_id
                    AND email_followups.followup_type = 
                        CASE 
                            WHEN julianday(?) - julianday(user_photos.found_at) BETWEEN 6 AND 8 THEN '7days'
                            WHEN julianday(?) - julianday(user_photos.found_at) BETWEEN 29 AND 31 THEN '30days'
                            WHEN julianday(?) - julianday(user_photos.found_at) BETWEEN 59 AND 61 THEN '60days'
                        END
                )
            """
            params = (now.isoformat(),) * 8
        
        rows = await _db_execute(query, params)
        return [row for row in rows if row.get('followup_type')]
    except Exception as e:
        logger.error(f"Error getting photos for followup: {e}")
    return []

async def _mark_followup_sent(email: str, photo_id: str, followup_type: str):
    """Marca follow-up email come inviata"""
    if not USE_POSTGRES and not SQLITE_AVAILABLE:
        return
    
    try:
        if USE_POSTGRES:
            # PostgreSQL: usa NOW() per evitare problemi con timezone
            await _db_execute_write("""
                INSERT INTO email_followups (email, photo_id, followup_type, sent_at)
                VALUES ($1, $2, $3, NOW())
                ON CONFLICT (email, photo_id, followup_type) 
                DO UPDATE SET sent_at = NOW()
            """, (email, photo_id, followup_type))
        else:
            # SQLite: usa parametro
            now = datetime.now(timezone.utc).isoformat()
            await _db_execute_write("""
                INSERT OR REPLACE INTO email_followups (email, photo_id, followup_type, sent_at)
                VALUES (?, ?, ?, ?)
            """, (email, photo_id, followup_type, now))
    except Exception as e:
        logger.error(f"Error marking followup sent: {e}")

async def _send_followup_emails():
    """Invia email follow-up per foto non pagate"""
    if not USE_SENDGRID:
        return
    
    try:
        photos_for_followup = await _get_photos_for_followup()
        
        if not photos_for_followup:
            return
        
        # Raggruppa per email e followup_type
        from collections import defaultdict
        grouped = defaultdict(lambda: defaultdict(list))
        
        for item in photos_for_followup:
            email = item['email']
            followup_type = item['followup_type']
            photo_id = item['photo_id']
            grouped[email][followup_type].append(photo_id)
        
        base_url = os.getenv("BASE_URL", "https://face-site.onrender.com")
        
        sent_count = 0
        for email, followups in grouped.items():
            for followup_type, photo_ids in followups.items():
                try:
                    success = await _send_followup_email(email, photo_ids, followup_type, base_url)
                    if success:
                        for photo_id in photo_ids:
                            await _mark_followup_sent(email, photo_id, followup_type)
                        sent_count += 1
                except Exception as e:
                    logger.error(f"Error sending followup to {email}: {e}")
        
        if sent_count > 0:
            logger.info(f"Sent {sent_count} follow-up emails")
    except Exception as e:
        logger.error(f"Error in follow-up email task: {e}")

async def _cleanup_expired_photos():
    """Elimina foto scadute dal database e dal filesystem"""
    if not USE_POSTGRES and not SQLITE_AVAILABLE:
        return
    
    try:
        # Trova foto scadute
        if USE_POSTGRES:
            # PostgreSQL: usa NOW() per evitare problemi con timezone
            expired = await _db_execute("""
                SELECT email, photo_id, status FROM user_photos
                WHERE expires_at < NOW() AND status != 'deleted'
            """, ())
        else:
            # SQLite: usa parametro
            now = datetime.now(timezone.utc).isoformat()
            expired = await _db_execute("""
                SELECT email, photo_id, status FROM user_photos
                WHERE expires_at < ? AND status != 'deleted'
            """, (now,))
        
        deleted_count = 0
        for row in expired:
            email = row['email']
            photo_id = row['photo_id']
            # Marca come deleted nel database
            await _db_execute_write("""
                UPDATE user_photos SET status = 'deleted' 
                WHERE email = ? AND photo_id = ?
            """, (email, photo_id))
            
            # Elimina file fisico (se locale, non Cloudinary)
            if not USE_CLOUDINARY:
                photo_path = PHOTOS_DIR / photo_id
                if photo_path.exists():
                    try:
                        photo_path.unlink()
                        deleted_count += 1
                        logger.info(f"Deleted expired photo: {photo_id}")
                    except Exception as e:
                        logger.error(f"Error deleting photo file {photo_id}: {e}")
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired photos ({deleted_count} files deleted)")
    except Exception as e:
        logger.error(f"Error cleaning up expired photos: {e}")

# ========== FUNZIONI EMAIL SENDGRID ==========

async def _send_email(to_email: str, subject: str, html_content: str, plain_content: str = None) -> bool:
    """Invia email tramite SendGrid"""
    if not USE_SENDGRID or not sg_client:
        logger.warning(f"SendGrid not available - email not sent to {to_email}")
        logger.warning(f"USE_SENDGRID={USE_SENDGRID}, sg_client={sg_client is not None}")
        logger.warning(f"SENDGRID_AVAILABLE={SENDGRID_AVAILABLE}, SENDGRID_API_KEY set={bool(SENDGRID_API_KEY)}")
        return False
    
    try:
        logger.info(f"=== SENDING EMAIL ===")
        logger.info(f"To: {to_email}")
        logger.info(f"From: {SENDGRID_FROM_EMAIL}")
        logger.info(f"Subject: {subject}")
        
        message = Mail(
            from_email=Email(SENDGRID_FROM_EMAIL, "Tenerife Stars Pictures"),
            to_emails=To(to_email),
            subject=subject,
            html_content=Content("text/html", html_content)
        )
        
        if plain_content:
            message.plain_text_content = Content("text/plain", plain_content)
        
        logger.info("Calling SendGrid API...")
        response = sg_client.send(message)
        logger.info(f"SendGrid response: status_code={response.status_code}")
        logger.info(f"SendGrid response headers: {dict(response.headers)}")
        
        if response.status_code in [200, 201, 202]:
            logger.info(f"✅ Email sent successfully to {to_email}")
            return True
        else:
            logger.error(f"❌ SendGrid error: status_code={response.status_code}")
            logger.error(f"SendGrid response body: {response.body}")
            return False
    except Exception as e:
        logger.error(f"❌ Error sending email to {to_email}: {e}", exc_info=True)
        logger.error(f"Exception type: {type(e).__name__}")
        return False

async def _send_payment_confirmation_email(email: str, photo_ids: List[str], download_token: str, base_url: str):
    """Invia email di conferma pagamento"""
    download_url = f"{base_url}/my-photos/{download_token}"
    expires_at = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    expires_date = datetime.fromisoformat(expires_at.replace('Z', '+00:00')).strftime("%d/%m/%Y")
    
    subject = "Le tue foto sono pronte! (Disponibili per 30 giorni)"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background: linear-gradient(135deg, #7b74ff, #5f58ff); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
            .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
            .button {{ display: inline-block; background: #7b74ff; color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; margin: 20px 0; }}
            .warning {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 20px 0; }}
            .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>✅ Pagamento completato!</h1>
            </div>
            <div class="content">
                <p>Ciao,</p>
                <p>Il tuo pagamento è stato confermato con successo!</p>
                <p>Hai acquistato <strong>{len(photo_ids)} foto</strong> e ora puoi scaricarle in alta qualità.</p>
                
                <div style="text-align: center;">
                    <a href="{download_url}" class="button">Scarica le tue foto</a>
                </div>
                
                <div class="warning">
                    <strong>⚠️ IMPORTANTE:</strong><br>
                    Le tue foto saranno disponibili per <strong>30 giorni</strong> (fino al {expires_date}).<br>
                    Assicurati di scaricarle nella tua galleria prima della scadenza!
                </div>
                
                <p>Puoi anche accedere alle tue foto usando questo link permanente:</p>
                <p style="background: #e9ecef; padding: 10px; border-radius: 5px; word-break: break-all;">
                    {download_url}
                </p>
                
                <p>Salva questo link per recuperare le tue foto in futuro.</p>
            </div>
            <div class="footer">
                <p>Tenerife Stars Pictures</p>
                <p>Se hai domande, rispondi a questa email.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    plain_content = f"""
Pagamento completato!

Hai acquistato {len(photo_ids)} foto e ora puoi scaricarle.

Link per scaricare: {download_url}

⚠️ IMPORTANTE: Le foto saranno disponibili per 30 giorni (fino al {expires_date}).
Assicurati di scaricarle prima della scadenza!

Salva questo link per recuperare le tue foto in futuro.
    """
    
    result = await _send_email(email, subject, html_content, plain_content)
    if result:
        logger.info(f"Payment confirmation email sent to {email} with {len(photo_ids)} photos")
    else:
        logger.error(f"Failed to send payment confirmation email to {email}")
    return result

async def _send_followup_email(email: str, photo_ids: List[str], followup_type: str, base_url: str):
    """Invia email follow-up per foto non pagate"""
    days_text = {
        '7days': ('7 giorni fa', 'ancora 83 giorni'),
        '30days': ('30 giorni fa', 'ancora 60 giorni'),
        '60days': ('60 giorni fa', 'ancora 30 giorni')
    }
    
    when_text, remaining_text = days_text.get(followup_type, ('', ''))
    
    if followup_type == '7days':
        subject = "Non perdere le tue foto! Hai ancora tempo per acquistarle"
        urgency = "Non perdere questa opportunità!"
    elif followup_type == '30days':
        subject = "Ultimi giorni! Le tue foto verranno eliminate tra 60 giorni"
        urgency = "Tempo limitato!"
    else:  # 60days
        subject = "Ultima possibilità! Le foto verranno eliminate tra 30 giorni"
        urgency = "ULTIMA CHANCE!"
    
    cart_url = f"{base_url}/?email={email}"
    price = calculate_price(len(photo_ids))
    price_euros = price / 100.0
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background: linear-gradient(135deg, #7b74ff, #5f58ff); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
            .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
            .button {{ display: inline-block; background: #7b74ff; color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; margin: 20px 0; }}
            .warning {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 20px 0; }}
            .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{urgency}</h1>
            </div>
            <div class="content">
                <p>Ciao,</p>
                <p>Hai trovato <strong>{len(photo_ids)} foto</strong> {when_text} e non le hai ancora acquistate.</p>
                <p>Hai {remaining_text} per acquistarle prima che vengano eliminate.</p>
                
                <div style="text-align: center; background: white; padding: 20px; border-radius: 8px; margin: 20px 0;">
                    <p style="font-size: 24px; margin: 0;"><strong>€{price_euros:.2f}</strong></p>
                    <p style="margin: 10px 0;">per tutte le {len(photo_ids)} foto</p>
                </div>
                
                <div style="text-align: center;">
                    <a href="{cart_url}" class="button">Acquista le tue foto</a>
                </div>
                
                <div class="warning">
                    <strong>⏰ Attenzione:</strong><br>
                    Le foto verranno eliminate automaticamente se non acquistate entro {remaining_text}.
                </div>
            </div>
            <div class="footer">
                <p>Tenerife Stars Pictures</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    plain_content = f"""
{urgency}

Hai trovato {len(photo_ids)} foto {when_text} e non le hai ancora acquistate.
Hai {remaining_text} per acquistarle.

Prezzo: €{price_euros:.2f} per tutte le {len(photo_ids)} foto

Link per acquistare: {cart_url}

⏰ Attenzione: Le foto verranno eliminate se non acquistate entro {remaining_text}.
    """
    
    return await _send_email(email, subject, html_content, plain_content)

# ========== FUNZIONI HELPER ESISTENTI ==========

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
    """Aggiunge watermark pattern a griglia come getpica.com con 'tenerifepictures'"""
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
        
        # Testo watermark (minuscolo senza spazio)
        text = "tenerifepictures"
        
        # Calcola dimensione font basata sull'altezza immagine (circa 2-3% per pattern a griglia)
        font_size = max(16, int(img.height * 0.025))
        
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
        
        # Colore bianco semi-trasparente come getpica.com (RGBA)
        # Opacità ~40-50% per essere visibile ma non invasivo
        watermark_color = (255, 255, 255, 120)  # Opacità 120/255 (~47% visibile)
        
        # Calcola dimensione celle griglia
        # Ogni cella deve contenere il testo con un po' di padding
        cell_padding = text_width * 0.3  # 30% di padding intorno al testo
        cell_width = text_width + cell_padding
        cell_height = text_height + cell_padding
        
        # Calcola quante celle servono per coprire tutta l'immagine
        num_cols = int((img.width / cell_width) + 2)  # +2 per margine
        num_rows = int((img.height / cell_height) + 2)  # +2 per margine
        
        # Disegna pattern a griglia
        # Ogni cella contiene il testo centrato
        for row in range(num_rows):
            for col in range(num_cols):
                # Calcola posizione centro cella
                cell_x = col * cell_width
                cell_y = row * cell_height
                
                # Centra il testo nella cella
                text_x = cell_x + (cell_width - text_width) / 2
                text_y = cell_y + (cell_height - text_height) / 2
                
                # Disegna solo se la cella è visibile nell'immagine
                if text_x + text_width > -50 and text_x < img.width + 50 and \
                   text_y + text_height > -50 and text_y < img.height + 50:
                    draw.text((text_x, text_y), text, font=font, fill=watermark_color)
        
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
    
    # Inizializza database (PostgreSQL o SQLite) all'avvio
    logger.info("Initializing database...")
    if USE_POSTGRES:
        logger.info("Using PostgreSQL database")
    elif SQLITE_AVAILABLE:
        logger.info("Using SQLite database (fallback)")
    else:
        logger.warning("No database available")
    
    await _init_database()
    
    # Verifica che il database sia stato creato
    if USE_POSTGRES:
        logger.info("✅ PostgreSQL database initialized")
    elif SQLITE_AVAILABLE and DB_PATH.exists():
        logger.info(f"✅ SQLite database verified: {DB_PATH} (size: {DB_PATH.stat().st_size} bytes)")
    elif SQLITE_AVAILABLE:
        logger.warning(f"⚠️  SQLite database file not found: {DB_PATH}")
    else:
        logger.warning("⚠️  SQLite not available")
    
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
    
    # Esegui cleanup iniziale
    logger.info("Running initial cleanup...")
    await _cleanup_expired_photos()
    
    # Avvia task periodico per cleanup e follow-up (ogni 6 ore)
    import asyncio
    async def periodic_tasks():
        while True:
            try:
                await asyncio.sleep(6 * 60 * 60)  # 6 ore
                logger.info("Running periodic cleanup and follow-up...")
                await _cleanup_expired_photos()
                await _send_followup_emails()
            except Exception as e:
                logger.error(f"Error in periodic tasks: {e}")
    
    # Avvia task in background
    asyncio.create_task(periodic_tasks())
    logger.info("Periodic tasks started (cleanup every 6 hours)")
    
    try:
        logger.info(f"Loading FAISS index from {INDEX_PATH}")
        faiss_index = faiss.read_index(str(INDEX_PATH))
        logger.info(f"FAISS index loaded: {faiss_index.ntotal} vectors")
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        faiss_index = None
        # Non fare return qui - continua a caricare metadata e back_photos
    
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
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        logger.error(f"index.html not found at: {index_path.resolve()}")
        raise HTTPException(status_code=500, detail=f"index.html not found: {index_path}")
    logger.info(f"Serving index.html from: {index_path.resolve()}")
    return FileResponse(index_path)

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
            "database_type": "SQLite" if SQLITE_AVAILABLE else "None"
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
    if not USE_POSTGRES and not SQLITE_AVAILABLE:
        return {"error": "Database not available", "orders": [], "user_photos": []}
    
    try:
        if email:
            email = _normalize_email(email)
        
        # Recupera tutti gli ordini
        if email:
            orders_rows = await _db_execute(
                "SELECT order_id, email, stripe_session_id, photo_ids, amount_cents, paid_at, download_token FROM orders WHERE email = ? ORDER BY paid_at DESC LIMIT 50",
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
                "SELECT email, photo_id, status, found_at, paid_at, expires_at FROM user_photos WHERE email = ? ORDER BY paid_at DESC, found_at DESC LIMIT 100",
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
    if not USE_POSTGRES and not SQLITE_AVAILABLE:
        return {"error": "Database not available"}
    
    try:
        email = _normalize_email(email)
        logger.info(f"Fixing paid photos for {email}")
        
        # Recupera tutti gli ordini per questa email
        orders_rows = await _db_execute(
            "SELECT order_id, photo_ids FROM orders WHERE email = ?",
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

@app.get("/photo/{filename:path}")
async def serve_photo(
    filename: str, 
    request: Request,
    paid: bool = Query(False, description="Se true, serve foto senza watermark (solo se pagata)"),
    token: Optional[str] = Query(None, description="Download token per verificare pagamento"),
    email: Optional[str] = Query(None, description="Email utente per verificare pagamento"),
    download: bool = Query(False, description="Se true, forza il download con header Content-Disposition")
):
    """Endpoint per servire le foto - con watermark se non pagata"""
    logger.info(f"=== PHOTO REQUEST ===")
    logger.info(f"Request path: {request.url.path}")
    logger.info(f"Filename parameter: {filename}")
    logger.info(f"Paid: {paid}, Token: {token is not None}, Email: {email is not None}")
    
    # Verifica se la foto è pagata usando token o email
    is_paid = paid
    if not is_paid and token:
        # Verifica token
        order = await _get_order_by_token(token)
        if order and filename in order.get('photo_ids', []):
            is_paid = True
            logger.info(f"Photo verified as paid via token: {filename}")
    elif not is_paid and email and SQLITE_AVAILABLE:
        # Verifica nel database se la foto è pagata per questo utente
        try:
            paid_photos = await _get_user_paid_photos(email)
            if filename in paid_photos:
                is_paid = True
                logger.info(f"Photo verified as paid via email: {filename}")
        except Exception as e:
            logger.error(f"Error checking paid photos: {e}")
    
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
                
                # Se la foto è pagata, scarica da Cloudinary e serviamo dal server
                # (così possiamo controllare watermark e tracciare download)
                if is_paid:
                    logger.info(f"Photo is paid, downloading from Cloudinary to serve without watermark")
                    import requests
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        # Servi la foto scaricata da Cloudinary (senza watermark)
                        logger.info(f"Serving paid photo from Cloudinary (no watermark)")
                        _track_download(filename)
                        
                        # Se download=true, forza il download con header Content-Disposition
                        if download:
                            headers = {
                                "Content-Disposition": f'attachment; filename="{filename}"',
                                "Content-Type": "image/jpeg"
                            }
                            return Response(content=response.content, headers=headers, media_type="image/jpeg")
                        
                        return Response(content=response.content, media_type="image/jpeg")
                    else:
                        logger.warning(f"Failed to download from Cloudinary: {response.status_code}")
                        # Fallback a locale
                else:
                    # Se non pagata, redirect a Cloudinary (più veloce)
                    logger.info(f"Photo not paid, redirecting to Cloudinary")
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
    if not is_paid:
        logger.info(f"Serving photo with watermark: {filename}")
        watermarked_bytes = _add_watermark(photo_path)
        return Response(content=watermarked_bytes, media_type="image/jpeg")
    
    # Se pagata, serve originale (ma traccia download solo se pagata)
    logger.info(f"Returning original file (paid): {resolved_path}")
    _track_download(filename)
    
    # Se download=true, forza il download con header Content-Disposition
    if download:
        with open(resolved_path, 'rb') as f:
            content = f.read()
        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Type": "image/jpeg"
        }
        return Response(content=content, headers=headers, media_type="image/jpeg")
    
    return FileResponse(resolved_path)

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
        
        # Conta foto di spalle per questa data
        photos_back = 0
        for back_photo in back_photos:
            photo_tour_date = back_photo.get("tour_date")
            if photo_tour_date and normalized_date in str(photo_tour_date):
                photos_back += 1
        
        total_photos = photos_with_faces + photos_back
        
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
            "photos_with_faces": photos_with_faces,
            "photos_back": photos_back
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
    """Registra un utente solo con email (senza selfie)"""
    try:
        # Normalizza email
        email = _normalize_email(email)
        
        # Valida email
        import re
        if not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email):
            raise HTTPException(status_code=400, detail="Invalid email format")
        
        # Salva/aggiorna utente (solo con email)
        success = await _create_or_update_user(email, None)
        
        if not success:
            raise HTTPException(status_code=500, detail="Error saving user")
        
        logger.info(f"User registered/updated: {email}")
        
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
    """Recupera tutte le foto di un utente (trovate e pagate)"""
    try:
        original_email = email
        email = _normalize_email(email)
        logger.info(f"User photos request - original: '{original_email}', normalized: '{email}'")
        
        found_photos = await _get_user_found_photos(email)
        paid_photos = await _get_user_paid_photos(email)
        
        logger.info(f"User photos request for {email}: found={len(found_photos)}, paid={len(paid_photos)}")
        
        # FALLBACK: Se non ci sono foto pagate ma ci sono ordini, estrai i photo_ids dagli ordini
        if not paid_photos or len(paid_photos) == 0:
            logger.warning(f"No paid photos found for {email} - checking orders as fallback...")
            if USE_POSTGRES or SQLITE_AVAILABLE:
                try:
                    # Recupera tutti gli ordini per questa email
                    orders_rows = await _db_execute(
                        "SELECT photo_ids FROM orders WHERE email = ? ORDER BY paid_at DESC",
                        (email,)
                    )
                    
                    if orders_rows:
                        logger.info(f"Found {len(orders_rows)} orders for {email} - extracting photo_ids")
                        # Raccogli tutti i photo_ids dagli ordini
                        all_paid_photo_ids = set()
                        for row in orders_rows:
                            photo_ids = json.loads(row['photo_ids']) if row['photo_ids'] else []
                            all_paid_photo_ids.update(photo_ids)
                        
                        if all_paid_photo_ids:
                            logger.info(f"Extracted {len(all_paid_photo_ids)} paid photo_ids from orders: {list(all_paid_photo_ids)}")
                            # Usa i photo_ids dagli ordini come paid_photos
                            paid_photos = list(all_paid_photo_ids)
                            
                            # AUTO-FIX: Marca automaticamente le foto come pagate (solo quelle non già pagate)
                            # Questo assicura che la prossima volta funzioni direttamente senza fallback
                            # Verifica quali foto non sono ancora marcate come pagate
                            existing_paid = set()
                            if found_photos:
                                existing_paid = {p['photo_id'] for p in found_photos if p.get('status') == 'paid'}
                            
                            photos_to_fix = all_paid_photo_ids - existing_paid
                            if photos_to_fix:
                                logger.info(f"Auto-fixing {len(photos_to_fix)} photos that are not yet marked as paid")
                                fixed_count = 0
                                for photo_id in photos_to_fix:
                                    try:
                                        success = await _mark_photo_paid(email, photo_id)
                                        if success:
                                            fixed_count += 1
                                    except Exception as e:
                                        logger.error(f"Error auto-fixing photo {photo_id}: {e}")
                                
                                if fixed_count > 0:
                                    logger.info(f"Auto-fixed {fixed_count} photos for {email} based on orders")
                            else:
                                logger.info(f"All photos from orders are already marked as paid - no fix needed")
                except Exception as e:
                    logger.error(f"Error checking orders as fallback: {e}", exc_info=True)
        
        if paid_photos:
            logger.info(f"Paid photos for {email}: {paid_photos}")
        
        return {
            "ok": True,
            "email": email,
            "found_photos": found_photos,
            "paid_photos": paid_photos
        }
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
        if SQLITE_AVAILABLE:
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
            photo_url = f"{base_url}/photo/{photo_id}?paid=true&email={email}"
            photo_id_escaped = photo_id.replace("'", "\\'").replace('"', '&quot;')
            email_escaped = email.replace("'", "\\'").replace('"', '&quot;')
            
            photos_html += f"""
                <div class="photo-item">
                    <img src="{photo_url}" alt="Photo" loading="lazy" class="photo-img" style="cursor: pointer; -webkit-touch-callout: default; -webkit-user-select: none; user-select: none;">
                    <button class="download-btn download-btn-desktop" onclick="downloadPhotoSuccess('{photo_id_escaped}', '{email_escaped}', this)" style="display: none;">📥 Download</button>
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
                }}
                .download-btn {{
                    position: absolute;
                    bottom: 10px;
                    right: 10px;
                    background: #22c55e;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 8px;
                    cursor: pointer;
                    font-weight: 600;
                    font-size: 14px;
                }}
                .download-btn:hover {{
                    background: #16a34a;
                }}
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
                <h1>YOUR PHOTOS</h1>
                <p class="message">You purchased {len(paid_photos)} photos</p>
                
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
                <div class="photos-grid">
                    {photos_html}
                </div>
                
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
                        const btn = btnElement || event?.target || document.querySelector(`button[onclick*="${{photoId}}"]`);
                        if (!btn) {{
                            console.error('Pulsante non trovato');
                            alert('Errore: pulsante non trovato');
                            return;
                        }}
                        
                        btn.disabled = true;
                        btn.textContent = '⏳ Scaricamento...';
                        
                        const filename = photoId.split('/').pop() || 'foto.jpg';
                        
                        // Costruisci URL con paid=true, email e download=true
                        let photoUrl = `/photo/${{encodeURIComponent(photoId)}}?paid=true&download=true`;
                        if (email) {{
                            photoUrl += `&email=${{encodeURIComponent(email)}}`;
                        }}
                        
                        // Su iOS: usa approccio semplice e diretto
                        if (isIOS()) {{
                            try {{
                                // Prova prima con Web Share API
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
                                    }}
                                }}
                                
                                // Fallback: apri l'immagine
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
                                alert('Errore nel caricamento della foto. Riprova.');
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
        
        photo_ids = order['photo_ids']
        expires_at = order.get('expires_at')
        
        if expires_at:
            expires_date = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            days_remaining = (expires_date - now).days
        else:
            days_remaining = 30
        
        # Genera HTML per pagina download
        email = order.get('email', '')
        photos_html = ""
        for photo_id in photo_ids:
            # Usa token per verifica, ma aggiungi anche email se disponibile
            photo_url = f"/photo/{photo_id}?token={token}&paid=true"
            if email:
                photo_url += f"&email={email}"
            photos_html += f"""
            <div class="photo-item">
                <img src="{photo_url}" alt="{photo_id}" loading="lazy">
                <button onclick="downloadPhoto('{photo_id}', '{token}', '{email}')" class="download-btn">Scarica</button>
            </div>
            """
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Le tue foto - TenerifePictures</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0a0a0a; color: #fff; padding: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .header h1 {{ font-size: 32px; margin-bottom: 10px; }}
                .warning {{ background: #fff3cd; color: #856404; padding: 15px; border-radius: 8px; margin: 20px 0; text-align: center; }}
                .warning strong {{ display: block; margin-bottom: 5px; }}
                .photos-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }}
                .photo-item {{ position: relative; border-radius: 12px; overflow: hidden; background: #1a1a1a; }}
                .photo-item img {{ width: 100%; height: auto; display: block; }}
                .download-btn {{ position: absolute; bottom: 10px; right: 10px; background: #7b74ff; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: 600; }}
                .download-btn:hover {{ background: #6a63e6; }}
                .download-all {{ display: block; width: 100%; max-width: 300px; margin: 30px auto; padding: 15px 30px; background: linear-gradient(135deg, #7b74ff, #5f58ff); color: white; border: none; border-radius: 8px; font-size: 18px; font-weight: 600; cursor: pointer; text-align: center; text-decoration: none; }}
                .download-all:hover {{ opacity: 0.9; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>✅ Le tue foto sono pronte!</h1>
                <p>Hai acquistato {len(photo_ids)} foto</p>
            </div>
            
            <div class="warning">
                <strong>⚠️ IMPORTANTE</strong>
                Le foto saranno disponibili per <strong>{days_remaining} giorni</strong>.
                Assicurati di scaricarle nella tua galleria prima della scadenza!
            </div>
            
            <div class="photos-grid">
                {photos_html}
            </div>
            
            <a href="#" onclick="downloadAll(); return false;" class="download-all">Scarica tutte le foto</a>
            
            <div style="text-align: center; margin-top: 30px;">
                <a href="/" style="color: #7b74ff; text-decoration: none; font-size: 16px;">← Torna alla home</a>
            </div>
            
            <script>
                // Salva email e token in localStorage per accesso futuro
                const email = '{email}';
                const token = '{token}';
                if(email) {{
                    localStorage.setItem('userEmail', email);
                }}
                if(token) {{
                    localStorage.setItem('downloadToken', token);
                }}
                
                function downloadPhoto(photoId, token, email) {{
                    // Costruisci URL con token, paid=true e download=true
                    let url = `/photo/${{encodeURIComponent(photoId)}}?token=${{token}}&paid=true&download=true`;
                    if(email) {{
                        url += `&email=${{encodeURIComponent(email)}}`;
                    }}
                    
                    // Su mobile, usa fetch per download corretto
                    if(/iPhone|iPad|iPod|Android/i.test(navigator.userAgent)) {{
                        fetch(url)
                            .then(response => response.blob())
                            .then(blob => {{
                                const blobUrl = window.URL.createObjectURL(blob);
                                const link = document.createElement('a');
                                link.href = blobUrl;
                                link.download = photoId;
                                document.body.appendChild(link);
                                link.click();
                                document.body.removeChild(link);
                                window.URL.revokeObjectURL(blobUrl);
                            }})
                            .catch(err => {{
                                console.error('Download error:', err);
                                // Fallback: apri in nuova tab
                                window.open(url, '_blank');
                            }});
                    }} else {{
                        // Desktop: download diretto
                        const link = document.createElement('a');
                        link.href = url;
                        link.download = photoId;
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                    }}
                }}
                
                async function downloadAll() {{
                    const photoIds = {json.dumps(photo_ids)};
                    const token = '{token}';
                    const email = '{email}';
                    for (const photoId of photoIds) {{
                        await new Promise(resolve => {{
                            setTimeout(() => {{
                                downloadPhoto(photoId, token, email);
                                resolve();
                            }}, 500);
                        }});
                    }}
                }}
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
        
        photo_ids = _get_cart(session_id)
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
            "matched_count": len(results),
            "back_photos_count": 0
        }
    except Exception as e:
        logger.error(f"Error in test-album endpoint: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e)
        }

# ========== ENDPOINT ESISTENTI ==========

@app.post("/match_selfie")
async def match_selfie(
    selfie: UploadFile = File(...),
    email: Optional[str] = Query(None, description="Email utente (opzionale, per salvare foto trovate)"),
    top_k_faces: int = Query(120),
    min_score: float = Query(0.25, description="Soglia minima di similarità (0.0-1.0). Più alta = meno falsi positivi"),
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
                
                # Filtra per tour_date se fornita
                if tour_date:
                    normalized_date = tour_date.replace("-", "") if "-" not in tour_date else tour_date
                    if len(normalized_date) == 8:  # YYYYMMDD
                        normalized_date = f"{normalized_date[:4]}-{normalized_date[4:6]}-{normalized_date[6:8]}"
                    
                    photo_tour_date = row.get("tour_date")
                    if photo_tour_date and normalized_date not in str(photo_tour_date):
                        continue  # Salta questa foto se non corrisponde alla data
                
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
        
        # Ordina per score decrescente e limita ai migliori match
        matched_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Limita a 50 risultati migliori per evitare troppi falsi positivi
        if len(matched_results) > 50:
            logger.info(f"Limiting results from {len(matched_results)} to 50 best matches")
            matched_results = matched_results[:50]
        
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
        
        # Se email fornita, salva foto trovate nel database
        if email:
            for result in matched_results:
                await _add_user_photo(email, result["photo_id"], "found")
            for result in back_results:
                await _add_user_photo(email, result["photo_id"], "found")
            logger.info(f"Saved {len(matched_results) + len(back_results)} photos for user {email}")
        
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
    photo_id: str = Query(..., description="ID foto da aggiungere"),
    email: Optional[str] = Query(None, description="Email utente per verificare se foto è già pagata")
):
    """Aggiunge una foto al carrello - previene aggiunta di foto già pagate"""
    # Verifica se la foto è già pagata (se email fornita)
    if email and (USE_POSTGRES or SQLITE_AVAILABLE):
        try:
            paid_photos = await _get_user_paid_photos(email)
            if photo_id in paid_photos:
                logger.warning(f"Attempt to add already paid photo to cart: {photo_id} for {email}")
                return {
                    "ok": False,
                    "error": "Questa foto è già stata acquistata",
                    "photo_ids": _get_cart(session_id),
                    "count": len(_get_cart(session_id)),
                    "price_cents": 0,
                    "price_euros": 0.0
                }
        except Exception as e:
            logger.error(f"Error checking paid photos in cart/add: {e}")
    
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
    session_id: str = Query(..., description="ID sessione"),
    email: Optional[str] = Query(None, description="Email utente (obbligatoria per salvare ordine)")
):
    """Crea una sessione di checkout Stripe"""
    logger.info(f"=== CHECKOUT REQUEST ===")
    logger.info(f"Session ID: {session_id}")
    logger.info(f"Email: {email}")
    logger.info(f"USE_STRIPE: {USE_STRIPE}")
    
    if not USE_STRIPE:
        logger.error("Stripe not configured")
        raise HTTPException(status_code=503, detail="Stripe not configured")
    
    photo_ids = _get_cart(session_id)
    logger.info(f"Cart photo_ids: {photo_ids}")
    
    if not photo_ids:
        logger.error("Cart is empty")
        raise HTTPException(status_code=400, detail="Cart is empty")
    
    if not email:
        logger.error("Email is required")
        raise HTTPException(status_code=400, detail="Email is required")
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
            metadata={
                'session_id': session_id,
                'email': email,  # Aggiungi email al metadata
                'photo_ids': ','.join(photo_ids),
                'photo_count': str(len(photo_ids)),
            }
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
    session_id: str = Query(..., description="Stripe session ID"),
    cart_session: str = Query(..., description="Cart session ID")
):
    """Pagina di successo dopo pagamento - mostra foto direttamente"""
    try:
        download_token = None
        order_data = {}
        photo_ids = []
        email = None
        
        # Il session_id è lo Stripe session ID (order_id)
        stripe_session_id = session_id
        
        # Prova prima dal database (più affidabile)
        if USE_POSTGRES or SQLITE_AVAILABLE:
            try:
                row = await _db_execute_one(
                    "SELECT download_token, photo_ids, email FROM orders WHERE stripe_session_id = ?",
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
            if USE_POSTGRES or SQLITE_AVAILABLE:
                try:
                    row = await _db_execute_one(
                        "SELECT download_token, photo_ids, email FROM orders WHERE stripe_session_id = ?",
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
                    # Se abbiamo email e photo_ids ma non ordine nel DB, crealo ora (PostgreSQL o SQLite)
                    if (USE_POSTGRES or SQLITE_AVAILABLE) and not download_token:
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
        photo_ids_str = metadata.get('photo_ids', '')
        
        # Log dettagliato per debug
        logger.info(f"=== WEBHOOK RECEIVED ===")
        logger.info(f"Event type: {event['type']}")
        logger.info(f"Session ID from metadata: {session_id}")
        logger.info(f"Session ID from session: {session.get('id')}")
        logger.info(f"Email from metadata: {metadata.get('email')}")
        logger.info(f"Email from session: {session.get('customer_email')}")
        logger.info(f"Email from customer_details: {session.get('customer_details', {}).get('email')}")
        logger.info(f"Final email: {email}")
        logger.info(f"Photo IDs from metadata: {photo_ids_str}")
        logger.info(f"Full metadata: {metadata}")
        logger.info(f"Session payment_status: {session.get('payment_status')}")
        logger.info(f"Session status: {session.get('status')}")
        
        if session_id and photo_ids_str:
            # Se manca email, prova a recuperarla
            if not email:
                email = session.get('customer_email') or session.get('customer_details', {}).get('email')
                logger.warning(f"Email not in metadata, recovered from session: {email}")
            
            if email:
                # Normalizza email
                original_email = email
                email = _normalize_email(email)
                logger.info(f"Email normalized: '{original_email}' -> '{email}'")
                photo_ids = [pid.strip() for pid in photo_ids_str.split(',') if pid.strip()]
                logger.info(f"Photo IDs parsed: {photo_ids} (count: {len(photo_ids)})")
                order_id = session.get('id')
                amount_cents = session.get('amount_total', 0)
                logger.info(f"Creating order: order_id={order_id}, email={email}, amount={amount_cents}, photos={len(photo_ids)}")
                
                # Crea ordine nel database con download token
                base_url = str(request.base_url).rstrip('/')
                download_token = await _create_order(email, order_id, order_id, photo_ids, amount_cents)
                
                if download_token:
                    logger.info(f"✅ Order created successfully: {order_id}, download_token generated")
                else:
                    logger.error(f"❌ Order creation failed: {order_id}")
                
                if download_token:
                    # Invia email di conferma pagamento
                    try:
                        logger.info(f"=== PAYMENT CONFIRMATION EMAIL ===")
                        logger.info(f"Email: {email}")
                        logger.info(f"Photos: {len(photo_ids)}")
                        logger.info(f"Download token: {download_token}")
                        logger.info(f"Base URL: {base_url}")
                        logger.info(f"USE_SENDGRID: {USE_SENDGRID}")
                        logger.info(f"sg_client available: {sg_client is not None}")
                        
                        email_sent = await _send_payment_confirmation_email(email, photo_ids, download_token, base_url)
                        if email_sent:
                            logger.info(f"✅ Payment confirmation email sent successfully to {email}")
                        else:
                            logger.error(f"❌ Payment confirmation email FAILED to send to {email}")
                            logger.error("Check SendGrid configuration and logs above for details")
                    except Exception as e:
                        logger.error(f"❌ Exception sending payment confirmation email: {e}", exc_info=True)
                        logger.error(f"Exception type: {type(e).__name__}")
                
                # Salva anche in file JSON per compatibilità
                order_data = {
                    'order_id': order_id,
                    'stripe_session_id': order_id,
                    'session_id': session_id,
                    'email': email,
                    'photo_ids': photo_ids,
                    'amount_cents': amount_cents,
                    'paid_at': datetime.now(timezone.utc).isoformat(),
                    'status': 'paid',
                    'download_token': download_token
                }
                
                order_file = ORDERS_DIR / f"{order_id}.json"
                with open(order_file, 'w', encoding='utf-8') as f:
                    json.dump(order_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Order completed: {order_id} - {len(photo_ids)} photos for {email}")
                
                # SVUOTA CARRELLO: rimuovi le foto acquistate dal carrello dopo il pagamento
                # Questo previene che l'utente possa riacquistare le stesse foto
                if session_id:
                    try:
                        # Rimuovi solo le foto acquistate dal carrello (non svuotare tutto)
                        # in caso l'utente abbia aggiunto altre foto dopo il checkout
                        cart_photo_ids = _get_cart(session_id)
                        if cart_photo_ids:
                            photo_ids_set = set(photo_ids)
                            remaining_photos = [p for p in cart_photo_ids if p not in photo_ids_set]
                            
                            if len(remaining_photos) != len(cart_photo_ids):
                                # Ci sono foto acquistate nel carrello, rimuovile
                                carts[session_id] = remaining_photos
                                logger.info(f"Removed {len(cart_photo_ids) - len(remaining_photos)} purchased photos from cart {session_id}")
                            else:
                                logger.info(f"No purchased photos found in cart {session_id} (already removed or not present)")
                    except Exception as e:
                        logger.error(f"Error clearing purchased photos from cart: {e}")
            else:
                logger.error(f"Order failed: missing email. session_id={session_id}, photo_ids={photo_ids_str}")
        else:
            logger.error(f"Order failed: missing required data. session_id={session_id}, email={email}, photo_ids={photo_ids_str}")
    
    return {"ok": True}

@app.get("/test-email")
async def test_email(
    email: str = Query(..., description="Email di test")
):
    """Endpoint di test per verificare l'invio email"""
    try:
        logger.info(f"=== TEST EMAIL ENDPOINT ===")
        logger.info(f"USE_SENDGRID: {USE_SENDGRID}")
        logger.info(f"SENDGRID_AVAILABLE: {SENDGRID_AVAILABLE}")
        logger.info(f"SENDGRID_API_KEY set: {bool(SENDGRID_API_KEY)}")
        logger.info(f"SENDGRID_FROM_EMAIL: {SENDGRID_FROM_EMAIL}")
        logger.info(f"sg_client available: {sg_client is not None}")
        
        if not USE_SENDGRID:
            return {
                "ok": False,
                "error": "SendGrid not configured",
                "details": {
                    "SENDGRID_AVAILABLE": SENDGRID_AVAILABLE,
                    "SENDGRID_API_KEY_set": bool(SENDGRID_API_KEY),
                    "sg_client_available": sg_client is not None
                }
            }
        
        # Test invio email
        test_subject = "Test Email - Tenerife Stars Pictures"
        test_html = "<h1>Test Email</h1><p>Questa è una email di test.</p>"
        test_plain = "Test Email - Questa è una email di test."
        
        result = await _send_email(email, test_subject, test_html, test_plain)
        
        return {
            "ok": result,
            "message": "Email sent successfully" if result else "Email failed to send",
            "email": email,
            "check_logs": True
        }
    except Exception as e:
        logger.error(f"Error in test email endpoint: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "type": type(e).__name__
        }

@app.get("/test-download")
async def test_download_page(
    request: Request,
    email: str = Query("test@example.com", description="Email di test"),
    photo_id: str = Query(None, description="ID foto di test (opzionale)")
):
    """Pagina di test per verificare il download su iPhone senza completare il flusso completo"""
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

# ========== ADMIN PANEL ==========

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")  # Cambia in produzione!

def _check_admin_auth(password: Optional[str] = None) -> bool:
    """Verifica autenticazione admin"""
    if not password:
        return False
    return password == ADMIN_PASSWORD

@app.get("/admin", response_class=HTMLResponse)
async def admin_panel():
    """Pagina admin"""
    admin_path = STATIC_DIR / "admin.html"
    if not admin_path.exists():
        raise HTTPException(status_code=404, detail="Admin page not found")
    return FileResponse(admin_path)

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
        
        if USE_POSTGRES or SQLITE_AVAILABLE:
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
        
        # Foto totali
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
        if USE_POSTGRES or SQLITE_AVAILABLE:
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
                rows = await cursor.fetchall()
                for row in rows:
                    photo_ids = json.loads(row['photo_ids']) if row['photo_ids'] else []
                    orders.append({
                        'order_id': row['order_id'],
                        'email': row['email'],
                        'photo_count': len(photo_ids),
                        'amount_cents': row['amount_cents'],
                        'paid_at': row['paid_at'],
                        'download_token': row['download_token']
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
    """Upload e indicizzazione foto admin"""
    if not _check_admin_auth(password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        global meta_rows, back_photos, faiss_index
        
        # Salva foto
        photo_path = PHOTOS_DIR / photo.filename
        content = await photo.read()
        with open(photo_path, 'wb') as f:
            f.write(content)
        
        # Indicizza foto (se face_app è disponibile)
        if face_app is not None:
            img = _read_image_from_bytes(content)
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
                        "photo_id": photo.filename,
                        "has_face": True,
                        "det_score": float(getattr(f, "det_score", 0.0)),
                        "bbox": [float(x) for x in f.bbox.tolist()],
                    }
                    if tour_date:
                        record["tour_date"] = tour_date
                    meta_rows.append(record)
                    
                    # Salva meta su file
                    with open(META_PATH, 'a', encoding='utf-8') as meta_f:
                        meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                # Salva indice aggiornato
                if faiss_index is not None:
                    faiss.write_index(faiss_index, INDEX_PATH)
                
                logger.info(f"Photo indexed: {photo.filename} - {len(faces)} faces (tour_date: {tour_date})")
            else:
                # Foto senza volti - aggiungi a back_photos
                back_record = {
                    "photo_id": photo.filename,
                    "has_face": False,
                }
                if tour_date:
                    back_record["tour_date"] = tour_date
                back_photos.append(back_record)
                
                # Salva su file
                with open(BACK_PHOTOS_PATH, 'a', encoding='utf-8') as back_f:
                    back_f.write(json.dumps(back_record, ensure_ascii=False) + "\n")
                
                logger.info(f"Photo added as back photo (no faces): {photo.filename} (tour_date: {tour_date})")
        
        return {"ok": True, "filename": photo.filename}
    except Exception as e:
        logger.error(f"Error uploading photo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/back-photos")
async def admin_back_photos(password: str = Query(..., description="Password admin")):
    """Lista foto senza volti"""
    if not _check_admin_auth(password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        photos = []
        if BACK_PHOTOS_PATH.exists():
            with open(BACK_PHOTOS_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        photos.append({
                            'filename': record.get('photo_id'),
                            'has_face': record.get('has_face', False)
                        })
        
        return {"ok": True, "photos": photos}
    except Exception as e:
        logger.error(f"Error getting back photos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/back-photo")
async def admin_add_back_photo(
    photo: UploadFile = File(...),
    password: str = Form(...),
    tour_date: Optional[str] = Form(None)
):
    """Aggiungi foto senza volti"""
    if not _check_admin_auth(password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        # Salva foto
        photo_path = PHOTOS_DIR / photo.filename
        content = await photo.read()
        with open(photo_path, 'wb') as f:
            f.write(content)
        
        # Aggiungi a back_photos (con tour_date se fornita)
        back_record = {
            "photo_id": photo.filename,
            "has_face": False,
        }
        if tour_date:
            back_record["tour_date"] = tour_date
        back_photos.append(back_record)
        
        # Salva su file
        with open(BACK_PHOTOS_PATH, 'a', encoding='utf-8') as back_f:
            back_f.write(json.dumps(back_record, ensure_ascii=False) + "\n")
        
        logger.info(f"Back photo added: {photo.filename} (tour_date: {tour_date})")
        return {"ok": True, "filename": photo.filename}
    except Exception as e:
        logger.error(f"Error adding back photo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/admin/back-photo/{filename:path}")
async def admin_remove_back_photo(
    filename: str,
    password: str = Query(..., description="Password admin")
):
    """Rimuovi foto senza volti"""
    if not _check_admin_auth(password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        # Rimuovi da back_photos
        global back_photos
        back_photos = [p for p in back_photos if p.get('photo_id') != filename]
        
        # Riscrivi file
        with open(BACK_PHOTOS_PATH, 'w', encoding='utf-8') as f:
            for record in back_photos:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        logger.info(f"Back photo removed: {filename}")
        return {"ok": True}
    except Exception as e:
        logger.error(f"Error removing back photo: {e}")
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
            if USE_POSTGRES:
                query += " AND paid_at >= $1"
            else:
                query += " AND paid_at >= ?"
            params.append(start_date)
            if end_date:
                if USE_POSTGRES:
                    query += " AND paid_at <= $2"
                else:
                    query += " AND paid_at <= ?"
                params.append(end_date)
        elif end_date:
            if USE_POSTGRES:
                query += " AND paid_at <= $1"
            else:
                query += " AND paid_at <= ?"
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
        if not (USE_POSTGRES or SQLITE_AVAILABLE):
            return {"ok": False, "error": "Database not available"}
        
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
                        "SELECT status, paid_at FROM user_photos WHERE email = ? AND photo_id = ?",
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

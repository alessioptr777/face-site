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

# SendGrid per email
try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail, Email, To, Content
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False

# SQLite per database utenti
try:
    import sqlite3
    import aiosqlite
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

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
    sg_client = SendGridAPIClient(SENDGRID_API_KEY)
    logger.info("SendGrid configured - email sending enabled")
else:
    sg_client = None
    if not SENDGRID_AVAILABLE:
        logger.warning("SendGrid not configured - sendgrid package not available")
    elif not SENDGRID_API_KEY:
        logger.warning("SendGrid not configured - SENDGRID_API_KEY not set")
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
# ========== DATABASE SQLITE ==========

def _init_database():
    """Inizializza il database SQLite con le tabelle necessarie"""
    if not SQLITE_AVAILABLE:
        logger.warning("SQLite not available - user tracking disabled")
        return
    
    try:
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
        logger.info(f"Database initialized: {DB_PATH}")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

async def _get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Recupera un utente per email"""
    if not SQLITE_AVAILABLE:
        return None
    
    try:
        async with aiosqlite.connect(DB_PATH) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM users WHERE email = ?",
                (email,)
            )
            row = await cursor.fetchone()
            if row:
                return dict(row)
    except Exception as e:
        logger.error(f"Error getting user: {e}")
    return None

async def _create_or_update_user(email: str, selfie_embedding: Optional[bytes] = None) -> bool:
    """Crea o aggiorna un utente"""
    if not SQLITE_AVAILABLE:
        return False
    
    try:
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(DB_PATH) as conn:
            # Verifica se esiste
            cursor = await conn.execute(
                "SELECT email FROM users WHERE email = ?",
                (email,)
            )
            exists = await cursor.fetchone()
            
            if exists:
                # Aggiorna
                if selfie_embedding:
                    await conn.execute("""
                        UPDATE users 
                        SET selfie_embedding = ?, last_login_at = ?, last_selfie_at = ?
                        WHERE email = ?
                    """, (selfie_embedding, now, now, email))
                else:
                    await conn.execute("""
                        UPDATE users 
                        SET last_login_at = ?
                        WHERE email = ?
                    """, (now, email))
            else:
                # Crea nuovo
                await conn.execute("""
                    INSERT INTO users (email, selfie_embedding, created_at, last_login_at, last_selfie_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (email, selfie_embedding, now, now, now))
            
            await conn.commit()
            return True
    except Exception as e:
        logger.error(f"Error creating/updating user: {e}")
    return False

async def _add_user_photo(email: str, photo_id: str, status: str = "found") -> bool:
    """Aggiunge una foto trovata per un utente"""
    if not SQLITE_AVAILABLE:
        return False
    
    try:
        now = datetime.now(timezone.utc).isoformat()
        
        # Calcola expires_at
        if status == "paid":
            expires_at = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        else:
            expires_at = (datetime.now(timezone.utc) + timedelta(days=90)).isoformat()
        
        async with aiosqlite.connect(DB_PATH) as conn:
            # Verifica se esiste già
            cursor = await conn.execute(
                "SELECT id FROM user_photos WHERE email = ? AND photo_id = ?",
                (email, photo_id)
            )
            exists = await cursor.fetchone()
            
            if exists:
                # Aggiorna
                await conn.execute("""
                    UPDATE user_photos 
                    SET found_at = ?, status = ?, expires_at = ?
                    WHERE email = ? AND photo_id = ?
                """, (now, status, expires_at, email, photo_id))
            else:
                # Inserisci nuovo
                await conn.execute("""
                    INSERT INTO user_photos (email, photo_id, found_at, status, expires_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (email, photo_id, now, status, expires_at))
            
            await conn.commit()
            return True
    except Exception as e:
        logger.error(f"Error adding user photo: {e}")
    return False

async def _mark_photo_paid(email: str, photo_id: str) -> bool:
    """Marca una foto come pagata"""
    if not SQLITE_AVAILABLE:
        return False
    
    try:
        now = datetime.now(timezone.utc).isoformat()
        expires_at = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        
        async with aiosqlite.connect(DB_PATH) as conn:
            await conn.execute("""
                UPDATE user_photos 
                SET paid_at = ?, status = 'paid', expires_at = ?
                WHERE email = ? AND photo_id = ?
            """, (now, expires_at, email, photo_id))
            await conn.commit()
            return True
    except Exception as e:
        logger.error(f"Error marking photo paid: {e}")
    return False

async def _get_user_paid_photos(email: str) -> List[str]:
    """Recupera lista foto pagate per un utente (non scadute)"""
    if not SQLITE_AVAILABLE:
        return []
    
    try:
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(DB_PATH) as conn:
            cursor = await conn.execute("""
                SELECT photo_id FROM user_photos 
                WHERE email = ? AND status = 'paid' AND expires_at > ?
            """, (email, now))
            rows = await cursor.fetchall()
            return [row[0] for row in rows]
    except Exception as e:
        logger.error(f"Error getting paid photos: {e}")
    return []

async def _get_user_found_photos(email: str) -> List[Dict[str, Any]]:
    """Recupera tutte le foto trovate per un utente"""
    if not SQLITE_AVAILABLE:
        return []
    
    try:
        async with aiosqlite.connect(DB_PATH) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("""
                SELECT photo_id, found_at, paid_at, expires_at, status 
                FROM user_photos 
                WHERE email = ?
                ORDER BY found_at DESC
            """, (email,))
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error getting found photos: {e}")
    return []

async def _match_selfie_embedding(selfie_embedding: bytes, threshold: float = 0.7) -> Optional[str]:
    """Confronta selfie embedding con quelli salvati, ritorna email se match"""
    if not SQLITE_AVAILABLE:
        return None
    
    try:
        async with aiosqlite.connect(DB_PATH) as conn:
            cursor = await conn.execute(
                "SELECT email, selfie_embedding FROM users WHERE selfie_embedding IS NOT NULL"
            )
            rows = await cursor.fetchall()
            
            selfie_emb = np.frombuffer(selfie_embedding, dtype=np.float32)
            selfie_emb = _normalize(selfie_emb)
            
            best_match = None
            best_score = 0.0
            
            for row in rows:
                saved_emb = np.frombuffer(row[1], dtype=np.float32)
                saved_emb = _normalize(saved_emb)
                
                # Calcola cosine similarity
                score = np.dot(selfie_emb, saved_emb)
                
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = row[0]
            
            if best_match:
                logger.info(f"Selfie matched with user: {best_match} (score: {best_score:.4f})")
                return best_match
    except Exception as e:
        logger.error(f"Error matching selfie: {e}")
    return None

async def _create_order(email: str, order_id: str, stripe_session_id: str, photo_ids: List[str], amount_cents: int) -> Optional[str]:
    """Crea un ordine e ritorna download token"""
    if not SQLITE_AVAILABLE:
        return None
    
    try:
        now = datetime.now(timezone.utc).isoformat()
        download_token = secrets.token_urlsafe(32)
        
        async with aiosqlite.connect(DB_PATH) as conn:
            await conn.execute("""
                INSERT INTO orders (order_id, email, stripe_session_id, photo_ids, amount_cents, paid_at, download_token)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (order_id, email, stripe_session_id, json.dumps(photo_ids), amount_cents, now, download_token))
            
            # Marca foto come pagate
            for photo_id in photo_ids:
                await _mark_photo_paid(email, photo_id)
            
            await conn.commit()
            return download_token
    except Exception as e:
        logger.error(f"Error creating order: {e}")
    return None

async def _get_order_by_token(token: str) -> Optional[Dict[str, Any]]:
    """Recupera ordine per download token"""
    if not SQLITE_AVAILABLE:
        return None
    
    try:
        async with aiosqlite.connect(DB_PATH) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM orders WHERE download_token = ?",
                (token,)
            )
            row = await cursor.fetchone()
            if row:
                order = dict(row)
                order['photo_ids'] = json.loads(order['photo_ids'])
                return order
    except Exception as e:
        logger.error(f"Error getting order: {e}")
    return None

async def _get_photos_for_followup() -> List[Dict[str, Any]]:
    """Recupera foto che necessitano follow-up email"""
    if not SQLITE_AVAILABLE:
        return []
    
    try:
        now = datetime.now(timezone.utc)
        async with aiosqlite.connect(DB_PATH) as conn:
            conn.row_factory = aiosqlite.Row
            # Foto non pagate trovate 7, 30, 60 giorni fa
            cursor = await conn.execute("""
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
            """, (now.isoformat(),) * 8)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows if row['followup_type']]
    except Exception as e:
        logger.error(f"Error getting photos for followup: {e}")
    return []

async def _mark_followup_sent(email: str, photo_id: str, followup_type: str):
    """Marca follow-up email come inviata"""
    if not SQLITE_AVAILABLE:
        return
    
    try:
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(DB_PATH) as conn:
            await conn.execute("""
                INSERT OR REPLACE INTO email_followups (email, photo_id, followup_type, sent_at)
                VALUES (?, ?, ?, ?)
            """, (email, photo_id, followup_type, now))
            await conn.commit()
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
    if not SQLITE_AVAILABLE:
        return
    
    try:
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(DB_PATH) as conn:
            # Trova foto scadute
            cursor = await conn.execute("""
                SELECT email, photo_id, status FROM user_photos
                WHERE expires_at < ? AND status != 'deleted'
            """, (now,))
            expired = await cursor.fetchall()
            
            deleted_count = 0
            for row in expired:
                email, photo_id, status = row
                # Marca come deleted nel database
                await conn.execute("""
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
            
            await conn.commit()
            if expired:
                logger.info(f"Cleaned up {len(expired)} expired photos ({deleted_count} files deleted)")
    except Exception as e:
        logger.error(f"Error cleaning up expired photos: {e}")

# ========== FUNZIONI EMAIL SENDGRID ==========

async def _send_email(to_email: str, subject: str, html_content: str, plain_content: str = None) -> bool:
    """Invia email tramite SendGrid"""
    if not USE_SENDGRID or not sg_client:
        logger.warning("SendGrid not available - email not sent")
        return False
    
    try:
        message = Mail(
            from_email=Email(SENDGRID_FROM_EMAIL, "Tenerife Stars Pictures"),
            to_emails=To(to_email),
            subject=subject,
            html_content=Content("text/html", html_content)
        )
        
        if plain_content:
            message.plain_text_content = Content("text/plain", plain_content)
        
        response = sg_client.send(message)
        if response.status_code in [200, 201, 202]:
            logger.info(f"Email sent successfully to {to_email}")
            return True
        else:
            logger.error(f"SendGrid error: {response.status_code} - {response.body}")
            return False
    except Exception as e:
        logger.error(f"Error sending email: {e}")
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
    
    return await _send_email(email, subject, html_content, plain_content)

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
    
    # Inizializza database SQLite all'avvio
    _init_database()
    
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
    logger.info("Periodic tasks started (cleanup and follow-up every 6 hours)")

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

@app.post("/check_user")
async def check_user(
    email: str = Query(..., description="Email utente"),
    selfie: UploadFile = File(..., description="Selfie per verifica")
):
    """Verifica se utente esiste e matcha selfie, ritorna storico"""
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

@app.get("/user/photos")
async def get_user_photos(
    email: str = Query(..., description="Email utente")
):
    """Recupera tutte le foto di un utente (trovate e pagate)"""
    try:
        found_photos = await _get_user_found_photos(email)
        paid_photos = await _get_user_paid_photos(email)
        
        return {
            "ok": True,
            "email": email,
            "found_photos": found_photos,
            "paid_photos": paid_photos
        }
    except Exception as e:
        logger.error(f"Error getting user photos: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/my-photos/{token}")
async def my_photos_page(
    token: str,
    request: Request
):
    """Pagina download foto dopo pagamento"""
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
                    <a href="/">Torna alla home</a>
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
        photos_html = ""
        for photo_id in photo_ids:
            photo_url = f"/photo/{photo_id}?paid=true"
            photos_html += f"""
            <div class="photo-item">
                <img src="{photo_url}" alt="{photo_id}" loading="lazy">
                <button onclick="downloadPhoto('{photo_id}')" class="download-btn">Scarica</button>
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
            
            <script>
                function downloadPhoto(photoId) {{
                    const url = `/photo/${{photoId}}?paid=true`;
                    const link = document.createElement('a');
                    link.href = url;
                    link.download = photoId;
                    link.click();
                }}
                
                async function downloadAll() {{
                    const photoIds = {json.dumps(photo_ids)};
                    for (const photoId of photoIds) {{
                        await new Promise(resolve => {{
                            setTimeout(() => {{
                                downloadPhoto(photoId);
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
        
        # Se email fornita, salva foto trovate nel database
        if email:
            for result in matched_results:
                await _add_user_photo(email, result["photo_id"], "found")
            for result in back_results:
                await _add_user_photo(email, result["photo_id"], "found")
            logger.info(f"Saved {len(matched_results) + len(back_results)} photos for user {email}")
        
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
    session_id: str = Query(..., description="ID sessione"),
    email: Optional[str] = Query(None, description="Email utente (obbligatoria per salvare ordine)")
):
    """Crea una sessione di checkout Stripe"""
    if not USE_STRIPE:
        raise HTTPException(status_code=503, detail="Stripe not configured")
    
    photo_ids = _get_cart(session_id)
    if not photo_ids:
        raise HTTPException(status_code=400, detail="Cart is empty")
    
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")
    
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
            customer_email=email,  # Aggiungi email cliente a Stripe
            success_url=f'{base_url}/checkout/success?session_id={{CHECKOUT_SESSION_ID}}&cart_session={session_id}',
            cancel_url=f'{base_url}/checkout/cancel?session_id={session_id}',
            metadata={
                'session_id': session_id,
                'email': email,  # Aggiungi email al metadata
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
    request: Request,
    session_id: str = Query(..., description="Stripe session ID"),
    cart_session: str = Query(..., description="Cart session ID")
):
    """Pagina di successo dopo pagamento"""
    try:
        # Recupera ordine da file JSON (il webhook lo salva)
        order_file = ORDERS_DIR / f"{session_id}.json"
        download_token = None
        order_data = {}
        
        if order_file.exists():
            with open(order_file, 'r', encoding='utf-8') as f:
                order_data = json.load(f)
                download_token = order_data.get('download_token')
        
        # Se non trovato, prova a recuperare dal database
        if not download_token and SQLITE_AVAILABLE:
            try:
                async with aiosqlite.connect(DB_PATH) as conn:
                    conn.row_factory = aiosqlite.Row
                    cursor = await conn.execute(
                        "SELECT download_token FROM orders WHERE stripe_session_id = ?",
                        (session_id,)
                    )
                    row = await cursor.fetchone()
                    if row:
                        download_token = row['download_token']
            except Exception as e:
                logger.error(f"Error getting order from database: {e}")
        
        base_url = str(request.base_url).rstrip('/')
        download_url = f"{base_url}/my-photos/{download_token}" if download_token else None
        
        # Prepara le parti HTML condizionali (evita backslash in f-string)
        email_message = "<p>Controlla la tua email per il link di download.</p>" if not download_url else ""
        download_button = f'<a href="{download_url}" class="button">Scarica le tue foto</a>' if download_url else ""
        link_box = f'<div class="link-box">Link permanente:<br>{download_url}</div>' if download_url else ""
        album_link = f'<a href="/?email={order_data.get("email", "")}&refresh=paid" class="button">Vai all\'album</a>' if order_data.get('email') else ""
        
        html_content = f"""
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
                    max-width: 600px;
                }}
                h1 {{ font-size: 32px; margin: 0 0 20px; }}
                p {{ font-size: 18px; margin: 10px 0; }}
                .warning {{
                    background: rgba(255, 243, 205, 0.2);
                    border-left: 4px solid #ffc107;
                    padding: 15px;
                    margin: 20px 0;
                    text-align: left;
                    border-radius: 8px;
                }}
                .link-box {{
                    background: rgba(255,255,255,0.1);
                    padding: 15px;
                    border-radius: 8px;
                    margin: 20px 0;
                    word-break: break-all;
                    font-family: monospace;
                    font-size: 14px;
                }}
                .button {{
                    display: inline-block;
                    margin: 10px;
                    padding: 15px 30px;
                    background: #fff;
                    color: #5f58ff;
                    text-decoration: none;
                    border-radius: 8px;
                    font-weight: 600;
                    font-size: 16px;
                }}
                .button-secondary {{
                    background: rgba(255,255,255,0.2);
                    color: #fff;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>✅ Pagamento completato!</h1>
                <p>Le tue foto sono state sbloccate.</p>
                
                <div class="warning">
                    <strong>⚠️ IMPORTANTE:</strong><br>
                    Le foto saranno disponibili per <strong>30 giorni</strong>.<br>
                    Assicurati di averle scaricate nella tua galleria prima della scadenza.
                </div>
                
                {email_message}
                
                {download_button}
                
                {link_box}
                
                {album_link}
                
                <a href="/" class="button button-secondary">Torna alla home</a>
            </div>
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
            <a href="/">Torna alla home</a>
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
        email = metadata.get('email') or session.get('customer_email') or session.get('customer_details', {}).get('email')
        photo_ids_str = metadata.get('photo_ids', '')
        
        if session_id and photo_ids_str and email:
            photo_ids = photo_ids_str.split(',')
            order_id = session.get('id')
            amount_cents = session.get('amount_total', 0)
            
            # Crea ordine nel database con download token
            base_url = str(request.base_url).rstrip('/')
            download_token = await _create_order(email, order_id, order_id, photo_ids, amount_cents)
            
            if download_token:
                # Invia email di conferma pagamento
                try:
                    await _send_payment_confirmation_email(email, photo_ids, download_token, base_url)
                except Exception as e:
                    logger.error(f"Error sending payment confirmation email: {e}")
            
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
        else:
            logger.warning(f"Order completed but missing data: session_id={session_id}, email={email}, photo_ids={photo_ids_str}")
    
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

#!/usr/bin/env python3
"""
Script per migrare foto da storage locale a Cloudflare R2.
Esegue una migrazione one-time delle foto esistenti.
"""
import os
import sys
from pathlib import Path
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# Aggiungi backend al path
sys.path.insert(0, str(Path(__file__).parent))

# Configurazione R2
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "")
R2_BUCKET = os.getenv("R2_BUCKET", "")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")

if not all([R2_ENDPOINT_URL, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
    print("❌ ERRORE: Variabili R2 non configurate!")
    print("   Assicurati di avere:")
    print("   - R2_ENDPOINT_URL")
    print("   - R2_BUCKET")
    print("   - R2_ACCESS_KEY_ID")
    print("   - R2_SECRET_ACCESS_KEY")
    sys.exit(1)

# Inizializza client R2
r2_client = boto3.client(
    's3',
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    config=Config(signature_version='s3v4')
)

# Test connessione
try:
    r2_client.head_bucket(Bucket=R2_BUCKET)
    print(f"✅ Connesso a R2: bucket={R2_BUCKET}")
except Exception as e:
    print(f"❌ ERRORE: Impossibile connettersi a R2: {e}")
    sys.exit(1)

# Trova cartella foto locale
BASE_DIR = Path(__file__).parent
PHOTOS_DIR = BASE_DIR / "photos"

if not PHOTOS_DIR.exists():
    print(f"❌ ERRORE: Cartella foto non trovata: {PHOTOS_DIR}")
    sys.exit(1)

# Trova tutte le foto
photo_files = list(PHOTOS_DIR.glob("*.jpg")) + list(PHOTOS_DIR.glob("*.jpeg")) + list(PHOTOS_DIR.glob("*.png"))
print(f"\n📸 Foto trovate in locale: {len(photo_files)}")

if len(photo_files) == 0:
    print("✅ Nessuna foto da migrare!")
    sys.exit(0)

# Controlla quali foto sono già su R2
print("\n🔍 Verificando foto già presenti su R2...")
existing_keys = set()
uploaded = 0
skipped = 0
failed = 0

for photo_file in photo_files:
    key = photo_file.name
    
    # Verifica se esiste già
    try:
        r2_client.head_object(Bucket=R2_BUCKET, Key=key)
        existing_keys.add(key)
        skipped += 1
        print(f"⏭️  Già presente: {key}")
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            # Non esiste, procedi con upload
            try:
                with open(photo_file, 'rb') as f:
                    r2_client.put_object(
                        Bucket=R2_BUCKET,
                        Key=key,
                        Body=f.read(),
                        ContentType='image/jpeg' if key.lower().endswith(('.jpg', '.jpeg')) else 'image/png'
                    )
                uploaded += 1
                print(f"✅ Caricata: {key} ({photo_file.stat().st_size / 1024:.1f} KB)")
            except Exception as e:
                failed += 1
                print(f"❌ ERRORE caricando {key}: {e}")
        else:
            failed += 1
            print(f"❌ ERRORE verificando {key}: {e}")

print("\n" + "=" * 60)
print("📊 RIEPILOGO MIGRAZIONE")
print("=" * 60)
print(f"✅ Caricate: {uploaded}")
print(f"⏭️  Già presenti (saltate): {skipped}")
print(f"❌ Errori: {failed}")
print(f"📸 Totale: {len(photo_files)}")
print("=" * 60)

if uploaded > 0:
    print(f"\n✅ Migrazione completata! {uploaded} foto caricate su R2.")
if failed > 0:
    print(f"\n⚠️  ATTENZIONE: {failed} foto non sono state caricate. Controlla gli errori sopra.")


#!/usr/bin/env python3
"""
Script per caricare le foto su Cloudinary.

Uso:
1. Installa cloudinary: pip install cloudinary
2. Configura CLOUDINARY_URL come variabile d'ambiente o nel file .env
3. Esegui: python upload_to_cloudinary.py
"""

import os
import sys
from pathlib import Path
import cloudinary
import cloudinary.uploader

# Configura Cloudinary dalla variabile d'ambiente
CLOUDINARY_URL = os.getenv("CLOUDINARY_URL", "")
if not CLOUDINARY_URL:
    print("ERRORE: CLOUDINARY_URL non configurata!")
    print("Configura la variabile d'ambiente CLOUDINARY_URL")
    print("Esempio: export CLOUDINARY_URL='cloudinary://api_key:api_secret@cloud_name'")
    sys.exit(1)

cloudinary.config()

# Cartella delle foto
PHOTOS_DIR = Path(__file__).parent / "photos"

if not PHOTOS_DIR.exists():
    print(f"ERRORE: Cartella {PHOTOS_DIR} non trovata!")
    sys.exit(1)

# Carica tutte le foto
photo_files = [f for f in PHOTOS_DIR.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

if not photo_files:
    print(f"Nessuna foto trovata in {PHOTOS_DIR}")
    sys.exit(0)

print(f"Trovate {len(photo_files)} foto da caricare...")

uploaded = 0
failed = 0

for photo_file in photo_files:
    # Usa il nome del file senza estensione come public_id
    public_id = photo_file.stem
    
    try:
        print(f"Caricando {photo_file.name}...", end=" ")
        
        result = cloudinary.uploader.upload(
            str(photo_file),
            public_id=public_id,
            folder="face-site",  # Opzionale: organizza in una cartella
            overwrite=True
        )
        
        print(f"✓ OK (URL: {result.get('secure_url', 'N/A')})")
        uploaded += 1
        
    except Exception as e:
        print(f"✗ ERRORE: {e}")
        failed += 1

print(f"\nRiepilogo:")
print(f"  Caricate: {uploaded}")
print(f"  Fallite: {failed}")
print(f"  Totale: {len(photo_files)}")


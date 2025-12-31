#!/usr/bin/env python3
"""
Script per caricare le foto su Cloudinary con compressione automatica.

Uso:
1. Installa dipendenze: pip install cloudinary pillow
2. Configura CLOUDINARY_URL come variabile d'ambiente o nel file .env
3. Esegui: python upload_to_cloudinary.py
"""

import os
import sys
import io
from pathlib import Path
from PIL import Image
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

# Configurazione compressione
MAX_DIMENSION = 2000  # Lato massimo in pixel (larghezza o altezza)
JPEG_QUALITY = 85  # Qualità JPEG (85-90 è ottimo, 100 = nessuna compressione)
MAX_FILE_SIZE_MB = 10  # Dimensione massima file dopo compressione (MB)

def compress_image(image_path: Path) -> io.BytesIO:
    """
    Comprimi e ridimensiona un'immagine se necessario.
    Ritorna un BytesIO con l'immagine compressa.
    """
    # Leggi l'immagine originale
    img = Image.open(image_path)
    original_size = image_path.stat().st_size
    original_format = img.format
    
    # Converti in RGB se necessario (per JPEG)
    if img.mode in ('RGBA', 'LA', 'P'):
        # Crea background bianco per immagini con trasparenza
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Ridimensiona se troppo grande
    width, height = img.size
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        # Calcola nuove dimensioni mantenendo aspect ratio
        if width > height:
            new_width = MAX_DIMENSION
            new_height = int(height * (MAX_DIMENSION / width))
        else:
            new_height = MAX_DIMENSION
            new_width = int(width * (MAX_DIMENSION / height))
        
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        print(f"  Ridimensionata: {width}x{height} → {new_width}x{new_height}")
    
    # Comprimi in memoria
    output = io.BytesIO()
    img.save(output, format='JPEG', quality=JPEG_QUALITY, optimize=True)
    output.seek(0)
    
    compressed_size = len(output.getvalue())
    reduction = ((original_size - compressed_size) / original_size) * 100
    
    print(f"  Dimensione: {original_size / 1024 / 1024:.2f}MB → {compressed_size / 1024 / 1024:.2f}MB ({reduction:.1f}% riduzione)")
    
    return output

# Carica tutte le foto
photo_files = [f for f in PHOTOS_DIR.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

if not photo_files:
    print(f"Nessuna foto trovata in {PHOTOS_DIR}")
    sys.exit(0)

print(f"Trovate {len(photo_files)} foto da caricare...")
print(f"Configurazione: max {MAX_DIMENSION}px, qualità JPEG {JPEG_QUALITY}%")
print("-" * 60)

uploaded = 0
failed = 0
total_original_size = 0
total_compressed_size = 0

for photo_file in photo_files:
    # Usa il nome del file senza estensione come public_id
    public_id = photo_file.stem
    
    try:
        original_size = photo_file.stat().st_size
        total_original_size += original_size
        
        print(f"\n[{uploaded + failed + 1}/{len(photo_files)}] {photo_file.name}")
        print(f"  Dimensione originale: {original_size / 1024 / 1024:.2f}MB")
        
        # Comprimi l'immagine
        compressed_image = compress_image(photo_file)
        compressed_size = len(compressed_image.getvalue())
        total_compressed_size += compressed_size
        
        # Verifica dimensione dopo compressione
        if compressed_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            print(f"  ⚠️  ATTENZIONE: File ancora troppo grande ({compressed_size / 1024 / 1024:.2f}MB)")
            print(f"     Prova a ridurre MAX_DIMENSION o JPEG_QUALITY nello script")
        
        # Carica su Cloudinary
        print(f"  Caricando su Cloudinary...", end=" ")
        result = cloudinary.uploader.upload(
            compressed_image,
            public_id=public_id,
            folder="face-site",
            overwrite=True,
            resource_type="image"
        )
        
        print(f"✓ OK")
        print(f"  URL: {result.get('secure_url', 'N/A')[:60]}...")
        uploaded += 1
        
    except Exception as e:
        print(f"✗ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

print("\n" + "=" * 60)
print(f"Riepilogo:")
print(f"  Caricate: {uploaded}")
print(f"  Fallite: {failed}")
print(f"  Totale: {len(photo_files)}")
print(f"\nDimensioni:")
print(f"  Originale totale: {total_original_size / 1024 / 1024:.2f}MB")
print(f"  Compressa totale: {total_compressed_size / 1024 / 1024:.2f}MB")
if total_original_size > 0:
    total_reduction = ((total_original_size - total_compressed_size) / total_original_size) * 100
    print(f"  Riduzione totale: {total_reduction:.1f}%")
print("=" * 60)

print(f"\nRiepilogo:")
print(f"  Caricate: {uploaded}")
print(f"  Fallite: {failed}")
print(f"  Totale: {len(photo_files)}")



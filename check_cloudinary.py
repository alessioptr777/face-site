#!/usr/bin/env python3
"""
Script per verificare le foto su Cloudinary
"""

import os
import sys
from pathlib import Path

# Prova a importare cloudinary
try:
    import cloudinary
    import cloudinary.api
    CLOUDINARY_AVAILABLE = True
except ImportError:
    print("❌ ERRORE: cloudinary non installato")
    print("Installa con: pip install cloudinary")
    sys.exit(1)

# Configura Cloudinary dalla variabile d'ambiente
CLOUDINARY_URL = os.getenv("CLOUDINARY_URL", "")
if not CLOUDINARY_URL:
    print("❌ ERRORE: CLOUDINARY_URL non configurata!")
    print("Configura la variabile d'ambiente CLOUDINARY_URL")
    print("Esempio: export CLOUDINARY_URL='cloudinary://api_key:api_secret@cloud_name'")
    sys.exit(1)

try:
    cloudinary.config()
    print("✅ Cloudinary configurato correttamente")
except Exception as e:
    print(f"❌ ERRORE nella configurazione Cloudinary: {e}")
    sys.exit(1)

# Verifica connessione
print("\n🔍 Verificando connessione a Cloudinary...")
try:
    # Prova a fare una chiamata API
    result = cloudinary.api.ping()
    print("✅ Connessione a Cloudinary OK")
except Exception as e:
    print(f"❌ ERRORE nella connessione: {e}")
    sys.exit(1)

# Lista tutte le foto su Cloudinary
print("\n📸 Cercando foto su Cloudinary...")
print("-" * 60)

try:
    # Lista tutte le risorse (foto) su Cloudinary
    # folder="face-site" se usi una cartella
    result = cloudinary.api.resources(
        type="upload",
        resource_type="image",
        max_results=500,  # Massimo 500 foto
        folder="face-site"  # Se usi una cartella
    )
    
    resources = result.get('resources', [])
    
    if not resources:
        print("⚠️  Nessuna foto trovata nella cartella 'face-site'")
        print("\n🔍 Provo a cercare in tutte le cartelle...")
        
        # Prova senza folder
        result = cloudinary.api.resources(
            type="upload",
            resource_type="image",
            max_results=500
        )
        resources = result.get('resources', [])
    
    if resources:
        print(f"✅ Trovate {len(resources)} foto su Cloudinary\n")
        
        # Mostra le prime 20 foto
        print("📋 Prime 20 foto trovate:")
        print("-" * 60)
        for i, resource in enumerate(resources[:20], 1):
            public_id = resource.get('public_id', 'N/A')
            format_type = resource.get('format', 'N/A')
            bytes_size = resource.get('bytes', 0)
            width = resource.get('width', 0)
            height = resource.get('height', 0)
            created_at = resource.get('created_at', 'N/A')
            
            size_mb = bytes_size / 1024 / 1024 if bytes_size else 0
            
            print(f"{i:3d}. {public_id}")
            print(f"     Dimensione: {size_mb:.2f}MB | {width}x{height}px | Formato: {format_type}")
            print(f"     Creato: {created_at}")
            print()
        
        if len(resources) > 20:
            print(f"... e altre {len(resources) - 20} foto")
        
        print("-" * 60)
        print(f"\n📊 Riepilogo:")
        print(f"   Totale foto: {len(resources)}")
        
        # Calcola dimensioni totali
        total_bytes = sum(r.get('bytes', 0) for r in resources)
        total_mb = total_bytes / 1024 / 1024
        print(f"   Dimensione totale: {total_mb:.2f}MB")
        
        # Conta per formato
        formats = {}
        for r in resources:
            fmt = r.get('format', 'unknown')
            formats[fmt] = formats.get(fmt, 0) + 1
        
        print(f"   Formati:")
        for fmt, count in formats.items():
            print(f"     - {fmt}: {count}")
            
    else:
        print("⚠️  Nessuna foto trovata su Cloudinary")
        print("\n💡 Suggerimenti:")
        print("   1. Verifica di aver caricato le foto")
        print("   2. Controlla il nome della cartella (folder)")
        print("   3. Vai su cloudinary.com → Dashboard → Media Library")
        
except cloudinary.api.NotFound:
    print("⚠️  Nessuna foto trovata")
except Exception as e:
    print(f"❌ ERRORE nel recupero foto: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Verifica foto locali per confronto
print("\n" + "=" * 60)
print("📁 Verificando foto locali...")
PHOTOS_DIR = Path(__file__).parent / "photos"

if PHOTOS_DIR.exists():
    local_photos = [f for f in PHOTOS_DIR.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    print(f"✅ Trovate {len(local_photos)} foto locali in {PHOTOS_DIR}")
    
    if local_photos:
        print("\n📋 Prime 10 foto locali:")
        for i, photo in enumerate(local_photos[:10], 1):
            size_mb = photo.stat().st_size / 1024 / 1024
            print(f"{i:3d}. {photo.name} ({size_mb:.2f}MB)")
        
        if len(local_photos) > 10:
            print(f"... e altre {len(local_photos) - 10} foto")
else:
    print(f"⚠️  Cartella {PHOTOS_DIR} non trovata")

print("\n" + "=" * 60)
print("✅ Verifica completata!")


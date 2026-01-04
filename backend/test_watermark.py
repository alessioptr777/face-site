#!/usr/bin/env python3
"""
Script per testare il watermark localmente prima del deploy.
Uso: python test_watermark.py <path_to_photo>
"""

import sys
from pathlib import Path

# Aggiungi il percorso del backend al path
sys.path.insert(0, str(Path(__file__).parent))

from main import _add_watermark

def test_watermark(image_path: str, output_path: str = None):
    """Testa il watermark su un'immagine"""
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"❌ Errore: file non trovato: {image_path}")
        return
    
    print(f"📸 Caricamento immagine: {image_path}")
    
    # Applica watermark
    try:
        watermarked_bytes = _add_watermark(image_path)
        
        # Salva risultato
        if output_path is None:
            output_path = image_path.parent / f"{image_path.stem}_watermarked{image_path.suffix}"
        else:
            output_path = Path(output_path)
        
        with open(output_path, 'wb') as f:
            f.write(watermarked_bytes)
        
        print(f"✅ Watermark applicato con successo!")
        print(f"📁 File salvato: {output_path}")
        print(f"💡 Apri il file per vedere il risultato")
        
    except Exception as e:
        print(f"❌ Errore durante l'applicazione del watermark: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python test_watermark.py <path_to_photo> [output_path]")
        print("\nEsempio:")
        print("  python test_watermark.py photos/_MIT0180.jpg")
        print("  python test_watermark.py photos/_MIT0180.jpg output/test.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    test_watermark(image_path, output_path)







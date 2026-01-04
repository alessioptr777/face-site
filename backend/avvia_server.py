#!/usr/bin/env python3
"""
Script semplice per avviare il server FastAPI
"""
import uvicorn
import sys
import os

# Cambia directory al backend
os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("🚀 Avvio server FaceSite...")
    print("📍 Server disponibile su: http://localhost:8000")
    print("📍 Oppure: http://127.0.0.1:8000")
    print("")
    print("Premi CTRL+C per fermare il server")
    print("")
    
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n✅ Server fermato")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Errore avvio server: {e}")
        print("\nVerifica che:")
        print("1. Tutte le dipendenze siano installate: pip install -r requirements.txt")
        print("2. Il file main.py esista nella cartella backend")
        sys.exit(1)






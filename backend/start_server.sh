#!/bin/bash
# Script per avviare il server di sviluppo

echo "🚀 Avvio server FaceSite..."
echo "📍 Server disponibile su: http://localhost:8000"
echo ""
echo "Premi CTRL+C per fermare il server"
echo ""

cd "$(dirname "$0")"
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000






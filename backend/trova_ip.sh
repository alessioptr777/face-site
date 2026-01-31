#!/bin/bash
# Script per trovare l'IP del Mac sulla rete locale

echo "🔍 Cerca IP del Mac sulla rete locale..."
echo ""

# Trova IP principale (non localhost)
IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}')

if [ -z "$IP" ]; then
    echo "❌ IP non trovato"
    echo ""
    echo "Verifica che:"
    echo "1. Il Mac sia connesso a WiFi"
    echo "2. La connessione sia attiva"
    exit 1
fi

echo "✅ IP trovato: $IP"
echo ""
echo "📱 Su iPhone Safari, vai su:"
echo "   http://$IP:8000"
echo ""
echo "⚠️  IMPORTANTE:"
echo "   - iPhone e Mac devono essere sulla stessa WiFi"
echo "   - Il server deve essere avviato con: python3 avvia_server.py"
echo ""




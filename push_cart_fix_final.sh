#!/bin/bash
cd /Users/metaproos/Desktop/face-site

echo "📦 Verifico stato git..."
git status --short

echo ""
echo "📝 Aggiungo modifiche..."
git add static/index.html

echo ""
echo "💾 Creo commit..."
git commit -m "Fix: add all photos to cart when clicking Buy all button + add Clear cart button"

echo ""
echo "🚀 Faccio push su GitHub..."
git push origin main

echo ""
echo "✅ Push completato!"
echo ""
echo "📋 Hash commit:"
git log -1 --format="%H"
echo ""
echo "📝 Messaggio:"
git log -1 --format="%s"

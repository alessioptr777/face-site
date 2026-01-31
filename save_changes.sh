#!/bin/bash
# Salva le modifiche correnti con un commit
cd "$(dirname "$0")"
echo "=== Salvataggio modifiche ==="
git status --short
git add backend/main.py
git status --short
echo ""
read -p "Messaggio commit (invio = default): " msg
if [ -z "$msg" ]; then
  msg="8 ref embeddings, protection chirurgica det>=0.90, timing [TIME] ms, diagnostica"
fi
git commit -m "$msg"
echo ""
echo "=== Fatto. Per inviare su GitHub: git push origin main ==="
git log -1 --oneline

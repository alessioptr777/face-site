#!/bin/bash
# Deploy: add, commit, push. Render fa auto-deploy su push a main.
set -e
cd "$(dirname "$0")"
git add backend/main.py static/ backend/requirements.txt render.yaml 2>/dev/null || true
git add -u
git status --short
if [ -z "$(git status --porcelain)" ]; then
  echo "Niente da committare."
  exit 0
fi
MSG="${1:-Deploy: aggiornamenti}"
git commit --no-verify -m "$MSG"
git push origin main
echo "Push fatto. Render avvia il deploy in automatico."

# Migrazione Foto a Cloudflare R2

## ✅ Modifiche Completate nel Codice

1. **Upload (`/admin/upload`)**: Ora salva solo su R2
2. **Servizio foto (`/photo/{filename}`)**: Legge solo da R2
3. **Cloudinary**: Rimosso dal servizio foto
4. **Fallback locale**: Rimosso completamente

## 📋 Migrazione Foto Esistenti

### Opzione 1: Eseguire su Render (Consigliato)

1. Vai su Render Dashboard → Il tuo servizio → Shell
2. Esegui:
   ```bash
   cd /opt/render/project/src
   python3 backend/migrate_to_r2.py
   ```

### Opzione 2: Eseguire Localmente

1. Configura le variabili d'ambiente:
   ```bash
   export R2_ENDPOINT_URL="https://..."
   export R2_BUCKET="metaproos-photos"
   export R2_ACCESS_KEY_ID="..."
   export R2_SECRET_ACCESS_KEY="..."
   ```

2. Esegui lo script:
   ```bash
   cd /Users/metaproos/Desktop/face-site
   python3 backend/migrate_to_r2.py
   ```

## 📊 Cosa Fa lo Script

- ✅ Carica tutte le foto da `backend/photos/` su R2
- ⏭️ Salta foto già presenti (evita duplicati)
- 📝 Mostra riepilogo: caricate, saltate, errori

## ⚠️ Importante

Dopo la migrazione:
- ✅ Tutte le foto sono su R2
- ✅ Nuove foto vanno automaticamente su R2
- ✅ Il sistema funziona solo con R2 (niente locale)

## 🔄 Backup Hard Disk Esterno

Le foto su R2 sono il backup online. Per il backup fisico:
- Scarica manualmente da R2 quando vuoi
- Oppure usa uno script di sincronizzazione R2 → hard disk


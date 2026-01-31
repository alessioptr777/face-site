# 📸 Compressione Automatica per Cloudinary

## ✅ Problema Risolto!

Le foto troppo pesanti (>20MB) vengono ora **comprimate automaticamente** prima dell'upload su Cloudinary.

## 🎯 Cosa Fa lo Script

1. **Legge la foto originale** (anche se 20MB+)
2. **Ridimensiona** se >2000px (mantiene proporzioni)
3. **Comprime** a qualità JPEG 85% (ottima qualità visiva)
4. **Carica su Cloudinary** (file finale <5MB, spesso <2MB)

## 📊 Risultati Attesi

- **Foto 20MB** → **~2-3MB** dopo compressione
- **Qualità visiva**: Eccellente (85% è il sweet spot)
- **Riduzione**: 80-90% in media
- **Upload**: Più veloce
- **Download**: Più veloce per gli utenti

## 🚀 Come Usare

### 1. Assicurati di avere le dipendenze

```bash
pip install cloudinary pillow
```

(Sei già a posto, sono già in `requirements.txt`)

### 2. Configura Cloudinary

```bash
export CLOUDINARY_URL='cloudinary://api_key:api_secret@cloud_name'
```

Oppure aggiungi su Render come variabile d'ambiente.

### 3. Esegui lo script

```bash
cd backend
python3 upload_to_cloudinary.py
```

## 📋 Output Esempio

```
Trovate 10 foto da caricare...
Configurazione: max 2000px, qualità JPEG 85%
------------------------------------------------------------

[1/10] _MIT0180.jpg
  Dimensione originale: 22.45MB
  Ridimensionata: 4000x3000 → 2000x1500
  Dimensione: 22.45MB → 2.15MB (90.4% riduzione)
  Caricando su Cloudinary... ✓ OK
  URL: https://res.cloudinary.com/...

[2/10] _MIT0181.jpg
  Dimensione originale: 18.32MB
  Dimensione: 18.32MB → 1.87MB (89.8% riduzione)
  Caricando su Cloudinary... ✓ OK
  ...

============================================================
Riepilogo:
  Caricate: 10
  Fallite: 0
  Totale: 10

Dimensioni:
  Originale totale: 195.23MB
  Compressa totale: 19.45MB
  Riduzione totale: 90.0%
============================================================
```

## ⚙️ Personalizzazione

Puoi modificare i parametri nello script:

```python
MAX_DIMENSION = 2000  # Lato massimo (aumenta per qualità superiore)
JPEG_QUALITY = 85     # Qualità 1-100 (85-90 è ottimo)
```

**Raccomandazioni:**
- **MAX_DIMENSION = 2000**: Perfetto per web, foto ancora molto nitide
- **MAX_DIMENSION = 3000**: Se vuoi qualità superiore (file più grandi)
- **JPEG_QUALITY = 85**: Sweet spot qualità/dimensione
- **JPEG_QUALITY = 90**: Qualità superiore (file ~30% più grandi)

## ⚠️ Note Importanti

1. **Le foto originali NON vengono modificate** - Solo la versione caricata su Cloudinary è compressa
2. **Le foto locali restano originali** - Puoi sempre ricaricare con impostazioni diverse
3. **Cloudinary applica ulteriore ottimizzazione** - Le foto vengono servite ottimizzate automaticamente

## 🔍 Verifica

Dopo l'upload, verifica su Cloudinary Dashboard:
- Le foto dovrebbero essere <5MB
- La qualità visiva dovrebbe essere eccellente
- Le dimensioni dovrebbero essere ≤2000px

## ❓ Problemi?

### Foto ancora troppo grande dopo compressione?

1. Riduci `MAX_DIMENSION` a 1500 o 1800
2. Riduci `JPEG_QUALITY` a 80
3. Verifica che la foto non sia già compressa (potrebbe essere già ottimale)

### Qualità troppo bassa?

1. Aumenta `JPEG_QUALITY` a 90
2. Aumenta `MAX_DIMENSION` a 2500 o 3000
3. Nota: file più grandi = upload più lento

### Errore "File too large"?

1. Verifica che la compressione funzioni (controlla output)
2. Se il file è ancora >10MB, riduci `MAX_DIMENSION` o `JPEG_QUALITY`
3. Alcune foto RAW potrebbero richiedere compressione più aggressiva

---

**Pronto per caricare le tue foto! 🚀**






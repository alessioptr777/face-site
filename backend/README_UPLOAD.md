# Come caricare le foto giornaliere

## Processo di caricamento foto

### 1. Carica le foto nella cartella corretta

**Opzione A: Storage Locale (backend/photos/)**
```bash
# Copia le foto nella cartella backend/photos/
cp /percorso/tue/foto/*.jpg backend/photos/
```

**Opzione B: Cloudinary (consigliato per produzione)**
- Usa lo script `upload_to_cloudinary.py` per caricare le foto
- Oppure carica manualmente dal dashboard Cloudinary

### 2. Formato nome file

Le foto devono avere la **data del tour** nel nome file per permettere il filtro per data:

**Formati accettati:**
- `20241231_IMG001.jpg` (YYYYMMDD)
- `2024-12-31_IMG001.jpg` (YYYY-MM-DD)
- `TOUR_20241231_001.jpg`
- Qualsiasi formato che contenga YYYYMMDD o YYYY-MM-DD

### 3. Indicizza le foto

Dopo aver caricato le foto, devi indicizzarle per il face matching:

```bash
cd backend
python index_folder.py
```

Questo script:
- Analizza tutte le foto in `backend/photos/`
- Estrae i volti e crea gli embedding
- Salva l'indice FAISS in `backend/data/faces.index`
- Salva i metadata in `backend/data/faces.meta.jsonl`
- Salva le foto senza volti (di spalle) in `backend/data/back_photos.jsonl`

### 4. Deploy su Render

Se usi storage locale:
1. Le foto vanno committate su git (se piccole) OPPURE
2. Caricate manualmente su Render tramite SSH/console

Se usi Cloudinary:
1. Le foto sono già online
2. Basta indicizzare e committare solo i file di indice

## Workflow giornaliero consigliato

1. **Fine giornata lavorativa:**
   ```bash
   # 1. Carica le foto (locale o Cloudinary)
   cp /foto/giornata/*.jpg backend/photos/
   
   # 2. Indicizza
   python backend/index_folder.py
   
   # 3. Commit e push (solo se locale, altrimenti solo indici)
   git add backend/photos/*.jpg backend/data/
   git commit -m "Aggiunte foto del 2024-12-31"
   git push
   ```

2. **Render si aggiorna automaticamente** e le foto sono disponibili

## Note importanti

- **Dimensione file**: Se le foto sono grandi (>10MB), considera Cloudinary
- **Privacy**: Le foto con watermark sono protette, quelle pagate no
- **Backup**: Fai sempre backup delle foto originali
- **Performance**: L'indicizzazione può richiedere tempo (1-2 min per 100 foto)

## Troubleshooting

**Errore "Index files not found":**
- Esegui `python index_folder.py` per creare gli indici

**Foto non trovate dopo upload:**
- Verifica che siano in `backend/photos/`
- Verifica che siano indicizzate
- Controlla i log su Render






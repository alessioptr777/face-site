# Report diagnostica: 11 foto in R2 → match mostra 8

**Obiettivo:** capire se le 3 foto mancanti non vengono indicizzate (face detect fallisce) o vengono indicizzate ma scartate dal match.  
**Vincolo:** solo diagnostica e log; nessuna modifica alla logica.

---

## Implementazione applicata (TASK 2, 3, 4)

- **File toccato:** `backend/main.py`
- **TASK 2:** Aggiunte `_diag_bucket_for_face`, `_log_index_diag_photo`; log `[DIAG_INDEX_PHOTO]` / `[DIAG_INDEX_FACE]` per ogni foto in rebuild e sync; `[INDEX_SUMMARY]` a fine rebuild e a fine sync.
- **TASK 3:** Aggiunto `GET /admin/index/diag` (auth come gli altri `/admin`): risponde con `indexed`, `in_r2_not_indexed`, `summary`.
- **TASK 4:** Aggiunta `_save_index_diag_images`: salva in `DEBUG_INDEX_DIR` (default `static/debug`) `index_diag_<base>_bbox.png` e `index_diag_<base>_face<N>_det<>_area<>.png` solo se la foto ha ≤1 faccia e `DIAG_INDEX_DEBUG_IMAGES=1`.

---

## TASK 1 — Punti esatti nel codice

### 1) Indexing / sync delle foto (file e funzioni)

| Ruolo | File | Funzione | Righe |
|-------|------|---------|-------|
| **Full rebuild** (tutte le foto da R2) | `backend/main.py` | `index_new_r2_photos()` (async, definita dentro `load_face_model()`) | ~3181–3385 |
| **Sync incrementale** (add/remove vs R2) | `backend/main.py` | `sync_index_with_r2_incremental()` | 3708–~3935 |
| **Trigger rebuild da admin** | `backend/main.py` | `admin_index_rebuild()` → chiama `index_new_r2_photos()` | 4015–4031 |
| **Trigger sync da admin** | `backend/main.py` | `admin_index_sync()` → chiama `sync_index_with_r2_incremental()` | 3995–4012 |

- **Rebuild:** lista R2 con prefix `R2_PHOTOS_PREFIX` (~3200–3218), poi loop su `original_photos` (~3264–3330): per ogni chiave scarica foto, `face_app.get(img)`, fallback `_indexing_fallback_split_faces`, poi per ogni faccia aggiunge embedding a `new_faiss_index` e riga a `new_meta_rows`.
- **Sync:** lista R2 (~3725–3765), calcola `to_add` / `to_remove` (~3766–3770), rimuove da indice le foto in `to_remove`, poi per ogni `filename` in `to_add` (~3836–3912) scarica, `face_app.get(img)`, fallback, aggiunge a `new_faiss_index` / `new_meta_rows`.

### 2) Face detection in indexing (detector, det_size, threshold, filtri)

| Cosa | File | Dove | Valori |
|------|------|------|--------|
| **Detector principale** | `backend/main.py` | `load_face_model()` | 3026–3028: `FaceAnalysis(name="buffalo_l")`, `det_size=(1024, 1024)` (default `det_thresh` InsightFace) |
| **Detector loose (solo fallback)** | `backend/main.py` | `load_face_model()` | 3037–3038: `face_app_loose`, `det_size=(1024, 1024)`, **`det_thresh=0.25`** |
| **Chiamata detection in rebuild** | `backend/main.py` | dentro `index_new_r2_photos()` | 3282: `faces = face_app.get(img)` |
| **Chiamata detection in sync** | `backend/main.py` | dentro `sync_index_with_r2_incremental()` | 3866: `faces = face_app.get(img)` |
| **Fallback (1 bbox sospetta, baci/profili)** | `backend/main.py` | `_indexing_fallback_split_faces()` | 1365–~1600: usa `detector.get(img, max_num=10)` (può essere `face_app_loose`), filtri IoU/centro, ROI, split left/right, multi-scala; dedup IoU 0.5 in `_dedup_iou` (~1444–1466); log "dedup: scarto faccia" ~1463 |
| **Filtro facce “deboli”** | `backend/main.py` | costante | 116: `MIN_FACE_DET_SCORE_FOR_FILTER = 0.75` (usato nel filtro `issubset(valid_faces)` altrove, non nel flusso indexing rebuild/sync) |

In indexing **non** c’è un filtro esplicito per `det_score` prima di aggiungere a `new_meta_rows`: tutte le facce restituite da `face_app.get(img)` (e dal fallback) vengono indicizzate. Il “filtro” è solo: nessuna faccia → la foto non viene aggiunta (skip con `continue`).

### 3) Match (scoring, soglie, bucket small/medium/large)

| Cosa | File | Dove | Dettaglio |
|------|------|------|-----------|
| **Entry point match selfie** | `backend/main.py` | endpoint che chiama la logica match | Cerca `match_selfie` / `_match_selfie_*`; lettura indice sotto lock ~6271–6276 |
| **Stato indice in match** | `backend/main.py` | ~6271–6276 | `local_faiss_index`, `local_meta_rows` copiati sotto `indexing_lock`; log `[MATCH_STATE] vectors_in_index=... meta_rows=...` 6276 |
| **Ricerca FAISS** | `backend/main.py` | ~6468–6514 | `local_faiss_index.search(ref_emb.reshape(1, -1), top_k)`; aggregazione per `r2_key` in `candidates_by_photo`; per ogni candidato si tiene `best_score`, `det_score`, `area`, `ref_max` |
| **Bucket (small/medium/large)** | `backend/main.py` | 6592–6596 | `area < 30000 or det_score_val < 0.75` → **small**; altrimenti `area < 60000 or det_score_val < 0.85` → **medium**; altrimenti **large** |
| **Soglie dinamiche min_score** | `backend/main.py` | 6413–6456 | `_dynamic_min_score(det_score_val, area)` (molte regole per area/det; default 0.35) |
| **Margin minimo** | `backend/main.py` | 6457–6460 | `_dynamic_margin_min(det_score_val, area)` → 0.03 (small) o 0.015 |
| **Rifiuto esplicito (reject_reason)** | `backend/main.py` | 6598–6685 (e hit/margin dopo) | det≥0.90 → score≥0.50 (o eccezione hits≥2 e score≥0.47); det≥0.85 → score≥0.50; area≥150000 e det 0.70–0.85 → score≥0.25; det≥0.80 → score≥0.30; det≥0.78 → score≥0.25; poi min_score_dyn, hits richiesti, margin |
| **Log ACCEPTED/REJECTED** | `backend/main.py` | ~6828–6832, 6876, 6883 | `[ACCEPTED]` / `[DIAG_SCORING] REJECTED ... reason=...` |
| **Statistiche filtro** | `backend/main.py` | 6572, 6604, 7069 | `filtered_by_score`, `filtered_by_margin`; log `[MATCH_STATS] filtered_by_score=... filtered_by_margin=...` |
| **Log top 10 rifiutati** | `backend/main.py` | ~7092 | `[MATCH_REJECTED] Top 10: ...` |

### 4) meta_rows e vectors_in_index (dove si salvano e come si aggiornano)

| Cosa | File | Dove | Comportamento |
|------|------|------|---------------|
| **Variabili globali** | `backend/main.py` | dichiarate in alto (cerca `faiss_index =` / `meta_rows =`) | `faiss_index`, `meta_rows`, `vectors_in_index` (quest’ultimo probabilmente derivato da `faiss_index.ntotal`) |
| **Rebuild: swap** | `backend/main.py` | 3333–3339 | `faiss_index = new_faiss_index`, `meta_rows = new_meta_rows`, `vectors_in_index = faiss_index.ntotal`; poi salvataggio su R2 (faces.index, faces.meta.jsonl) ~3341–3355 |
| **Sync incrementale: swap** | `backend/main.py` | 3912–3918 | Stesso schema: `faiss_index = new_faiss_index`, `meta_rows = new_meta_rows`, `vectors_in_index = faiss_index.ntotal`; poi salvataggio R2 ~3922–3932 |
| **Lettura da disco (avvio)** | `backend/main.py` | 3058–3163 (R2) e 3151–3163 (locale) | In `load_face_model()`: si carica `faces.index` e `faces.meta.jsonl` da R2 (o da `INDEX_PATH`/`META_PATH`), si assegnano `faiss_index` e `meta_rows` |
| **Caricamento meta da file** | `backend/main.py` | `_load_meta_jsonl()` | 1911: legge JSONL da `meta_path` e restituisce lista di dict |

Ogni riga di `meta_rows` ha almeno: `r2_key`, `display_name`, `photo_id`, `bbox` (4 float), `det_score` (float). L’area in match è calcolata da `bbox` con `_compute_face_area(row)` (~6405–6411).

---

## TASK 2 — Log/diagnostica da aggiungere (solo log, no cambio logica)

### In sync/indexing, per OGNI foto

- **photo_id** (es. `r2_key` o `Path(r2_key).name`).
- **Numero facce:** `n_faces_strict` = `len(faces)` subito dopo `face_app.get(img)`; dopo `_indexing_fallback_split_faces(…)` → `n_faces_after_fallback` = `len(faces)`; loggare entrambi.
- **Per ogni faccia (dopo fallback):** `det_score`, `bbox`, `area` (= (x2-x1)*(y2-y1)), `area_ratio` = area / (img_w * img_h), `bucket` = small/medium/large (stessa formula del match: area < 30000 or det < 0.75 → small; area < 60000 or det < 0.85 → medium; else large).
- **Motivo filtro:** se una faccia viene scartata *dentro* `_indexing_fallback_split_faces` (es. dedup IoU), è già loggato con `[INDEX_FALLBACK] dedup: scarto faccia ...`. Per chiarezza si può aggiungere un log unico per foto: “faces_strict=N, faces_after_fallback=M, faces_to_index=M” (e se M=0, motivo: “no_faces” o “all_filtered”).
- **Totale finale:** `faces_to_index` = `len(faces)` usato per il loop che fa `new_meta_rows.append(...)`.

**Punti di inserimento:**

- **Rebuild:** subito dopo `faces = face_app.get(img)` (r. ~3282): memorizzare `n_faces_strict = len(faces)`; se `not faces` loggare photo_id, n_faces_strict=0, motivo=no_faces e fare `continue`. Dopo `faces = _indexing_fallback_split_faces(...)` (r. ~3289): log per-foto (photo_id, n_faces_strict, n_faces_after_fallback, per ogni faccia det_score/bbox/area/area_ratio/bucket, faces_to_index).
- **Sync incrementale:** stesso schema dopo `faces = face_app.get(img)` (r. ~3866) e dopo `_indexing_fallback_split_faces` (r. ~3874), usando `filename` come photo_id.

### Riepilogo a fine sync/rebuild

- `total_photos_in_r2`: numero di chiavi “originali” (es. `len(original_photos)` in rebuild, `len(r2_originals_keys)` in sync).
- `total_photos_indexed`: numero di foto che hanno almeno una faccia in indice = `len(set(m["r2_key"] or m["photo_id"] for m in new_meta_rows))` (dopo il loop, prima dello swap).
- `faces_total`: già presente (conteggio di tutte le facce aggiunte).
- `vectors_in_index`: `new_faiss_index.ntotal` (o dopo swap `faiss_index.ntotal`).
- `meta_rows`: `len(new_meta_rows)`.

Log unico tipo:

```text
[INDEX_SUMMARY] total_photos_in_r2=11 total_photos_indexed=11 faces_total=16 vectors_in_index=16 meta_rows=16
```

(Se una foto ha 0 facce, `total_photos_indexed` < `total_photos_in_r2`.)

---

## TASK 3 — Diagnostica post-sync (endpoint o CLI)

- **Endpoint consigliato:** `GET /admin/index/diag` (protetto da stessa auth admin degli altri `/admin`).
- **Comportamento:**
  1. Leggere `meta_rows` (stato in memoria) e, se serve, lista R2 con prefix `R2_PHOTOS_PREFIX` (stesso filtro “solo originali” di sync).
  2. **Lista photo_id indicizzati:** da `meta_rows` raggruppare per `r2_key` (o `photo_id`); per ogni photo_id: numero facce, per ogni faccia `det_score`, `area` (da bbox), `bucket` (stessa formula del match).
  3. **Photo_id in R2 ma NON indicizzati:** set R2 originali vs set di photo_id presenti in `meta_rows` (confronto per basename se necessario); per ognuno “non indicizzato” si può solo inferire “0 facce in indexing” (nessun dettaglio senza rieseguire il detector).
- **Output:** JSON ad esempio:
  - `indexed`: lista di `{ "photo_id": "...", "face_count": N, "faces": [ { "det_score": ..., "area": ..., "bucket": "small"|"medium"|"large" } ] }`
  - `in_r2_not_indexed`: [ "photo1.jpg", "photo2.jpg" ]
  - `summary`: `{ "vectors_in_index", "meta_rows", "photos_in_r2", "photos_indexed" }`

Implementazione: nuova route che usa `meta_rows` e (opzionale) chiama `list_objects_v2` con `R2_PHOTOS_PREFIX` per costruire `in_r2_not_indexed`.

---

## TASK 4 — Debug immagini (solo casi problematici)

- **Condizione:** solo se una foto ha **≤ 1 faccia** indicizzata (cioè `len(faces)` dopo fallback ≤ 1).
- **Directory:** `/static/debug/` (es. `STATIC_DIR / "debug"` o `DEBUG_INDEX_DIR` se già definito).
- **Contenuto da salvare:**
  1. **Immagine con bbox disegnate:** stessa immagine usata in indexing (`img` dopo decode), con ogni `face.bbox` disegnata (es. `cv2.rectangle`); nome tipo `index_diag_<basename>_bbox.png`.
  2. **Crop per ogni faccia:** ritaglio `img[y1:y2, x1:x2]` per ogni faccia; nome tipo `index_diag_<basename>_face<idx>_det<det_score>_area<area>.png` (anche facce eventualmente “filtrate” prima del fallback non sono più disponibili qui; si salvano le facce *dopo* fallback che sono le stesse indicizzate; se ne hai solo 0 o 1, hai al massimo 1 crop).
- **Punto di chiamata:** subito dopo il log per-foto in rebuild e in sync, se `len(faces) <= 1` e una variabile tipo `DIAG_INDEX_DEBUG_IMAGES` è True: chiamare una funzione che scrive in `STATIC_DIR / "debug"` (o `DEBUG_INDEX_DIR`) i file sopra. Creare la directory con `mkdir(parents=True, exist_ok=True)`.

Nessun cambio alla logica di indexing/match: solo scrittura file e (opzionale) log “saved index_diag_...”.

---

## OUTPUT RICHIESTO (riepilogo)

### Elenco file toccati

- **Solo:** `backend/main.py`  
  (eventualmente `static/debug/` creato a runtime; nessun altro file di codice.)

### Patch / diff (solo diagnostica)

Le modifiche sono solo:

1. **TASK 2:**  
   - In rebuild: dopo 3282 memorizzare `n_faces_strict`; se `not faces` log + continue; dopo 3289 log per-foto (photo_id, n_faces_strict, n_faces_after_fallback, per faccia det_score/bbox/area/area_ratio/bucket, faces_to_index); dopo il loop (prima di 3333) log `[INDEX_SUMMARY] total_photos_in_r2=... total_photos_indexed=... faces_total=... vectors_in_index=... meta_rows=...`.  
   - In sync: stesso schema dopo 3866 e 3874; riepilogo dopo il loop prima di 3912.

2. **TASK 3:**  
   - Aggiungere `@app.get("/admin/index/diag")` che legge `meta_rows`, opzionalmente lista R2, e restituisce JSON `indexed`, `in_r2_not_indexed`, `summary`.

3. **TASK 4:**  
   - Se `len(faces) <= 1` e flag debug: salvare `index_diag_<basename>_bbox.png` e `index_diag_<basename>_face<idx>_det<>_area<>.png` in `STATIC_DIR/debug` (o `DEBUG_INDEX_DIR`).

Non è incluso un diff testuale completo per brevità; le righe e i punti sopra sono sufficienti per applicare le patch a mano o generare il diff.

### Esempio di log atteso dopo un sync completo con 11 foto

```text
[INDEXING] SYNC start
[INDEXING] list prefix='originals/'
[INDEXING] list total keys=...
[INDEXING] after filter originals: 11 keys, sample=[...]
[INDEXING] r2_originals=11 indexed=... to_add=... to_remove=...
...
[DIAG_INDEX_PHOTO] photo_id=IMG_1129_maree.jpg n_faces_strict=1 n_faces_after_fallback=2 faces_to_index=2
[DIAG_INDEX_FACE] photo_id=IMG_1129_maree.jpg face_idx=0 det_score=0.774 bbox=[...] area=12000 area_ratio=0.05 bucket=small
[DIAG_INDEX_FACE] photo_id=IMG_1129_maree.jpg face_idx=1 det_score=0.383 bbox=[...] area=8000 area_ratio=0.03 bucket=small
...
[INDEX_FALLBACK] FINAL faces_to_index=2 for IMG_1129_maree.jpg
...
[INDEX_SUMMARY] total_photos_in_r2=11 total_photos_indexed=11 faces_total=16 vectors_in_index=16 meta_rows=16
[INDEX_SWAP] vectors_in_index=16 meta_rows=16 hash=...
[INDEXING] faces_total=16 vectors_in_index=16 meta_rows=16
```

Se una delle 11 foto ha 0 facce:

```text
[DIAG_INDEX_PHOTO] photo_id=Qualcosa.jpg n_faces_strict=0 n_faces_after_fallback=0 faces_to_index=0 reason=no_faces
...
[INDEX_SUMMARY] total_photos_in_r2=11 total_photos_indexed=10 faces_total=14 vectors_in_index=14 meta_rows=14
```

### Comandi per far partire sync e leggere il report

- **Render (o ovunque l’app sia in esecuzione):**
  1. **Sync incrementale:**  
     `curl -X POST "https://face-site.onrender.com/admin/index/sync?password=TUO_ADMIN_PASSWORD"`  
     (sostituire con il metodo di auth usato, es. header o query param.)
  2. **Rebuild completo:**  
     `curl -X POST "https://face-site.onrender.com/admin/index/rebuild?password=TUO_ADMIN_PASSWORD"`.
  3. **Leggere il report:**  
     - Log: dalla dashboard Render (Logs) cercare le righe `[DIAG_INDEX_PHOTO]`, `[INDEX_SUMMARY]`, `[INDEX_FALLBACK]`.  
     - Se implementi TASK 3:  
       `curl "https://face-site.onrender.com/admin/index/diag?password=TUO_ADMIN_PASSWORD"`.

- **Locale:**
  1. Avviare il backend (es. `uvicorn backend.main:app --reload` dalla root del repo).
  2. Sync: `curl -X POST "http://127.0.0.1:8000/admin/index/sync?password=..."`.
  3. Rebuild: `curl -X POST "http://127.0.0.1:8000/admin/index/rebuild?password=..."`.
  4. Log: in terminale dove gira uvicorn; diag: `curl "http://127.0.0.1:8000/admin/index/diag?password=..."`.
  5. Debug immagini: in `static/debug/` (o `DEBUG_INDEX_DIR`) dopo un run con foto con ≤1 faccia.

---

## Interpretazione rapida per “8 vs 11”

- **Se `total_photos_indexed` = 8 e `total_photos_in_r2` = 11:** le 3 mancanti non hanno mai messo una faccia in indice (0 facce in indexing, probabilmente face detect o fallback non le trova).
- **Se `total_photos_indexed` = 11:** tutte le foto sono in indice; le 3 “mancanti” nel match sono scartate in **match** (soglie/det_score/area/bucket/margin). Controllare i log `[DIAG_SCORING] REJECTED` e `[MATCH_REJECTED] Top 10` per vedere `reason` e soglie.

I tuoi log mostrano `vectors_in_index=16 meta_rows=16` e `[DIAG_SCORING] REJECTED r2_key=IMG_1129_maree.jpg ... score=0.204<0.31` e altri REJECTED: quindi **IMG_1129_maree.jpg è indicizzata** (c’è una faccia in indice) ma **rifiutata in match** per score basso. Le altre 2 delle 3 “mancanti” si possono identificare confrontando gli 8 ACCEPTED con l’elenco delle 11 foto e controllando per ciascuna r2_key i log REJECTED e il motivo.

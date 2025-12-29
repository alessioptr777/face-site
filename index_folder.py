import os, json, glob
import re
from datetime import datetime
import numpy as np
import cv2
import faiss
from insightface.app import FaceAnalysis

# CARTELLA DOVE HAI MESSO LE FOTO
PHOTOS_DIR = "../photos"

# CARTELLA DOVE SALVIAMO INDICE E META
OUT_DIR = "./data"
os.makedirs(OUT_DIR, exist_ok=True)

INDEX_PATH = os.path.join(OUT_DIR, "faces.index")
META_PATH  = os.path.join(OUT_DIR, "faces.meta.jsonl")
BACK_PHOTOS_PATH = os.path.join(OUT_DIR, "back_photos.jsonl")  # Foto senza volti

EMB_DIM = 512

def norm(v: np.ndarray) -> np.ndarray:
    v = v.astype("float32")
    return v / (np.linalg.norm(v) + 1e-8)

def extract_date_from_filename(filename: str) -> str:
    """Estrae la data dal filename (formato YYYYMMDD o YYYY-MM-DD)"""
    # Cerca pattern YYYYMMDD
    match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
    if match:
        year, month, day = match.groups()
        return f"{year}-{month}-{day}"
    # Cerca pattern YYYY-MM-DD
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})', filename)
    if match:
        return match.group(0)
    # Se non trova, prova a estrarre dal timestamp del file
    return None

print("Carico motore facciale...")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(1024, 1024))

print("Creo indice...")
index = faiss.IndexFlatIP(EMB_DIM)

img_paths = sorted(
    glob.glob(os.path.join(PHOTOS_DIR, "*.jpg")) +
    glob.glob(os.path.join(PHOTOS_DIR, "*.jpeg")) +
    glob.glob(os.path.join(PHOTOS_DIR, "*.png"))
)

print("Foto trovate:", len(img_paths))

face_count = 0
back_photos_count = 0

with open(META_PATH, "w", encoding="utf-8") as meta_f, \
     open(BACK_PHOTOS_PATH, "w", encoding="utf-8") as back_f:
    
    for p in img_paths:
        photo_id = os.path.basename(p)
        img = cv2.imread(p)
        if img is None:
            continue

        # Estrai data dal filename
        tour_date = extract_date_from_filename(photo_id)
        
        faces = app.get(img)
        
        if not faces:
            # Foto senza volti (di spalle) - salva come back photo
            back_record = {
                "photo_id": photo_id,
                "has_face": False,
                "tour_date": tour_date,
            }
            back_f.write(json.dumps(back_record, ensure_ascii=False) + "\n")
            back_photos_count += 1
            continue

        # Foto con volti - indicizza ogni volto
        for f in faces:
            emb = norm(f.embedding)
            index.add(emb.reshape(1, -1))

            record = {
                "face_idx": face_count,
                "photo_id": photo_id,
                "has_face": True,
                "tour_date": tour_date,
                "det_score": float(getattr(f, "det_score", 0.0)),
                "bbox": [float(x) for x in f.bbox.tolist()],
            }
            meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            face_count += 1

faiss.write_index(index, INDEX_PATH)

print("=" * 60)
print("FACCIE INDICIZZATE:", face_count)
print("FOTO DI SPALLE (senza volti):", back_photos_count)
print("SALVATO INDICE:", INDEX_PATH)
print("SALVATO META:", META_PATH)
print("SALVATO BACK PHOTOS:", BACK_PHOTOS_PATH)
print("=" * 60)


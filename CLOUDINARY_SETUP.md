# Setup Cloudinary per Storage Foto

## Perché Cloudinary?

- ✅ CDN globale (immagini caricate velocemente ovunque)
- ✅ Free tier generoso (25GB storage, 25GB bandwidth/mese)
- ✅ Nessun problema con storage effimero su Render
- ✅ Ottimizzazione automatica immagini

## Setup Rapido

### 1. Crea account Cloudinary

1. Vai su https://cloudinary.com/users/register/free
2. Registrati (gratis)
3. Dopo il login, vai su Dashboard
4. Copia la **Cloudinary URL** (formato: `cloudinary://api_key:api_secret@cloud_name`)

### 2. Configura su Render

1. Vai su Render Dashboard → Il tuo servizio
2. Environment → Add Environment Variable
3. Nome: `CLOUDINARY_URL`
4. Valore: incolla la Cloudinary URL
5. Salva

### 3. Carica le foto

**Opzione A: Da locale (prima del deploy)**

```bash
cd backend
export CLOUDINARY_URL='cloudinary://...'  # La tua URL
pip install cloudinary
python upload_to_cloudinary.py
```

**Opzione B: Da Render (dopo deploy)**

1. Vai su Render Dashboard → Shell
2. Esegui:
```bash
cd backend
pip install cloudinary
export CLOUDINARY_URL='cloudinary://...'  # La tua URL
python upload_to_cloudinary.py
```

### 4. Verifica

Dopo il deploy, le foto verranno servite da Cloudinary automaticamente!

Testa: `https://tuo-sito.onrender.com/photo/_MIT0180.jpg`

## Come Funziona

- Se `CLOUDINARY_URL` è configurata → foto servite da Cloudinary CDN
- Se non configurata → fallback a file locali (come prima)

Il frontend non cambia: continua a chiamare `/photo/<filename>.jpg`

## Note

- Le foto su Cloudinary usano il nome file (senza estensione) come `public_id`
- Esempio: `_MIT0180.jpg` → public_id: `_MIT0180`
- Le foto vengono organizzate nella cartella `face-site` su Cloudinary (opzionale)



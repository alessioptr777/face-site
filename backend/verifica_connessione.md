# 🔧 RISOLUZIONE: "Connessione al server non riuscita"

## ✅ Il server è già avviato correttamente!

Il server è in ascolto su `*.8000`, quindi è configurato per accettare connessioni dalla rete locale.

---

## 🔍 Possibili cause e soluzioni:

### 1. **Firewall del Mac blocca le connessioni** (PIÙ PROBABILE)

**Soluzione:**
1. Apri **Preferenze di Sistema** → **Sicurezza e Privacy** → **Firewall**
2. Se il firewall è **attivo**, clicca su **Opzioni Firewall...**
3. Cerca **Python** o **uvicorn** nella lista
4. Se non c'è, aggiungi manualmente:
   - Clicca **+**
   - Vai su `/usr/bin/python3` o `/usr/local/bin/python3`
   - Seleziona **Consenti connessioni in entrata**
5. **OPPURE** disabilita temporaneamente il firewall per testare

---

### 2. **iPhone e Mac non sono sulla stessa WiFi**

**Verifica:**
- Su iPhone: **Impostazioni** → **WiFi** → Vedi nome rete
- Su Mac: **Preferenze di Sistema** → **Rete** → Vedi nome rete WiFi
- Devono essere **identici**

---

### 3. **Prova da Mac prima**

Sul Mac, apri Safari e vai su:
```
http://192.168.1.98:8000
```

Se funziona sul Mac ma non su iPhone, è un problema di rete/firewall.

---

### 4. **Riavvia il server**

Se il server è già avviato, fermalo (CTRL+C nel terminale) e riavvialo:

```bash
cd /Users/metaproos/Desktop/face-site/backend
python3 avvia_server.py
```

Assicurati di vedere:
```
📍 Server disponibile su: http://localhost:8000
```

---

### 5. **Prova con IP diverso**

A volte l'IP può cambiare. Verifica di nuovo l'IP:

**Metodo veloce:**
- **Preferenze di Sistema** → **Rete** → **WiFi** → Vedi **Indirizzo IP**

---

## ✅ Checklist rapida:

- [ ] Server avviato (`python3 avvia_server.py`)
- [ ] Firewall Mac disabilitato o Python autorizzato
- [ ] iPhone e Mac sulla stessa WiFi
- [ ] IP corretto: `192.168.1.98`
- [ ] URL corretto su iPhone: `http://192.168.1.98:8000` (con `http://` non `https://`)

---

## 🎯 Test rapido:

1. **Sul Mac**, apri Terminale e digita:
   ```bash
   curl http://192.168.1.98:8000/health
   ```
   
   Dovresti vedere: `{"status":"ok",...}`

2. **Se funziona sul Mac ma non su iPhone**, è il firewall.

3. **Se non funziona neanche sul Mac**, riavvia il server.

---

## 💡 Soluzione rapida (per testare):

**Disabilita temporaneamente il firewall:**
1. **Preferenze di Sistema** → **Sicurezza e Privacy** → **Firewall**
2. Clicca sul lucchetto per sbloccare
3. Clicca **Disattiva Firewall** (temporaneamente)
4. Prova di nuovo su iPhone
5. **Riattiva il firewall dopo il test**

---

## 📱 URL corretto su iPhone Safari:

```
http://192.168.1.98:8000
```

**IMPORTANTE:**
- Usa `http://` (non `https://`)
- Includi la porta `:8000`
- Non aggiungere `/` alla fine (o aggiungilo, dovrebbe funzionare comunque)




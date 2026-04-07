# ── C2 — Backend (orchestrateur) ─────────────────────────────
# Rôle : reçoit l'audio du frontend, extrait les features,
#        appelle C3 (model-api) pour la prédiction,
#        sauvegarde l'audio dans V1 (pseudo-label).
# ─────────────────────────────────────────────────────────────

FROM python:3.12-slim

# Dépendances système :
#   libsndfile1 + ffmpeg  — requis par librosa / soundfile
#   curl                  — requis pour le HEALTHCHECK Docker
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dépendances Python (couche de cache séparée)
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Code source
COPY features.py .
COPY backend/ ./backend/

# Dossier cible pour les audios (sera remplacé par le volume V1)
RUN mkdir -p /data/dataset

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=20s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]

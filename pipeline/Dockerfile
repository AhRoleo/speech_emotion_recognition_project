# ── C4 — Pipeline d'entraînement ─────────────────────────────
FROM python:3.12-slim

# libsndfile1 + ffmpeg pour librosa / soundfile
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pipeline/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Code source
COPY features.py .
COPY pipeline/train.py ./pipeline/train.py

# Points de montage des volumes (créés au démarrage par docker-compose)
RUN mkdir -p /data/dataset /models

CMD ["python", "pipeline/train.py"]

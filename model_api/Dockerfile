# ── C3 — Model API ────────────────────────────────────────────
FROM python:3.12-slim

# curl requis pour le HEALTHCHECK Docker
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dépendances Python (couche cachée)
COPY model_api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Code source
COPY features.py .
COPY model_api/ ./model_api/

# Modèles par défaut — copiés dans V2 (/models/) au 1er démarrage
# via l'entrypoint.sh (si V2 est vide)
COPY ser_conv1d_model.keras      /models_default/
COPY ser_conv1d_lstm_model.keras /models_default/
COPY scaler.pkl                  /models_default/
COPY label_encoder.pkl           /models_default/

# Point de montage du volume V2
RUN mkdir -p /models

# Script d'initialisation du volume
COPY model_api/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=15s --retries=5 --start-period=90s \
  CMD curl -f http://localhost:8001/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "model_api.app:app", "--host", "0.0.0.0", "--port", "8001"]

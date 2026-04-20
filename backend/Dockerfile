FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt .

RUN apt-get update \
    && apt-get install -y curl \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*

COPY backend/ ./backend/
COPY Models/ ./models/
COPY scaler.pkl ./models
COPY label_encoder.pkl ./models

EXPOSE 8000

ENTRYPOINT ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]

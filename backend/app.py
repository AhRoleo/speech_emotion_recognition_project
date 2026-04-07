"""
Backend FastAPI — Orchestrateur C2

Rôle :
  1. Reçoit l'audio brut depuis le frontend C1
  2. Extrait les features audio (185 dimensions via features.py)
  3. Appelle C3 (model-api) pour la prédiction
  4. Sauvegarde l'audio dans le volume V1 avec l'émotion prédite
     comme pseudo-label (dossier = nom de l'émotion)
  5. Renvoie la prédiction au frontend
"""

import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Permet d'importer features.py situé à la racine du projet
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import extract_features  # noqa: E402

from .schemas import (  # noqa: E402
    EmotionProbability,
    HealthResponse,
    ModelInfo,
    ModelsListResponse,
    PredictionResponse,
)

# ============================================================
# CONFIGURATION VIA VARIABLES D'ENVIRONNEMENT
# ============================================================

# URL de C3 (model-api) — défaut pour lancement local hors Docker
MODEL_API_URL = os.environ.get("MODEL_API_URL", "http://localhost:8001")

# Chemin du volume V1 — dossier racine pour les audios pseudo-labellisés
DATASET_DIR = os.environ.get(
    "SER_DATASET_DIR",
    str(Path(__file__).resolve().parent.parent / "data" / "dataset"),
)

# ============================================================
# INITIALISATION FASTAPI
# ============================================================

app = FastAPI(
    title="SER Backend — Orchestrateur",
    description=(
        "API orchestratrice : extraction features → prédiction via C3 → "
        "sauvegarde audio dans V1 (pseudo-labeling)."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================


async def save_audio_to_dataset(audio_bytes: bytes, emotion: str, original_filename: str) -> None:
    """
    Sauvegarde le fichier audio dans le volume V1, classé par émotion.

    Structure créée dans V1 :
        /data/dataset/
            happy/
                20240101_120000_a3f9b2.wav
            angry/
                ...

    Le nom du dossier = émotion prédite = pseudo-label utilisé par C4.
    """
    try:
        # Déduire l'extension depuis le nom d'origine (wav ou mp3)
        ext = Path(original_filename).suffix or ".wav"

        # Dossier cible : <DATASET_DIR>/<émotion>/
        emotion_dir = Path(DATASET_DIR) / emotion
        emotion_dir.mkdir(parents=True, exist_ok=True)

        # Nom de fichier horodaté + UUID court pour éviter les collisions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:6]
        dest_path = emotion_dir / f"{timestamp}_{short_id}{ext}"

        dest_path.write_bytes(audio_bytes)

    except Exception as exc:
        # On ne bloque pas la réponse si la sauvegarde échoue
        print(f"[WARN] Sauvegarde audio dans V1 impossible : {exc}")


# ============================================================
# ENDPOINT : HEALTHCHECK
# ============================================================


@app.get("/health", response_model=HealthResponse)
async def health():
    """Vérifie que le backend fonctionne et interroge C3 pour l'état des modèles."""
    model_data: dict = {}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{MODEL_API_URL}/health", timeout=5)
            if resp.status_code == 200:
                model_data = resp.json()
    except Exception:
        pass  # C3 peut être temporairement indisponible

    return HealthResponse(
        status="ok",
        models_loaded=model_data.get("models_loaded", 0),
        models_available=model_data.get("models_available", 0),
    )


# ============================================================
# ENDPOINT : LISTE DES MODÈLES (proxy vers C3)
# ============================================================


@app.get("/models", response_model=ModelsListResponse)
async def list_models():
    """Retourne la liste des modèles disponibles dans V2 (via C3)."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{MODEL_API_URL}/models", timeout=10)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Erreur C3 : {exc.response.text}",
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model API (C3) inaccessible : {exc}")

    return ModelsListResponse(
        models=[ModelInfo(**m) for m in data.get("models", [])]
    )


# ============================================================
# ENDPOINT : PRÉDICTION D'ÉMOTION
# ============================================================


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    audio: UploadFile = File(..., description="Fichier audio WAV ou MP3"),
    model_name: str = Form(..., description="Nom du fichier .keras à utiliser"),
):
    """
    Pipeline complet de prédiction :
    1. Lecture du fichier audio
    2. Extraction des features (185 dimensions)
    3. Envoi des features à C3 (model-api) pour prédiction
    4. Sauvegarde de l'audio dans V1 avec l'émotion comme pseudo-label
    5. Retour de la prédiction au client
    """

    # 1. Lire le contenu du fichier audio
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Fichier audio vide.")

    # 2. Extraction des features
    features = extract_features(audio_bytes)
    if all(f == 0.0 for f in features):
        raise HTTPException(
            status_code=422,
            detail="Impossible d'extraire les features de ce fichier audio.",
        )

    # 3. Appel à C3 (model-api) avec les features en JSON
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{MODEL_API_URL}/predict",
                json={"features": features, "model_name": model_name},
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Erreur C3 : {exc.response.text}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model API (C3) inaccessible : {exc}",
        )

    # 4. Sauvegarde de l'audio dans V1 avec pseudo-label
    predicted_emotion = result.get("main_emotion", "unknown")
    await save_audio_to_dataset(
        audio_bytes,
        predicted_emotion,
        audio.filename or "audio.wav",
    )

    # 5. Retourner la réponse au frontend
    return PredictionResponse(**result)

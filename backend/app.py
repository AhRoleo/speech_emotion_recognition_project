"""
Application FastAPI pour le service de prédiction SER.

Endpoints :
    GET  /health   — Healthcheck
    GET  /models   — Liste des modèles disponibles
    POST /predict  — Prédiction d'émotion à partir d'un fichier audio
"""

import os
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Ajouter le répertoire parent au path pour importer features.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import EMOTION_DISPLAY, extract_features  # noqa: E402

from .model_manager import ModelManager  # noqa: E402
from .schemas import (  # noqa: E402
    EmotionProbability,
    HealthResponse,
    ModelInfo,
    ModelsListResponse,
    PredictionResponse,
)

# --- Configuration ---
PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
MODELS_DIR = os.environ.get("SER_MODELS_DIR", PROJECT_DIR)
SCALER_PATH = os.environ.get("SER_SCALER_PATH", os.path.join(PROJECT_DIR, "scaler.pkl"))
LABEL_ENCODER_PATH = os.environ.get(
    "SER_LABEL_ENCODER_PATH", os.path.join(PROJECT_DIR, "label_encoder.pkl")
)

# --- Initialisation ---
app = FastAPI(
    title="SER API — Speech Emotion Recognition",
    description="API de prédiction d'émotions vocales avec support multi-modèles.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ModelManager(
    models_dir=MODELS_DIR,
    scaler_path=SCALER_PATH,
    label_encoder_path=LABEL_ENCODER_PATH,
)


# --- Endpoints ---


@app.get("/health", response_model=HealthResponse)
async def health():
    """Vérifie que l'API est opérationnelle."""
    return HealthResponse(
        status="ok",
        models_loaded=manager.num_loaded,
        models_available=manager.num_available,
    )


@app.get("/models", response_model=ModelsListResponse)
async def list_models():
    """Liste tous les modèles .keras disponibles."""
    models_data = manager.list_models()
    models = [
        ModelInfo(
            name=m["name"],
            file_name=m["file_name"],
            num_classes=m["num_classes"],
        )
        for m in models_data
    ]
    return ModelsListResponse(models=models)


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    audio: UploadFile = File(..., description="Fichier audio WAV ou MP3"),
    model_name: str = Form(
        ..., description="Nom du fichier .keras à utiliser (ex: ser_conv1d_model.keras)"
    ),
):
    """
    Prédit l'émotion d'un fichier audio.

    1. Lit le fichier audio uploadé
    2. Extrait les features (185 dimensions)
    3. Normalise avec le StandardScaler
    4. Prédit avec le modèle sélectionné
    5. Retourne les probabilités par émotion
    """
    # Lire le fichier audio
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Fichier audio vide.")

    # Extraire les features
    features = extract_features(audio_bytes)
    if all(f == 0.0 for f in features):
        raise HTTPException(
            status_code=422,
            detail="Impossible d'extraire les features de ce fichier audio.",
        )

    # Prédire
    try:
        result = manager.predict(features, model_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {e}")

    # Construire la réponse
    predicted_class = result["predicted_class"]
    probabilities = result["probabilities"]

    # Tri décroissant par probabilité
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    emotion_probs = []
    for emotion, prob in sorted_probs:
        display = EMOTION_DISPLAY.get(emotion, (emotion.title(), "❓"))
        emotion_probs.append(
            EmotionProbability(
                emotion=emotion,
                label_fr=display[0],
                emoji=display[1],
                probability=round(prob, 4),
            )
        )

    main_display = EMOTION_DISPLAY.get(predicted_class, (predicted_class.title(), "❓"))

    return PredictionResponse(
        model_name=model_name,
        main_emotion=predicted_class,
        main_label_fr=main_display[0],
        main_emoji=main_display[1],
        confidence=round(probabilities[predicted_class], 4),
        probabilities=emotion_probs,
    )

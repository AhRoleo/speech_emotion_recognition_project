"""
Model API — C3 (FastAPI de prédiction pure)

Rôle :
  - Reçoit un vecteur de features (185 floats) + nom de modèle depuis C2
  - Charge le modèle .keras depuis le volume V2 (/models/)
  - Retourne la prédiction d'émotion avec les probabilités complètes

C3 est le seul container à utiliser TensorFlow et à lire V2.
"""

import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# features.py est copié à la racine du container (/app/features.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import EMOTION_DISPLAY  # noqa: E402

from .model_manager import ModelManager  # noqa: E402
from .schemas import (  # noqa: E402
    EmotionProbability,
    HealthResponse,
    ModelInfo,
    ModelsListResponse,
    PredictRequest,
    PredictionResponse,
)

# ============================================================
# CONFIGURATION
# ============================================================

MODELS_DIR = os.environ.get("SER_MODELS_DIR", "/models")
SCALER_PATH = os.environ.get("SER_SCALER_PATH", "/models/scaler.pkl")
LABEL_ENCODER_PATH = os.environ.get("SER_LABEL_ENCODER_PATH", "/models/label_encoder.pkl")

# ============================================================
# INITIALISATION
# ============================================================

app = FastAPI(
    title="SER Model API — C3",
    description="API de prédiction pure : reçoit des features, retourne une émotion.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instanciation unique du gestionnaire de modèles (lazy loading + cache)
manager = ModelManager(
    models_dir=MODELS_DIR,
    scaler_path=SCALER_PATH,
    label_encoder_path=LABEL_ENCODER_PATH,
)

# ============================================================
# ENDPOINT : HEALTHCHECK
# ============================================================


@app.get("/health", response_model=HealthResponse)
async def health():
    """Indique si C3 est opérationnel et combien de modèles sont disponibles."""
    return HealthResponse(
        status="ok",
        models_loaded=manager.num_loaded,
        models_available=manager.num_available,
    )


# ============================================================
# ENDPOINT : LISTE DES MODÈLES
# ============================================================


@app.get("/models", response_model=ModelsListResponse)
async def list_models():
    """Retourne la liste des modèles .keras disponibles dans V2."""
    models_data = manager.list_models()
    models = [
        ModelInfo(name=m["name"], file_name=m["file_name"], num_classes=m["num_classes"])
        for m in models_data
    ]
    return ModelsListResponse(models=models)


# ============================================================
# ENDPOINT : PRÉDICTION
# ============================================================


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictRequest):
    """
    Prédiction d'émotion à partir d'un vecteur de features.

    Reçoit de C2 :
      - features : liste de 185 floats
      - model_name : nom du fichier .keras dans V2

    Retourne :
      - l'émotion principale prédite (+ label FR + emoji)
      - la confiance
      - les probabilités de toutes les émotions
    """
    try:
        result = manager.predict(request.features, request.model_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {exc}")

    predicted_class = result["predicted_class"]
    probabilities = result["probabilities"]

    # Trier les émotions par probabilité décroissante
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
        model_name=request.model_name,
        main_emotion=predicted_class,
        main_label_fr=main_display[0],
        main_emoji=main_display[1],
        confidence=round(probabilities[predicted_class], 4),
        probabilities=emotion_probs,
    )

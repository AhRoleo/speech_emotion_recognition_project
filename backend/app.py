
import os
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Permet d'importer features.py situé dans le dossier parent
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

# ============================================================
# 2) CHEMINS ET VARIABLES D’ENVIRONNEMENT
# ============================================================

PROJECT_DIR = str(Path(__file__).resolve().parent.parent)

# Dossier contenant les modèles .keras
MODELS_DIR = os.environ.get("SER_MODELS_DIR", PROJECT_DIR)

# Chemins du scaler et du label encoder
SCALER_PATH = os.environ.get("SER_SCALER_PATH", os.path.join(PROJECT_DIR, "scaler.pkl"))
LABEL_ENCODER_PATH = os.environ.get(
    "SER_LABEL_ENCODER_PATH", os.path.join(PROJECT_DIR, "label_encoder.pkl")
)

# ============================================================
# 3) INITIALISATION DE L’APPLICATION FASTAPI
# ============================================================

app = FastAPI(
    title="SER API — Speech Emotion Recognition",
    description="API de prédiction d'émotions vocales avec support multi-modèles.",
    version="1.0.0",
)

# Autoriser toutes les origines (utile pour le front-end)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gestionnaire de modèles (chargement, prédiction, etc.)
manager = ModelManager(
    models_dir=MODELS_DIR,
    scaler_path=SCALER_PATH,
    label_encoder_path=LABEL_ENCODER_PATH,
)

# ============================================================
# 4) ENDPOINT : HEALTHCHECK
# ============================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """Vérifie que l'API fonctionne et que les modèles sont chargés."""
    return HealthResponse(
        status="ok",
        models_loaded=manager.num_loaded,
        models_available=manager.num_available,
    )

# ============================================================
# 5) ENDPOINT : LISTE DES MODÈLES DISPONIBLES
# ============================================================

@app.get("/models", response_model=ModelsListResponse)
async def list_models():
    """Retourne la liste des modèles .keras présents dans le dossier."""
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

# ============================================================
# 6) ENDPOINT : PRÉDICTION D’ÉMOTION
# ============================================================

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    audio: UploadFile = File(..., description="Fichier audio WAV ou MP3"),
    model_name: str = Form(
        ..., description="Nom du fichier .keras à utiliser (ex: ser_conv1d_model.keras)"
    ),
):
    """
    Pipeline de prédiction :
    1. Lecture du fichier audio
    2. Extraction des features (185 dimensions)
    3. Normalisation via StandardScaler
    4. Prédiction via le modèle choisi
    5. Construction de la réponse JSON
    """

    # --- 1) Lire le fichier audio ---
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Fichier audio vide.")

    # --- 2) Extraire les features ---
    features = extract_features(audio_bytes)
    if all(f == 0.0 for f in features):
        raise HTTPException(
            status_code=422,
            detail="Impossible d'extraire les features de ce fichier audio.",
        )

    # --- 3) Prédire avec le modèle ---
    try:
        result = manager.predict(features, model_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {e}")

    predicted_class = result["predicted_class"]
    probabilities = result["probabilities"]

    # --- 4) Trier les probabilités ---
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    # Construire la liste des émotions avec labels FR + emoji
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

    # --- 5) Retourner la réponse ---
    return PredictionResponse(
        model_name=model_name,
        main_emotion=predicted_class,
        main_label_fr=main_display[0],
        main_emoji=main_display[1],
        confidence=round(probabilities[predicted_class], 4),
        probabilities=emotion_probs,
    )

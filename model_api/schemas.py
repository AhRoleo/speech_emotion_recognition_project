"""
Schémas Pydantic pour l'API Modèle (C3).

Reprend les schémas du backend + PredictRequest
(les features arrivent en JSON depuis C2, pas en multipart).
"""

from pydantic import BaseModel


# ── Requête entrante depuis C2 ────────────────────────────────
class PredictRequest(BaseModel):
    """Payload reçu de C2 : vecteur de features + nom du modèle."""
    features: list[float]
    model_name: str


# ── Probabilité d'une émotion ─────────────────────────────────
class EmotionProbability(BaseModel):
    emotion: str
    label_fr: str
    emoji: str
    probability: float


# ── Réponse complète de prédiction ────────────────────────────
class PredictionResponse(BaseModel):
    model_name: str
    main_emotion: str
    main_label_fr: str
    main_emoji: str
    confidence: float
    probabilities: list[EmotionProbability]


# ── Informations sur un modèle disponible ────────────────────
class ModelInfo(BaseModel):
    name: str
    file_name: str
    num_classes: int


# ── Liste des modèles ─────────────────────────────────────────
class ModelsListResponse(BaseModel):
    models: list[ModelInfo]


# ── Réponse healthcheck ───────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    models_available: int

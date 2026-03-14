"""
Schémas Pydantic pour l'API SER.
"""

from pydantic import BaseModel


class EmotionProbability(BaseModel):
    """Probabilité d'une émotion."""
    emotion: str
    label_fr: str
    emoji: str
    probability: float


class PredictionResponse(BaseModel):
    """Réponse d'une prédiction d'émotion."""
    model_name: str
    main_emotion: str
    main_label_fr: str
    main_emoji: str
    confidence: float
    probabilities: list[EmotionProbability]


class ModelInfo(BaseModel):
    """Informations sur un modèle disponible."""
    name: str
    file_name: str
    num_classes: int


class ModelsListResponse(BaseModel):
    """Liste des modèles disponibles."""
    models: list[ModelInfo]


class HealthResponse(BaseModel):
    """Réponse du healthcheck."""
    status: str
    models_loaded: int
    models_available: int

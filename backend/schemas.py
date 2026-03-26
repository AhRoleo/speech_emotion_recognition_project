"""
Schémas Pydantic pour l'API SER.

Ces schémas définissent :
- la structure des données envoyées par l'API,
- la validation automatique des types,
- la forme exacte des réponses JSON.

Chaque classe correspond à un "modèle" de réponse utilisé par FastAPI.
"""

# ============================================================
# 1) IMPORTS
# ============================================================

from pydantic import BaseModel


# ============================================================
# 2) SCHÉMA : Probabilité d'une émotion
# ============================================================

class EmotionProbability(BaseModel):
    """
    Représente la probabilité d'une émotion prédite.

    Ce schéma est utilisé dans la réponse finale pour afficher :
    - le nom interne de l'émotion (ex: "angry")
    - le label français (ex: "Colère")
    - un emoji associé (ex: "😡")
    - la probabilité (ex: 0.72)
    """
    emotion: str
    label_fr: str
    emoji: str
    probability: float


# ============================================================
# 3) SCHÉMA : Réponse complète d'une prédiction
# ============================================================

class PredictionResponse(BaseModel):
    """
    Structure de la réponse renvoyée par l'endpoint /predict.

    Contient :
    - le modèle utilisé
    - l'émotion principale prédite
    - son label français + emoji
    - la confiance (probabilité)
    - la liste complète des probabilités pour chaque émotion
    """
    model_name: str
    main_emotion: str
    main_label_fr: str
    main_emoji: str
    confidence: float
    probabilities: list[EmotionProbability]


# ============================================================
# 4) SCHÉMA : Informations sur un modèle disponible
# ============================================================

class ModelInfo(BaseModel):
    """
    Informations sur un modèle .keras détecté dans le dossier.

    Utilisé par l'endpoint /models pour afficher :
    - un nom lisible
    - le nom du fichier .keras
    - le nombre de classes que le modèle peut prédire
    """
    name: str
    file_name: str
    num_classes: int


# ============================================================
# 5) SCHÉMA : Liste des modèles disponibles
# ============================================================

class ModelsListResponse(BaseModel):
    """
    Réponse renvoyée par l'endpoint /models.

    Contient simplement une liste d'objets ModelInfo.
    """
    models: list[ModelInfo]


# ============================================================
# 6) SCHÉMA : Réponse du healthcheck
# ============================================================

class HealthResponse(BaseModel):
    """
    Réponse renvoyée par l'endpoint /health.

    Permet de vérifier :
    - si l'API fonctionne ("ok")
    - combien de modèles sont chargés en mémoire
    - combien de modèles .keras sont disponibles sur le disque
    """
    status: str
    models_loaded: int
    models_available: int

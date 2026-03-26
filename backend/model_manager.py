"""
Gestionnaire de modèles pour l'API SER.

Ce module :
- détecte les modèles .keras disponibles,
- charge les modèles uniquement quand ils sont utilisés (lazy loading),
- applique le scaler et le label encoder,
- effectue la prédiction complète (normalisation + reshape + modèle).
"""

# ============================================================
# 1) IMPORTS ET CONFIGURATION
# ============================================================

import os
import glob
from pathlib import Path

import joblib
import numpy as np

# Réduire les logs TensorFlow (évite les messages inutiles)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf  # noqa: E402


# ============================================================
# 2) CLASSE PRINCIPALE : ModelManager
# ============================================================

class ModelManager:
    """
    Gère le chargement et l'utilisation des modèles SER.

    Fonctionnalités :
    - Scanne un dossier pour trouver les fichiers .keras
    - Charge les modèles uniquement quand ils sont demandés (lazy loading)
    - Stocke les modèles en cache pour éviter de les recharger
    - Applique le scaler et le label encoder
    - Effectue la prédiction complète
    """

    # ------------------------------------------------------------
    # 2.1) Constructeur : initialisation du gestionnaire
    # ------------------------------------------------------------
    def __init__(self, models_dir: str, scaler_path: str, label_encoder_path: str):
        """
        Args:
            models_dir: Dossier contenant les modèles .keras
            scaler_path: Chemin vers scaler.pkl (StandardScaler)
            label_encoder_path: Chemin vers label_encoder.pkl (LabelEncoder)
        """
        self.models_dir = Path(models_dir)

        # Cache des modèles déjà chargés
        self._models: dict[str, tf.keras.Model] = {}

        # Charger scaler et label encoder s'ils existent
        self.scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        self.label_encoder = (
            joblib.load(label_encoder_path)
            if os.path.exists(label_encoder_path)
            else None
        )

    # ------------------------------------------------------------
    # 2.2) Lister les modèles disponibles
    # ------------------------------------------------------------
    def list_models(self) -> list[dict]:
        """
        Retourne la liste des modèles .keras présents dans le dossier.

        Returns:
            Liste de dicts contenant :
            - name : nom lisible du modèle
            - file_name : nom du fichier .keras
            - num_classes : nombre de classes prédites par le modèle
        """
        model_files = sorted(glob.glob(str(self.models_dir / "*.keras")))
        models = []

        for f in model_files:
            file_name = os.path.basename(f)

            # Nom lisible : "ser_conv1d_model" → "Ser Conv1D Model"
            name = file_name.replace(".keras", "").replace("_", " ").title()

            # Charger le modèle pour connaître le nombre de classes
            model = self._load_model(file_name)
            num_classes = model.output_shape[-1] if model else 0

            models.append({
                "name": name,
                "file_name": file_name,
                "num_classes": num_classes,
            })

        return models

    # ------------------------------------------------------------
    # 2.3) Chargement d’un modèle (avec cache)
    # ------------------------------------------------------------
    def _load_model(self, file_name: str) -> tf.keras.Model | None:
        """
        Charge un modèle depuis le cache ou depuis le disque.

        - Si le modèle est déjà chargé → on le réutilise
        - Sinon → on le charge depuis le fichier .keras
        """
        # Si déjà en cache → on le renvoie
        if file_name in self._models:
            return self._models[file_name]

        model_path = self.models_dir / file_name
        if not model_path.exists():
            return None

        try:
            # Chargement du modèle Keras
            model = tf.keras.models.load_model(str(model_path))

            # Stockage dans le cache
            self._models[file_name] = model
            return model

        except Exception as e:
            print(f"Erreur chargement modèle {file_name}: {e}")
            return None

    # ------------------------------------------------------------
    # 2.4) Prédiction complète
    # ------------------------------------------------------------
    def predict(self, features: list[float], model_file_name: str) -> dict:
        """
        Effectue une prédiction à partir des features extraites.

        Pipeline :
        1. Charger le modèle demandé
        2. Convertir les features en numpy
        3. Normaliser avec le scaler
        4. Reshape pour Conv1D
        5. Prédire avec le modèle
        6. Associer les probabilités aux émotions
        """
        # 1) Charger le modèle
        model = self._load_model(model_file_name)
        if model is None:
            raise ValueError(f"Modèle '{model_file_name}' introuvable.")

        # 2) Convertir en array numpy
        x = np.array(features, dtype=np.float32).reshape(1, -1)

        # 3) Normalisation
        if self.scaler is not None:
            x = self.scaler.transform(x)

        # 4) Reshape pour Conv1D : (1, 185) → (1, 185, 1)
        x = x.reshape(1, -1, 1)

        # 5) Prédiction
        probs = model.predict(x, verbose=0)[0]

        # 6) Associer les probabilités aux classes
        if self.label_encoder is not None:
            classes = list(self.label_encoder.classes_)
        else:
            # Fallback si pas de label encoder
            from features import EMOTION_LABELS
            classes = EMOTION_LABELS[:probs.shape[0]]

        probabilities = {
            classes[i]: float(probs[i]) for i in range(len(classes))
        }

        predicted_class = classes[int(np.argmax(probs))]

        return {
            "probabilities": probabilities,
            "predicted_class": predicted_class,
        }

    # ------------------------------------------------------------
    # 2.5) Propriétés utilitaires
    # ------------------------------------------------------------
    @property
    def num_loaded(self) -> int:
        """Nombre de modèles actuellement chargés en mémoire."""
        return len(self._models)

    @property
    def num_available(self) -> int:
        """Nombre total de modèles .keras présents sur le disque."""
        return len(glob.glob(str(self.models_dir / "*.keras")))

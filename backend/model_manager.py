"""
Gestionnaire de modèles pour l'API SER.

Charge les modèles Keras à la demande (lazy loading avec cache),
gère le StandardScaler et le LabelEncoder pour l'inférence.
"""

import os
import glob
from pathlib import Path

import joblib
import numpy as np

# Réduire la verbosité de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf  # noqa: E402


class ModelManager:
    """
    Gère le chargement et l'utilisation des modèles SER.

    Scanne un répertoire pour trouver les fichiers .keras,
    les charge à la demande, et fournit l'inférence complète
    (normalisation + reshape + prédiction).
    """

    def __init__(self, models_dir: str, scaler_path: str, label_encoder_path: str):
        """
        Args:
            models_dir: Répertoire contenant les fichiers .keras
            scaler_path: Chemin vers le fichier scaler.pkl
            label_encoder_path: Chemin vers le fichier label_encoder.pkl
        """
        self.models_dir = Path(models_dir)
        self._models: dict[str, tf.keras.Model] = {}

        # Charger le scaler et le label encoder
        self.scaler = None
        self.label_encoder = None

        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        if os.path.exists(label_encoder_path):
            self.label_encoder = joblib.load(label_encoder_path)

    def list_models(self) -> list[dict]:
        """
        Liste tous les modèles .keras disponibles dans le répertoire.

        Returns:
            Liste de dicts avec 'name', 'file_name', et 'num_classes'.
        """
        model_files = sorted(glob.glob(str(self.models_dir / "*.keras")))
        models = []
        for f in model_files:
            file_name = os.path.basename(f)
            name = file_name.replace(".keras", "").replace("_", " ").title()
            # Charger le modèle si nécessaire pour obtenir le nb de classes
            model = self._load_model(file_name)
            num_classes = model.output_shape[-1] if model else 0
            models.append({
                "name": name,
                "file_name": file_name,
                "num_classes": num_classes,
            })
        return models

    def _load_model(self, file_name: str) -> tf.keras.Model | None:
        """Charge un modèle depuis le cache ou le disque."""
        if file_name in self._models:
            return self._models[file_name]

        model_path = self.models_dir / file_name
        if not model_path.exists():
            return None

        try:
            model = tf.keras.models.load_model(str(model_path))
            self._models[file_name] = model
            return model
        except Exception as e:
            print(f"Erreur chargement modèle {file_name}: {e}")
            return None

    def predict(self, features: list[float], model_file_name: str) -> dict:
        """
        Effectue une prédiction à partir des features extraites.

        Args:
            features: Vecteur de 185 features (sortie de extract_features).
            model_file_name: Nom du fichier .keras à utiliser.

        Returns:
            Dict avec 'probabilities' (dict emotion→proba) et 'predicted_class' (str).
        """
        model = self._load_model(model_file_name)
        if model is None:
            raise ValueError(f"Modèle '{model_file_name}' introuvable.")

        # Convertir en numpy array
        x = np.array(features, dtype=np.float32).reshape(1, -1)

        # Normaliser si le scaler est disponible
        if self.scaler is not None:
            x = self.scaler.transform(x)

        # Reshape pour Conv1D : (1, 185) → (1, 185, 1)
        x = x.reshape(1, -1, 1)

        # Prédiction
        probs = model.predict(x, verbose=0)[0]

        # Mapper les probabilités aux émotions
        if self.label_encoder is not None:
            classes = list(self.label_encoder.classes_)
        else:
            # Fallback : ordre alphabétique standard
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

    @property
    def num_loaded(self) -> int:
        """Nombre de modèles actuellement en cache."""
        return len(self._models)

    @property
    def num_available(self) -> int:
        """Nombre de modèles .keras disponibles sur le disque."""
        return len(glob.glob(str(self.models_dir / "*.keras")))

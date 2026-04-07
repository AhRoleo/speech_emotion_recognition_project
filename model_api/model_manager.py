"""
Gestionnaire de modèles pour C3 (model-api).

Fonctionne de manière identique au backend/model_manager.py :
  - Détecte les modèles .keras dans le volume V2 (/models/)
  - Lazy loading + cache en mémoire
  - Applique scaler et label encoder avant prédiction
"""

import glob
import os
from pathlib import Path

import joblib
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf  # noqa: E402


class ModelManager:
    """
    Gère le chargement et l'utilisation des modèles SER depuis V2.

    Les modèles sont chargés en lazy loading et mis en cache pour
    éviter de recharger TensorFlow à chaque requête.
    """

    def __init__(self, models_dir: str, scaler_path: str, label_encoder_path: str):
        self.models_dir = Path(models_dir)
        self._models: dict[str, tf.keras.Model] = {}

        self.scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        self.label_encoder = (
            joblib.load(label_encoder_path) if os.path.exists(label_encoder_path) else None
        )

    def list_models(self) -> list[dict]:
        """Retourne la liste des modèles .keras disponibles dans V2."""
        model_files = sorted(glob.glob(str(self.models_dir / "*.keras")))
        models = []

        for f in model_files:
            file_name = os.path.basename(f)
            name = file_name.replace(".keras", "").replace("_", " ").title()
            model = self._load_model(file_name)
            num_classes = model.output_shape[-1] if model else 0
            models.append({"name": name, "file_name": file_name, "num_classes": num_classes})

        return models

    def _load_model(self, file_name: str) -> tf.keras.Model | None:
        """Charge un modèle depuis le cache ou depuis le disque (V2)."""
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
            print(f"[ERROR] Chargement modèle {file_name} : {e}")
            return None

    def predict(self, features: list[float], model_file_name: str) -> dict:
        """
        Pipeline de prédiction complet.
          1. Charger le modèle depuis le cache V2
          2. Convertir les features en numpy
          3. Normaliser avec le scaler
          4. Reshape pour Conv1D : (1, 185, 1)
          5. Prédire + associer les probabilités aux labels
        """
        model = self._load_model(model_file_name)
        if model is None:
            raise ValueError(f"Modèle '{model_file_name}' introuvable dans V2.")

        x = np.array(features, dtype=np.float32).reshape(1, -1)

        if self.scaler is not None:
            x = self.scaler.transform(x)

        x = x.reshape(1, -1, 1)  # (1, 185, 1)

        probs = model.predict(x, verbose=0)[0]

        if self.label_encoder is not None:
            classes = list(self.label_encoder.classes_)
        else:
            from features import EMOTION_LABELS
            classes = EMOTION_LABELS[: probs.shape[0]]

        probabilities = {classes[i]: float(probs[i]) for i in range(len(classes))}
        predicted_class = classes[int(np.argmax(probs))]

        return {"probabilities": probabilities, "predicted_class": predicted_class}

    @property
    def num_loaded(self) -> int:
        return len(self._models)

    @property
    def num_available(self) -> int:
        return len(glob.glob(str(self.models_dir / "*.keras")))

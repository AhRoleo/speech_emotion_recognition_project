"""
Pipeline d'entraînement autonome — C4 (SER Project)

Cycle : toutes les TRAINING_INTERVAL_HOURS heures (défaut 24h)

Étapes d'un cycle :
  1.  Scanner V1 (/data/dataset/<émotion>/*.wav) pour collecter les audios
      pseudolabellisés par C2
  2.  Extraire les features (via features.py) pour chaque fichier
  3.  Encoder les labels et normaliser les features
  4.  Entraîner un modèle Conv1D (architecture identique au projet)
  5.  Sauvegarder le modèle, le scaler et le label encoder dans V2 (/models/)
      → disponibles immédiatement pour C3 (model-api)

Notes sur le pseudo-labeling :
  - C2 sauvegarde chaque audio dans un dossier portant l'émotion prédite
  - Ce dossier sert de label d'entraînement
  - Des données de meilleure qualité (RAVDESS, CREMA-D…) peuvent être
    pré-chargées dans V1 pour améliorer la base de départ
"""

import glob
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# TensorFlow / Keras
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

# features.py partagé — copié à la racine du container (/app/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import extract_features, EMOTION_LABELS  # noqa: E402

# ============================================================
# CONFIGURATION VIA VARIABLES D'ENVIRONNEMENT
# ============================================================

DATASET_DIR = os.environ.get("SER_DATASET_DIR", "/data/dataset")
MODELS_DIR = os.environ.get("SER_MODELS_DIR", "/models")
INTERVAL_HOURS = int(os.environ.get("TRAINING_INTERVAL_HOURS", "24"))
MIN_SAMPLES = int(os.environ.get("MIN_TRAINING_SAMPLES", "50"))

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [C4-PIPELINE] %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================
# MODÈLE : Architecture Conv1D
# ============================================================


def build_conv1d_model(input_dim: int, num_classes: int) -> keras.Model:
    """
    Construit un modèle Conv1D pour la classification d'émotions.

    Architecture :
      Conv1D(64) → BN → MaxPool → Dropout
      Conv1D(128) → BN → MaxPool → Dropout
      Conv1D(64) → BN → GlobalAvgPool → Dropout
      Dense(128) → Dropout → Dense(num_classes, softmax)
    """
    inputs = keras.Input(shape=(input_dim, 1), name="features_input")

    x = layers.Conv1D(64, 3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.30)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="emotion_output")(x)

    model = keras.Model(inputs, outputs, name="ser_conv1d_pipeline")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ============================================================
# CHARGEMENT DU DATASET
# ============================================================


def load_dataset(dataset_dir: str) -> tuple[np.ndarray, list[str]]:
    """
    Parcourt V1 et extrait les features de tous les audios trouvés.

    Structure attendue dans V1 :
        /data/dataset/
            angry/   ← pseudo-label = nom du dossier
                20240101_120000_abc123.wav
            happy/
                ...

    Returns:
        X : tableau numpy (n_samples, 185)
        y : liste des labels string (ex: ["angry", "happy", ...])
    """
    # Recherche récursive de tous les WAVs et MP3s
    audio_files = (
        glob.glob(os.path.join(dataset_dir, "**", "*.wav"), recursive=True)
        + glob.glob(os.path.join(dataset_dir, "**", "*.mp3"), recursive=True)
    )

    log.info(f"Fichiers audio trouvés dans V1 : {len(audio_files)}")

    X, y = [], []
    errors = 0

    for filepath in audio_files:
        # Pseudo-label = nom du dossier parent immédiat
        emotion = Path(filepath).parent.name

        if emotion not in EMOTION_LABELS:
            log.debug(f"Émotion inconnue '{emotion}' pour {filepath}, ignoré.")
            continue

        try:
            with open(filepath, "rb") as f:
                audio_bytes = f.read()

            features = extract_features(audio_bytes)

            # Rejeter les vecteurs nuls (extraction échouée)
            if all(v == 0.0 for v in features):
                log.debug(f"Features nulles pour {filepath}, ignoré.")
                continue

            X.append(features)
            y.append(emotion)

        except Exception as exc:
            log.warning(f"Erreur lecture {filepath} : {exc}")
            errors += 1

    log.info(f"Features extraites : {len(X)} | Erreurs : {errors}")

    return np.array(X, dtype=np.float32), y


# ============================================================
# CYCLE D'ENTRAÎNEMENT
# ============================================================


def run_training() -> None:
    """Exécute un cycle complet d'entraînement et sauvegarde dans V2."""
    log.info("══════════════════════════════════════════════════")
    log.info(" Démarrage d'un nouveau cycle d'entraînement")
    log.info("══════════════════════════════════════════════════")

    # ── 1. Chargement du dataset ──────────────────────────────
    X, y_raw = load_dataset(DATASET_DIR)

    if len(X) < MIN_SAMPLES:
        log.warning(
            f"Données insuffisantes ({len(X)} échantillons, minimum = {MIN_SAMPLES}). "
            "Cycle ignoré — attente du prochain cycle."
        )
        return

    # ── 2. Encodage des labels ────────────────────────────────
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)

    distribution = {cls: list(y_raw).count(cls) for cls in le.classes_}
    log.info(f"Classes : {list(le.classes_)} ({num_classes} classes)")
    log.info(f"Distribution : {distribution}")

    # ── 3. Normalisation ──────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 4. Reshape Conv1D : (n, 185) → (n, 185, 1) ───────────
    X_conv = X_scaled.reshape(len(X_scaled), -1, 1)

    # ── 5. Split train / validation ───────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X_conv, y, test_size=0.20, random_state=42, stratify=y
    )
    log.info(f"Train : {len(X_train)} | Val : {len(X_val)}")

    # ── 6. Construction du modèle ─────────────────────────────
    input_dim = X_scaled.shape[1]  # 185
    model = build_conv1d_model(input_dim, num_classes)
    log.info(f"Modèle créé : {model.name} — {model.count_params():,} paramètres")

    # ── 7. Entraînement ───────────────────────────────────────
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    best_val_acc = max(history.history["val_accuracy"])
    log.info(f"Meilleure accuracy validation : {best_val_acc:.4f} ({best_val_acc * 100:.1f}%)")

    # ── 8. Sauvegarde dans V2 ─────────────────────────────────
    models_dir = Path(MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Modèle horodaté (historique)
    versioned_path = models_dir / f"ser_conv1d_pipeline_{timestamp}.keras"
    model.save(str(versioned_path))
    log.info(f"Modèle versionné sauvegardé : {versioned_path}")

    # Modèle "latest" (écrase la version précédente — utilisé par C3)
    latest_path = models_dir / "ser_conv1d_pipeline_latest.keras"
    model.save(str(latest_path))
    log.info(f"Modèle 'latest' mis à jour : {latest_path}")

    # Scaler et LabelEncoder (pipeline-specific, ne remplace pas les .pkl d'origine)
    joblib.dump(scaler, models_dir / "scaler_pipeline.pkl")
    joblib.dump(le, models_dir / "label_encoder_pipeline.pkl")
    log.info("Scaler et LabelEncoder du pipeline sauvegardés dans V2.")

    log.info("══ Cycle terminé avec succès ══")


# ============================================================
# BOUCLE PRINCIPALE (toutes les INTERVAL_HOURS heures)
# ============================================================


def main() -> None:
    log.info(f"Pipeline SER — C4 démarré")
    log.info(f"  Dataset (V1)  : {DATASET_DIR}")
    log.info(f"  Modèles (V2)  : {MODELS_DIR}")
    log.info(f"  Intervalle    : {INTERVAL_HOURS}h")
    log.info(f"  Min. samples  : {MIN_SAMPLES}")

    while True:
        try:
            run_training()
        except Exception as exc:
            log.error(f"Erreur inattendue lors du cycle : {exc}", exc_info=True)

        next_run = INTERVAL_HOURS * 3600
        log.info(f"Prochain cycle dans {INTERVAL_HOURS}h. En veille...")
        time.sleep(next_run)


if __name__ == "__main__":
    main()

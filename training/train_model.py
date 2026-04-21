#!/usr/bin/env python
# coding: utf-8

# # PySpark SER Pipeline — Phase 2 : Conv1D Training
# 
# **Transition Spark → TensorFlow** — Entraînement d'un Conv1D
# 
# ### Étapes du pipeline :
# 1. **Imports & Configuration**
# 2. **Chargement Parquet → NumPy** (vérification des 185 dimensions)
# 3. **Préparation du Tenseur** (StandardScaler, Reshape, One-Hot)
# 4. **Construction du Conv1D** (256 → 128 → 64 filtres, BatchNorm, Dropout 0.4)
# 5. **Entraînement avec Callbacks** (ReduceLROnPlateau, EarlyStopping)
# 6. **Évaluation & Diagnostic** (Courbes Accuracy/Loss, Matrice de Confusion)
# 
# ### Pré-requis :
# - Les fichiers `train_features.parquet` et `test_features.parquet` doivent avoir été générés par la Phase 1
# - Vecteur de caractéristiques : **185 dimensions** (MFCC, ZCR, RMS, Spectral, Chroma, Mel, Entropy)

# ## Étape 1 — Imports & Configuration

import os
from time import time
import joblib
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import requests

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, MaxPooling1D,
    Flatten, Dense, Dropout, Input
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.utils import to_categorical

PARQUET_BASE = os.getenv("PROCESSED_DATA_DIR", "./Dataset/processed")
TRAIN_PARQUET = os.path.join(PARQUET_BASE, "train_features.parquet")
TEST_PARQUET  = os.path.join(PARQUET_BASE, "test_features.parquet")

OUTPUT_DIR = os.getenv("MODELS_DIR", "./Models")
MODEL_PATH  = os.path.join(OUTPUT_DIR, "ser_conv1d_model.keras")
CURVES_PATH = os.path.join(OUTPUT_DIR, "training_curves.png")
CM_PATH     = os.path.join(OUTPUT_DIR, "confusion_matrix.png")

PROCESSING_API_URL = os.getenv("PROCESSING_API_URL", "http://localhost:8020")

def processed_data_ready() -> bool:
    processed_dir = Path(PARQUET_BASE)
    if not processed_dir.exists():
        return False
    return all(filename.exists() for filename in [TRAIN_PARQUET, TEST_PARQUET])

def trigger_processing() -> None:
    response = requests.post(f"{PROCESSING_API_URL}/process", timeout=10)
    response.raise_for_status()

def wait_for_processing_ready(timeout: int = 600, interval: int = 5) -> None:
    start = time.time()

    while time.time() - start < timeout:
        try:
            response = requests.get(f"{PROCESSING_API_URL}/ready", timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass

        time.sleep(interval)
    raise TimeoutError("Processing did not become ready in time.")

def ensure_processed_data() -> None:
    if processed_data_ready():
        print("Processed data already available.")
        return

    print("Processed data not found. Triggering processing...")
    trigger_processing()
    wait_for_processing_ready()

    if not processed_data_ready():
        raise FileNotFoundError(
            f"Processing finished but required files are still missing in {PARQUET_BASE}"
        )

def run_training():
    ensure_processed_data()

    print(f"Train path: {TRAIN_PARQUET}")

    FEATURE_DIM = 185
    BATCH_SIZE  = 64
    EPOCHS      = 100
    DROPOUT     = 0.4
    RANDOM_SEED = 42

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    print(f"TensorFlow {tf.__version__}")
    print(f"   GPU disponible : {tf.config.list_physical_devices('GPU')}")

    # ## Étape 2 — Chargement Parquet → NumPy
    # 
    # Chargement des fichiers Parquet générés par la Phase 1 (PySpark).
    # Vérification systématique que **chaque vecteur** fait exactement **185 dimensions**.

    # --- Chargement ---
    df_train = pd.read_parquet(TRAIN_PARQUET)
    df_test  = pd.read_parquet(TEST_PARQUET)

    print(f" Train : {len(df_train)} lignes, colonnes = {list(df_train.columns)}")
    print(f" Test  : {len(df_test)} lignes,  colonnes = {list(df_test.columns)}")

    # --- Extraction features → NumPy ---
    X_train = np.vstack(df_train["features"].values)
    X_test  = np.vstack(df_test["features"].values)

    y_train_labels = df_train["label"].values
    y_test_labels  = df_test["label"].values

    # --- Vérification de la taille des vecteurs ---
    print(f" Vérification des dimensions :")
    print(f"   X_train shape : {X_train.shape}")
    print(f"   X_test  shape : {X_test.shape}")

    print(f"Tous les vecteurs ont exactement {FEATURE_DIM} dimensions")

    # Libérer les DataFrames Pandas
    del df_train, df_test


    # ##  Étape 3 — Préparation du Tenseur
    # 
    # | Opération | Détail |
    # |-----------|--------|
    # | **StandardScaler** | `fit()` sur le train uniquement, `transform()` sur train + test |
    # | **Reshape** | `(n, 185)` → `(n, 185, 1)` pour l'entrée Conv1D |
    # | **One-Hot Encoding** | `LabelEncoder` + `to_categorical()` → 8 classes |

    # --- StandardScaler (fit sur train, transform sur les deux) ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # --- Reshape pour Conv1D : (batch, 185) → (batch, 185, 1) ---
    X_train = X_train.reshape(-1, FEATURE_DIM, 1)
    X_test  = X_test.reshape(-1, FEATURE_DIM, 1)
    print(f" Reshape : X_train={X_train.shape}, X_test={X_test.shape}")

    # --- Encodage des labels → One-Hot ---
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_labels)
    y_test_enc  = le.transform(y_test_labels)

    NUM_CLASSES = len(le.classes_)
    y_train = to_categorical(y_train_enc, num_classes=NUM_CLASSES)
    y_test  = to_categorical(y_test_enc,  num_classes=NUM_CLASSES)

    print(f" One-Hot Encoding : {NUM_CLASSES} classes → {list(le.classes_)}")
    print(f"   y_train shape : {y_train.shape}")
    print(f"   y_test  shape : {y_test.shape}")

    # ## Étape 4 — Construction du Conv1D
    # 
    # ```
    # Input: (185, 1)
    # ├── Conv1D(256, kernel=5, relu) + BatchNorm + MaxPool1D(5)
    # ├── Conv1D(128, kernel=5, relu) + BatchNorm + MaxPool1D(5)
    # ├── Conv1D(64,  kernel=5, relu) + BatchNorm + MaxPool1D(5)
    # ├── Flatten
    # ├── Dense(256, relu) + BatchNorm + Dropout(0.4)
    # ├── Dense(128, relu) + BatchNorm + Dropout(0.4)
    # └── Dense(NUM_CLASSES, softmax)
    # ```

    model = Sequential([
        Input(shape=(FEATURE_DIM, 1)),

        # --- Bloc Conv 1 : 256 filtres ---
        Conv1D(256, kernel_size=5, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=5),

        # --- Bloc Conv 2 : 128 filtres ---
        Conv1D(128, kernel_size=5, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=5),

        # --- Bloc Conv 3 : 64 filtres ---
        Conv1D(64, kernel_size=5, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=5),

        # --- Couches denses ---
        Flatten(),

        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(DROPOUT),

        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(DROPOUT),

        Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # ## Étape 5 — Entraînement avec Callbacks
    # 
    # | Callback | Paramètres |
    # |----------|------------|
    # | **ReduceLROnPlateau** | `monitor='val_loss'`, `factor=0.5`, `patience=5`, `min_lr=1e-6` |
    # | **EarlyStopping** | `monitor='val_loss'`, `patience=10`, `restore_best_weights=True` |

    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print(f" Lancement de l'entraînement")
    print(f"   Batch size : {BATCH_SIZE}")
    print(f"   Max epochs : {EPOCHS}")
    print(f"   Train size : {X_train.shape[0]}, Val size : {X_test.shape[0]}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ##  Étape 6 — Évaluation & Diagnostic

    # --- Score sur le test set ---
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f" Test Loss     : {test_loss:.4f}")
    print(f" Test Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")

    # --- Classification Report ---
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    print(f"\n Classification Report :")
    print(classification_report(
        y_true_classes, y_pred_classes,
        target_names=le.classes_,
        digits=3,
    ))

    # ### Courbes Accuracy / Loss

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"],    label="Train Accuracy", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy",   linewidth=2)
    axes[0].set_title("Accuracy — Train vs Validation", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train Loss", linewidth=2)
    axes[1].plot(history.history["val_loss"],  label="Val Loss",   linewidth=2)
    axes[1].set_title("Loss — Train vs Validation", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CURVES_PATH, dpi=150, bbox_inches="tight")
    print(f" Courbes sauvegardées → {CURVES_PATH}")

    # ### Matrice de Confusion

    cm = confusion_matrix(y_true_classes, y_pred_classes)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
    )
    ax.set_title("Matrice de Confusion — Conv1D SER", fontsize=14, fontweight="bold")
    ax.set_xlabel("Prédiction", fontsize=12)
    ax.set_ylabel("Vérité", fontsize=12)
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150, bbox_inches="tight")
    plt.show()
    print(f" Matrice de confusion sauvegardée → {CM_PATH}")

    # ## Sauvegarde du modèle

    model.save(MODEL_PATH)
    joblib.dump(scaler, os.path.join(MODEL_PATH, "scaler.pkl"))
    joblib.dump(le, os.path.join(MODEL_PATH, "label_encoder.pkl"))

    print(f" Modèle sauvegardé → {MODEL_PATH}")

    # --- Résumé final ---
    stopped_epoch = len(history.history["loss"])
    best_val_loss = min(history.history["val_loss"])
    best_val_acc  = max(history.history["val_accuracy"])

    print(f"\n{'=' * 50}")
    print(f" Entraînement Terminé")
    print(f"{'=' * 50}")
    print(f"   Epochs effectués   : {stopped_epoch}/{EPOCHS}")
    print(f"   Meilleur val_loss  : {best_val_loss:.4f}")
    print(f"   Meilleur val_acc   : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"   Test accuracy      : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Modèle             : {MODEL_PATH}")
    print(f"   Courbes            : {CURVES_PATH}")
    print(f"   Matrice confusion  : {CM_PATH}")

"""
Script utilitaire pour sauvegarder le StandardScaler et le LabelEncoder
à partir des données d'entraînement Parquet.

Usage :
    python save_scaler.py [chemin_train_parquet]

Si aucun chemin n'est fourni, cherche dans les emplacements par défaut.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Chemins par défaut pour les données d'entraînement
DEFAULT_PATHS = [
    # Colab / Drive
    "/content/drive/MyDrive/Colab_Notebooks/train_features.parquet",
    # WSL2
    "/mnt/c/Users/joane/Desktop/ESGI4/Spark Core/voice-rec/train_features.parquet",
    # Windows
    r"C:\Users\joane\Desktop\ESGI4\Spark Core\voice-rec\train_features.parquet",
]

# Répertoire de sortie (racine du projet)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR


def find_parquet(custom_path: str | None = None) -> str:
    """Trouve le fichier Parquet d'entraînement."""
    if custom_path and os.path.exists(custom_path):
        return custom_path

    for path in DEFAULT_PATHS:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Fichier train_features.parquet introuvable.\n"
        "Utilisation : python save_scaler.py <chemin_vers_train_features.parquet>"
    )


def main():
    custom_path = sys.argv[1] if len(sys.argv) > 1 else None
    parquet_path = find_parquet(custom_path)
    print(f"📂 Chargement de : {parquet_path}")

    df = pd.read_parquet(parquet_path)
    print(f"   {len(df)} lignes, colonnes = {list(df.columns)}")

    # Extraire les features
    X = np.vstack(df["features"].values)
    print(f"   X shape : {X.shape}")

    # Fit StandardScaler
    scaler = StandardScaler()
    scaler.fit(X)

    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler sauvegardé → {scaler_path}")

    # Fit LabelEncoder
    le = LabelEncoder()
    le.fit(df["label"].values)

    le_path = os.path.join(OUTPUT_DIR, "label_encoder.pkl")
    joblib.dump(le, le_path)
    print(f"✅ LabelEncoder sauvegardé → {le_path}")
    print(f"   Classes : {list(le.classes_)}")


if __name__ == "__main__":
    main()

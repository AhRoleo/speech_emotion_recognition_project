"""
Script utilitaire pour sauvegarder le StandardScaler et le LabelEncoder
à partir des données d'entraînement Parquet.

Objectifs :
- Charger les features extraites (train_features.parquet)
- Entraîner un StandardScaler sur les features
- Entraîner un LabelEncoder sur les labels
- Sauvegarder scaler.pkl et label_encoder.pkl dans le projet

Usage :
    python save_scaler.py [chemin_train_parquet]

Si aucun chemin n'est fourni, le script cherche dans plusieurs chemins par défaut.
"""

# ============================================================
# 1) IMPORTS & CONFIGURATION
# ============================================================

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Chemins par défaut pour trouver train_features.parquet
DEFAULT_PATHS = [
    "/content/drive/MyDrive/Colab_Notebooks/train_features.parquet",  # Google Colab
    "/mnt/c/Users/joane/Desktop/ESGI4/Spark Core/voice-rec/train_features.parquet",  # WSL2
    r"C:\Users\joane\Desktop\ESGI4\Spark Core\voice-rec\train_features.parquet",  # Windows
]

# Répertoire où seront sauvegardés scaler.pkl et label_encoder.pkl
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR


# ============================================================
# 2) FONCTION : TROUVER LE FICHIER PARQUET
# ============================================================

def find_parquet(custom_path: str | None = None) -> str:
    """
    Trouve le fichier Parquet contenant les features d'entraînement.

    Priorité :
    1. Si un chemin est fourni en argument → on l'utilise
    2. Sinon → on teste les chemins par défaut
    3. Sinon → erreur explicite

    Returns:
        Chemin valide vers train_features.parquet
    """
    # 1) Chemin fourni par l'utilisateur
    if custom_path and os.path.exists(custom_path):
        return custom_path

    # 2) Chemins par défaut
    for path in DEFAULT_PATHS:
        if os.path.exists(path):
            return path

    # 3) Aucun fichier trouvé → erreur
    raise FileNotFoundError(
        "Fichier train_features.parquet introuvable.\n"
        "Utilisation : python save_scaler.py <chemin_vers_train_features.parquet>"
    )


# ============================================================
# 3) FONCTION PRINCIPALE : ENTRAÎNER & SAUVEGARDER SCALER + LABELENCODER
# ============================================================

def main():
    """
    Pipeline complet :
    1. Trouver le fichier Parquet
    2. Charger les données
    3. Extraire les features (X) et labels (y)
    4. Entraîner StandardScaler
    5. Entraîner LabelEncoder
    6. Sauvegarder scaler.pkl et label_encoder.pkl
    """
    # 1) Récupérer chemin du parquet (argument CLI ou défaut)
    custom_path = sys.argv[1] if len(sys.argv) > 1 else None
    parquet_path = find_parquet(custom_path)
    print(f"📂 Chargement de : {parquet_path}")

    # 2) Charger le fichier Parquet
    df = pd.read_parquet(parquet_path)
    print(f"   {len(df)} lignes chargées")
    print(f"   Colonnes disponibles : {list(df.columns)}")

    # 3) Extraction des features (colonne 'features')
    X = np.vstack(df["features"].values)
    print(f"   X shape : {X.shape} (nb_samples, nb_features)")

    # 4) Entraîner StandardScaler
    scaler = StandardScaler()
    scaler.fit(X)

    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler sauvegardé → {scaler_path}")

    # 5) Entraîner LabelEncoder
    le = LabelEncoder()
    le.fit(df["label"].values)

    le_path = os.path.join(OUTPUT_DIR, "label_encoder.pkl")
    joblib.dump(le, le_path)
    print(f"✅ LabelEncoder sauvegardé → {le_path}")
    print(f"   Classes détectées : {list(le.classes_)}")


# ============================================================
# 4) POINT D'ENTRÉE DU SCRIPT
# ============================================================

if __name__ == "__main__":
    main()

#!/bin/bash
# ── Entrypoint C3 — Initialisation du volume V2 ───────────────
#
# Au premier démarrage, V2 (/models) est vide.
# Ce script copie les modèles par défaut (intégrés dans l'image)
# vers V2 avant de lancer l'API.
#
# Lors des démarrages suivants, V2 contient déjà des fichiers
# (potentiellement mis à jour par C4) → on ne touche à rien.

set -e

MODELS_DIR="/models"
DEFAULTS_DIR="/models_default"

if [ -z "$(ls -A ${MODELS_DIR} 2>/dev/null)" ]; then
    echo "[INIT] Volume V2 vide — copie des modèles par défaut..."
    cp -v "${DEFAULTS_DIR}/"* "${MODELS_DIR}/"
    echo "[INIT] Modèles initiaux copiés dans V2 ($(ls ${MODELS_DIR} | wc -l) fichiers)."
else
    echo "[INIT] Volume V2 déjà peuplé ($(ls ${MODELS_DIR} | wc -l) fichiers). Aucune copie nécessaire."
fi

# Lancer la commande passée en argument (CMD du Dockerfile)
exec "$@"

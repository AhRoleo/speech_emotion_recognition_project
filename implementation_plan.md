# Dockerisation SER — Architecture 4 Containers + 2 Volumes

## Vue d'ensemble

Architecture orientée **amélioration continue** : les données remontent du frontend vers le dataset, et le pipeline réentraîne automatiquement les modèles pour les mettre à disposition de l'API.

```
┌─────────────┐     audio      ┌─────────────────┐    features    ┌──────────────┐
│  C1         │ ─────────────► │  C2             │ ─────────────► │  C3          │
│  Frontend   │ ◄───────────── │  Backend        │ ◄────────────  │  API Model   │
│  Streamlit  │   prédiction   │  FastAPI        │   prédiction   │  FastAPI     │
└─────────────┘                │                 │                └──────┬───────┘
                               │  📥 sauvegarde  │                       │ pioche
                               │  dans V1        │                  ┌────▼──────┐
                               └─────────────────┘                  │  V2       │
                                                                     │  Modèles  │
                                        ┌────────────────┐           │  .keras   │
                                        │  C4            │           └────▲──────┘
                                        │  Pipeline /    │ ─enregistre──┘
                                        │  Engine        │
                                        │  (Training)    │ ◄── pioche V1 (dataset)
                                        └────────────────┘
```

## Containers

| ID | Nom | Rôle | Image de base | Port |
|---|---|---|---|---|
| C1 | `frontend` | Interface Streamlit | `python:3.12-slim` | 8501 |
| C2 | `backend` | API FastAPI orchestratrice | `python:3.12-slim` | 8000 |
| C3 | `model-api` | API FastAPI de prédiction pure | `python:3.12-slim` | 8001 |
| C4 | `pipeline` | Script d'entraînement (PySpark + TF) | `python:3.12-slim` | — |

## Volumes

| ID | Nom Docker | Contenu | Accès |
|---|---|---|---|
| V1 | `ser-dataset` | Fichiers audio (.wav) uploadés + données d'entraînement | C2 (écriture), C4 (lecture) |
| V2 | `ser-models` | Fichiers `.keras` entraînés, `scaler.pkl`, `label_encoder.pkl` | C3 (lecture), C4 (écriture) |

## Proposed Changes

---

### Structure des fichiers à créer

```
speech_emotion_recognition_project-main/
│
├── docker-compose.yml           ← [NEW] Orchestrateur principal
├── .dockerignore                ← [NEW]
│
├── backend/
│   ├── Dockerfile               ← [NEW] C2 - FastAPI orchestratrice
│   ├── requirements.txt         ← [NEW] Dépendances allégées
│   └── app.py                   ← [MODIFY] Ajout sauvegarde audio dans V1
│
├── front-end/
│   ├── Dockerfile               ← [NEW] C1 - Streamlit
│   ├── requirements.txt         ← [NEW] Dépendances allégées
│   └── frontend.py              ← [MODIFY] URL backend par variable d'env
│
├── model-api/
│   ├── Dockerfile               ← [NEW] C3 - API de prédiction
│   ├── requirements.txt         ← [NEW]
│   └── app.py                   ← [NEW] FastAPI légère (features + predict)
│
└── pipeline/
    ├── Dockerfile               ← [NEW] C4 - Pipeline d'entraînement
    ├── requirements.txt         ← [NEW]
    └── train.py                 ← [NEW] Script d'entraînement (wrapper)
```

---

### [NEW] `docker-compose.yml`

Services, volumes, réseau interne `ser-network`, variables d'environnement et healthchecks.

```yaml
# Résumé des relations
services:
  frontend:      # C1 — dépend de backend
  backend:       # C2 — dépend de model-api, monte V1
  model-api:     # C3 — monte V2 (lecture)
  pipeline:      # C4 — monte V1 (lecture) + V2 (écriture)

volumes:
  ser-dataset:   # V1
  ser-models:    # V2
```

---

### [MODIFY] `backend/app.py`

Ajout d'un comportement de sauvegarde : à chaque appel `/predict`, le fichier audio reçu est copié dans le volume V1 (`/data/dataset/`) avec un nom horodaté pour enrichir le dataset.

---

### [MODIFY] `front-end/frontend.py`

L'URL de l'API backend sera lue depuis une variable d'environnement `BACKEND_URL` (défaut : `http://backend:8000`), ce qui rend le container portable sans modifier le code.

---

### [NEW] `model-api/app.py`

Extraction de la logique de prédiction de l'actuel `backend/app.py` :
- Reçoit des **features** (vecteur 185 floats) depuis C2 (pas l'audio brut)
- Charge les modèles depuis V2 (`/models/`)
- Retourne les probabilités

---

### [NEW] `pipeline/train.py`

Script d'entraînement autonome :
- Scanne V1 pour les nouveaux fichiers audio
- Extrait les features via `features.py`
- Réentraîne (fine-tune ou from scratch) un modèle Conv1D
- Sauvegarde le nouveau modèle dans V2 + met à jour `scaler.pkl`

> [!NOTE]
> C4 peut être configuré pour s'exécuter **on-demand** (démarré manuellement via `docker-compose run pipeline`) ou **automatiquement** à intervalles réguliers.

---

## User Review Required

> [!IMPORTANT]
> **Où se trouve le pipeline d'entraînement ?** Le projet actuel contient deux notebooks (`data_processing.ipynb` et `model_training1.ipynb`) mais pas de script Python autonome. Je devrai créer `pipeline/train.py` en extrayant la logique des notebooks. **Est-ce que tu préfères que je crée un script Python propre, ou que je conserve l'approche notebook avec `papermill` ?**

> [!IMPORTANT]
> **Quand C4 doit-il se déclencher ?** Options possibles :
> - **Manuel** : `docker-compose run pipeline` quand tu veux l'exécuter.
> - **Scheduled** : Via `cron` intégré dans le container (ex: toutes les nuits).
> - **Event-driven** : Dès qu'un certain nombre de nouveaux audios arrivent dans V1.

> [!WARNING]
> **Les audios sauvegardés dans V1 n'auront pas de label d'émotion** (on ne sait pas ce que l'utilisateur a dit). Le pipeline d'entraînement sur ces données sera **non supervisé** ou utilisera les **prédictions comme labels** (pseudo-labeling). Est-ce que c'est acceptable, ou les données d'entraînement dans V1 seront uniquement les datasets officiels (RAVDESS, CREMA-D) ?

## Verification Plan

```bash
# Build et lancement de tous les services
docker-compose up --build

# Tests
curl http://localhost:8000/health   # C2 backend
curl http://localhost:8001/health   # C3 model-api
# + ouvrir http://localhost:8501    # C1 frontend
```

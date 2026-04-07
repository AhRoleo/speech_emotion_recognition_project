# Walkthrough — Dockerisation SER

## Architecture finale

```
┌─────────────────────────────────────────────────────────────────┐
│  ser-network (réseau Docker interne)                            │
│                                                                 │
│  C1 :8501          C2 :8000          C3 :8001                   │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐              │
│  │ frontend │─────►│ backend  │─────►│model-api │◄──┐          │
│  │Streamlit │◄─────│ FastAPI  │◄─────│ FastAPI  │   │          │
│  └──────────┘      └────┬─────┘      └──────────┘   │          │
│                         │ sauvegarde                  │ lit      │
│                         ▼ audio                       │ modèles  │
│  ┌─────────────────────────────┐   ┌──────────────────┘         │
│  │  V1 — ser-dataset           │   │  V2 — ser-models            │
│  │  /data/dataset/<émotion>/   │   │  /models/*.keras            │
│  │  (pseudo-labels par dossier)│   │  /models/scaler.pkl         │
│  └──────────────┬──────────────┘   └──────────▲─────────────────│
│                 │ lit                          │ écrit            │
│                 │        C4                    │                  │
│                 └──────►┌──────────┐───────────┘                 │
│                         │ pipeline │ (toutes les 24h)            │
│                         └──────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

## Fichiers créés / modifiés

### Nouveaux fichiers

| Fichier | Rôle |
|---|---|
| `docker-compose.yml` | Orchestrateur — 4 services + 2 volumes + réseau |
| `.dockerignore` | Exclut notebooks, datasets, venvs du contexte de build |
| `backend/Dockerfile` | Image C2 (librosa + httpx, sans TensorFlow) |
| `backend/requirements.txt` | Dépendances allégées C2 |
| `front-end/Dockerfile` | Image C1 (Streamlit seul) |
| `front-end/requirements.txt` | Dépendances allégées C1 |
| `model_api/__init__.py` | Package Python |
| `model_api/Dockerfile` | Image C3 (TensorFlow + entrypoint seed V2) |
| `model_api/entrypoint.sh` | Copie les modèles dans V2 si V2 est vide |
| `model_api/requirements.txt` | Dépendances C3 (TF + sklearn) |
| `model_api/app.py` | API FastAPI de prédiction (JSON features in → prediction out) |
| `model_api/model_manager.py` | Lazy loading + cache des modèles depuis V2 |
| `model_api/schemas.py` | Schémas Pydantic (+ `PredictRequest`) |
| `pipeline/Dockerfile` | Image C4 (TF + librosa) |
| `pipeline/requirements.txt` | Dépendances C4 |
| `pipeline/train.py` | Script d'entraînement Conv1D — cycle 24h |

### Fichiers modifiés

| Fichier | Modification |
|---|---|
| `backend/app.py` | Suppression de `ModelManager`, ajout d'appels `httpx` vers C3, sauvegarde audio dans V1 |
| `front-end/frontend.py` | `BACKEND_URL` lue depuis `os.environ` (compatible Docker + local) |

## Démarrage

```bash
# Build et lancement de tous les containers
docker-compose up --build

# En arrière-plan
docker-compose up --build -d
```

> [!IMPORTANT]
> Le premier démarrage est **lent** (~5-15 min) car Docker télécharge les images
> et pip installe TensorFlow (~2 Go). Les lancements suivants sont rapides.

## Vérification

```bash
# Statut des containers
docker-compose ps

# Logs d'un service spécifique
docker-compose logs -f backend
docker-compose logs -f model-api
docker-compose logs -f pipeline

# Healthchecks
curl http://localhost:8000/health    # C2 backend
curl http://localhost:8001/health    # C3 model-api

# Interface utilisateur
# → http://localhost:8501            # C1 frontend
# → http://localhost:8000/docs       # Swagger C2
# → http://localhost:8001/docs       # Swagger C3
```

## Boucle d'amélioration continue

```
Upload audio (C1)
    ↓
Prédiction (C2 → C3)
    ↓
Sauvegarde dans V1 avec pseudo-label (C2)
    /data/dataset/happy/20240407_180000_abc123.wav
    ↓
Cycle 24h — C4 scanne V1
    ↓
Nouvel entraînement Conv1D
    ↓
Sauvegarde dans V2
    /models/ser_conv1d_pipeline_latest.keras
    ↓
C3 charge le nouveau modèle à la prochaine requête (lazy loading)
```

> [!NOTE]
> Le pipeline C4 nécessite au minimum **50 fichiers audio** (`MIN_TRAINING_SAMPLES`)
> pour démarrer un entraînement. En dessous de ce seuil, il log un avertissement
> et attend le prochain cycle.

## Arrêt propre

```bash
docker-compose down          # Stoppe et supprime les containers
docker-compose down -v       # + supprime les volumes V1 et V2
```

## Utilisation en local (sans Docker)

Les modifications sont rétro-compatibles. Pour lancer en local :

```bash
# Surcharger l'URL backend pour le frontend
set BACKEND_URL=http://localhost:8000   # Windows
export BACKEND_URL=http://localhost:8000  # Linux/Mac

# Lancer backend + model-api + frontend séparément
uvicorn model_api.app:app --port 8001
uvicorn backend.app:app --port 8000
streamlit run front-end/frontend.py
```

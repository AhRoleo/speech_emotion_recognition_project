# Speech Emotion Recognition — Documentation Projet

> **Reconnaissance automatique des émotions vocales** à partir d'un fichier audio,
> déployée sous forme de micro-services Docker avec une boucle d'amélioration continue.

---

## Objectif du projet

Ce projet a pour ambition de détecter automatiquement l'**émotion portée par une voix humaine** à partir d'un enregistrement audio (.wav ou .mp3).

Un utilisateur soumet un fichier audio via une interface web. En quelques secondes, le système lui retourne l'émotion dominante détectée parmi **8 classes** :

| Émotion | Label anglais |
|---|---|
| Colère | angry |
| Dégoût | disgust |
| Peur | fear |
| Joie | happy |
| Neutre | neutral |
| Surprise+ | ps |
| Tristesse | sad |
| Surprise | surprise |

La particularité de ce système : **il apprend en continu**. Chaque audio analysé est sauvegardé et réutilisé automatiquement pour réentraîner les modèles toutes les 24 heures.

---

## Architecture — Vue d'ensemble

Le projet est découpé en **4 micro-services Docker** orchestrés par `docker-compose`, communiquant via un réseau interne et deux volumes partagés.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ser-network (réseau Docker)                  │
│                                                                     │
│   ┌──────────────┐  HTTP   ┌──────────────┐  HTTP   ┌───────────┐  │
│   │     C1       │ ──────► │     C2       │ ──────► │    C3     │  │
│   │  Frontend    │ ◄────── │   Backend    │ ◄────── │ Model API │  │
│   │  Streamlit   │         │   FastAPI    │         │  FastAPI  │  │
│   │  :8501       │         │   :8000      │         │  :8001    │  │
│   └──────────────┘         └──────┬───────┘         └─────┬─────┘  │
│                                   │ save audio             │ lit     │
│                                   ▼                        ▼        │
│             ┌─────────────────────────┐    ┌──────────────────────┐ │
│             │  V1 — ser-dataset       │    │  V2 — ser-models     │ │
│             │  /data/dataset/         │    │  /models/            │ │
│             │  ├── angry/             │    │  ├── *.keras         │ │
│             │  ├── happy/             │    │  ├── scaler.pkl      │ │
│             │  ├── sad/   ...         │    │  └── label_enc.pkl   │ │
│             └──────────┬──────────────┘    └──────────▲───────────┘ │
│                        │ lit                          │ écrit       │
│                        │        ┌──────────────┐       │            │
│                        └──────► │     C4       │ ──────┘            │
│                                 │   Pipeline   │                    │
│                                 │  (24h cycle) │                    │
│                                 └──────────────┘                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Description des services

### C1 — Frontend (`Streamlit`, port 8501)

**Rôle** : Interface utilisateur web

L'interface permet à l'utilisateur de :
- Uploader un fichier audio (WAV ou MP3)
- Choisir le modèle d'IA à utiliser (liste dynamique depuis C3)
- Lancer l'analyse et visualiser le résultat
- Consulter les probabilités de toutes les émotions
- Revoir l'historique des 10 dernières analyses de la session

> L'URL de l'API backend est configurable via la sidebar ou la variable d'environnement `BACKEND_URL`.

---

### C2 — Backend orchestrateur (`FastAPI`, port 8000)

**Rôle** : Chef d'orchestre du pipeline de prédiction

Le backend est le cœur fonctionnel de l'application. À chaque requête de prédiction, il exécute 4 étapes :

```
1. Réception de l'audio (multipart/form-data depuis C1)
         ↓
2. Extraction des features audio (185 dimensions)
   ├── MFCC           [0:40]   — 40 coefficients
   ├── ZCR            [40]     — taux de passage par zéro
   ├── RMS Energy     [41]     — énergie RMS
   ├── Spectral Centr.[42]     — centroïde spectral
   ├── Spectral Rolloff[43]    — rolloff spectral
   ├── Chroma STFT    [44:56]  — 12 bins chromatiques
   ├── Mel Spectrogram[56:184] — 128 bins mel
   └── Entropy Energy [184]    — entropie de Shannon
         ↓
3. Appel à C3 (model-api) avec le vecteur de features en JSON
         ↓
4. Sauvegarde de l'audio dans V1 avec l'émotion prédite comme dossier
   (pseudo-label pour le pipeline C4)
```

Il expose aussi la liste des modèles disponibles (proxied depuis C3) et un endpoint healthcheck.

---

### C3 — Model API (`FastAPI`, port 8001)

**Rôle** : Service de prédiction pure (le seul à utiliser TensorFlow)

Ce service est délibérément **isolé** pour deux raisons :
- TensorFlow est lourd (~2 Go) : limiter son exposition à un seul container
- Les modèles peuvent être mis à jour par C4 sans redémarrer le reste du système

À chaque requête, C3 :
1. Reçoit un vecteur JSON de **185 features**
2. Charge le modèle `.keras` demandé (lazy loading + cache mémoire)
3. Normalise les features via `StandardScaler`
4. Reshape les données : `(1, 185)` → `(1, 185, 1)` pour Conv1D
5. Prédit les probabilités sur toutes les classes
6. Retourne l'émotion principale + toutes les probabilités triées

**Lazy loading + cache** : les modèles ne sont chargés en RAM qu'à la première utilisation et restent en cache pour toutes les requêtes suivantes.

**Initialisation de V2** : au premier démarrage, si le volume V2 est vide, un script `entrypoint.sh` copie automatiquement les modèles par défaut (intégrés dans l'image) dans V2.

---

### C4 — Pipeline d'entraînement (`Python`, sans port exposé)

**Rôle** : Réentraînement automatique toutes les 24h

Ce container tourne en arrière-plan selon un cycle régulier :

```
Toutes les 24h :

  1. Scanner V1 (/data/dataset/<émotion>/*.wav)
         ↓
  2. Extraire les features de chaque audio (via features.py)
         ↓
  3. Encoder les labels + normaliser (LabelEncoder + StandardScaler)
         ↓
  4. Entraîner un modèle Conv1D (50 epochs max, EarlyStopping)
         ↓
  5. Sauvegarder dans V2 :
       - ser_conv1d_pipeline_<timestamp>.keras   (versionné)
       - ser_conv1d_pipeline_latest.keras        (disponible immédiatement pour C3)
       - scaler_pipeline.pkl
       - label_encoder_pipeline.pkl
```

> [!NOTE]
> Le pipeline requiert un minimum de **50 fichiers audio** pour démarrer
> un entraînement. En dessous de ce seuil, il attend le prochain cycle.

---

## Les Volumes

### V1 — `ser-dataset` (Dataset audio)

Partagé entre **C2** (écriture) et **C4** (lecture).

```
/data/dataset/
├── angry/
│   ├── 20240407_181500_a3f9b2.wav   ← pseudo-labellisé par C2
│   └── ...
├── happy/
│   └── ...
├── sad/
│   └── ...
└── ...  (une sous-dossier par émotion)
```

Chaque audio est sauvegardé dans un **dossier portant l'émotion prédite**. Ce dossier sert de label pour C4 (pseudo-labeling). Il est possible de pré-charger des données officelles (RAVDESS, CREMA-D, TESS, SAVEE) dans cette structure pour améliorer la qualité d'entraînement dès le départ.

### V2 — `ser-models` (Modèles ML)

Partagé entre **C3** (lecture) et **C4** (écriture).

```
/models/
├── ser_conv1d_model.keras            ← modèle original (copié au 1er démarrage)
├── ser_conv1d_lstm_model.keras       ← modèle original Conv1D+LSTM
├── scaler.pkl                        ← normalisation originale
├── label_encoder.pkl                 ← encodage des labels
│
├── ser_conv1d_pipeline_20240407.keras  ← généré par C4
├── ser_conv1d_pipeline_latest.keras    ← dernière version C4
├── scaler_pipeline.pkl               ← scaler généré par C4
└── label_encoder_pipeline.pkl        ← encodeur généré par C4
```

---

## Boucle d'amélioration continue

C'est la valeur ajoutée architecturale majeure du projet :

```
  Utilisateur uploade un audio
          │
          ▼
  C2 extrait les features
          │
          ▼
  C3 prédit l'émotion (ex: "happy")
          │
          ├─────────────────────────────────────────────►  Réponse à C1
          │
          ▼
  C2 sauvegarde l'audio dans :
  V1/happy/20240407_181500_abc.wav
          │
          │ (toutes les 24h)
          ▼
  C4 lit tous les audios de V1
  C4 entraîne un nouveau Conv1D
  C4 sauvegarde le modèle dans V2
          │
          ▼
  C3 chargera ce nouveau modèle
  à la prochaine requête
```

> [!IMPORTANT]
> **Pseudo-labeling** : l'émotion prédite par le modèle courant est utilisée
> comme vérité terrain pour l'entraînement futur. Plus le modèle est précis,
> meilleure est la qualité des pseudo-labels, ce qui crée un cercle vertueux.

---

## API Reference

### C2 — Backend (port 8000)

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Statut du service + nombre de modèles disponibles |
| `GET` | `/models` | Liste des modèles `.keras` disponibles dans V2 |
| `POST` | `/predict` | **Prédiction** — envoyer `audio` (file) + `model_name` (form) |

**Exemple `/predict` :**
```bash
curl -X POST http://localhost:8000/predict \
  -F "audio=@mon_audio.wav" \
  -F "model_name=ser_conv1d_model.keras"
```

**Réponse :**
```json
{
  "model_name": "ser_conv1d_model.keras",
  "main_emotion": "happy",
  "main_label_fr": "Joie",
  "confidence": 0.8732,
  "probabilities": [
    { "emotion": "happy",   "label_fr": "Joie",      "probability": 0.8732 },
    { "emotion": "neutral", "label_fr": "Neutre",     "probability": 0.0821 },
    { "emotion": "sad",     "label_fr": "Tristesse",  "probability": 0.0447 }
  ]
}
```

### C3 — Model API (port 8001)

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Statut + modèles chargés en mémoire |
| `GET` | `/models` | Liste des modèles dans V2 |
| `POST` | `/predict` | Prédiction à partir d'un vecteur JSON de features |

> C3 est un service interne — en production, seul C2 devrait y accéder.

---

## Les modèles d'IA

### Architecture Conv1D

Les modèles utilisent une architecture de réseau de neurones convolutif 1D, adaptée aux séries temporelles audio :

```
Input (185, 1)
    │
    ▼  Conv1D(64)  → BatchNorm → MaxPool → Dropout(0.25)
    ▼  Conv1D(128) → BatchNorm → MaxPool → Dropout(0.25)
    ▼  Conv1D(64)  → BatchNorm → GlobalAvgPool → Dropout(0.30)
    ▼  Dense(128)  → Dropout(0.30)
    ▼  Dense(8, softmax)   ← 8 émotions
```

### Modèles disponibles

| Fichier | Architecture | Usage |
|---|---|---|
| `ser_conv1d_model.keras` | Conv1D pur | Modèle original, rapide |
| `ser_conv1d_lstm_model.keras` | Conv1D + LSTM | Modèle original, plus contextuel |
| `ser_conv1d_pipeline_latest.keras` | Conv1D (pipeline) | Généré par C4, mis à jour en continu |

---

## Structure du projet

```
speech_emotion_recognition_project-main/
│
├── docker-compose.yml           ← Orchestrateur des 4 services
├── .dockerignore                ← Fichiers exclus du build Docker
│
├── features.py                  ← Extraction audio (185 features) — partagé
├── main.py                      ← Lancement local (hors Docker)
│
├── backend/                     ── C2 — Orchestrateur ──────────────────
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py                   ← API FastAPI + sauvegarde V1
│   └── schemas.py               ← Modèles Pydantic
│
├── front-end/                   ── C1 — Interface ──────────────────────
│   ├── Dockerfile
│   ├── requirements.txt
│   └── frontend.py              ← Application Streamlit
│
├── model_api/                   ── C3 — Prédiction ─────────────────────
│   ├── Dockerfile
│   ├── entrypoint.sh            ← Seed initial de V2
│   ├── requirements.txt
│   ├── app.py                   ← API FastAPI de prédiction
│   ├── model_manager.py         ← Lazy loading + cache des modèles
│   └── schemas.py               ← Modèles Pydantic (+ PredictRequest)
│
├── pipeline/                    ── C4 — Entraînement ───────────────────
│   ├── Dockerfile
│   ├── requirements.txt
│   └── train.py                 ← Script d'entraînement (cycle 24h)
│
├── ser_conv1d_model.keras       ← Modèles pré-entraînés (copiés dans V2)
├── ser_conv1d_lstm_model.keras
├── scaler.pkl
└── label_encoder.pkl
```

---

## Stack technique

| Catégorie | Technologie | Usage |
|---|---|---|
| **IA / ML** | TensorFlow 2.19+ / Keras | Entraînement et inférence Conv1D |
| **Audio** | Librosa 0.11 | Extraction de features (MFCC, Mel, Chroma…) |
| **Audio I/O** | SoundFile | Décodage des fichiers audio |
| **Preprocessing** | Scikit-Learn | StandardScaler, LabelEncoder |
| **Backend API** | FastAPI + Uvicorn | API REST asynchrone |
| **Frontend** | Streamlit | Interface web interactive |
| **HTTP interne** | HTTPX | Appels async entre C2 et C3 |
| **Sérialisation** | Pydantic v2 | Validation et schémas des données |
| **Persistance** | Joblib | Sérialisation des scalers/encoders |
| **Containerisation** | Docker + Docker Compose | Orchestration des micro-services |
| **Réseau Docker** | Bridge network | Communication inter-containers |

---

## Démarrage rapide

### Prérequis

- Docker Desktop installé et démarré
- Python 3.12 (pour une exécution locale uniquement)

### Lancement avec Docker (recommandé)

```bash
# Cloner / se placer dans le projet
cd speech_emotion_recognition_project-main

# Build + lancement de tous les services
docker-compose up --build

# En arrière-plan
docker-compose up --build -d
```

> [!IMPORTANT]
> Le **premier démarrage prend 5 à 15 minutes** : Docker télécharge les images
> de base et pip installe TensorFlow (~2 Go). Les démarrages suivants
> se font en quelques secondes.

### Accès aux interfaces

| Service | URL | Description |
|---|---|---|
| Interface utilisateur | http://localhost:8501 | Application Streamlit |
| Swagger Backend | http://localhost:8000/docs | Documentation API C2 |
| Swagger Model API | http://localhost:8001/docs | Documentation API C3 |

### Arrêt propre

```bash
docker-compose down          # Arrête les containers (conserve V1 et V2)
docker-compose down -v       # Arrête + supprime les volumes (reset complet)
```

### Commandes utiles

```bash
# Voir l'état de tous les services
docker-compose ps

# Suivre les logs en temps réel
docker-compose logs -f

# Logs d'un service spécifique
docker-compose logs -f pipeline      # Suivre les cycles d'entraînement
docker-compose logs -f model-api     # Suivre les prédictions TF

# Forcer un cycle d'entraînement immédiat
docker-compose exec pipeline python -c "
from pipeline.train import run_training; run_training()
"

# Inspecter le contenu de V1 (dataset)
docker-compose exec backend ls /data/dataset/

# Inspecter le contenu de V2 (modèles)
docker-compose exec model-api ls /models/
```

---

## Configuration avancée

Toutes les variables d'environnement sont surchargeables dans `docker-compose.yml` :

| Variable | Service | Valeur par défaut | Description |
|---|---|---|---|
| `BACKEND_URL` | C1 | `http://backend:8000` | URL du backend C2 |
| `MODEL_API_URL` | C2 | `http://model-api:8001` | URL de la model-api C3 |
| `SER_DATASET_DIR` | C2, C4 | `/data/dataset` | Chemin du dataset dans V1 |
| `SER_MODELS_DIR` | C3, C4 | `/models` | Chemin des modèles dans V2 |
| `TRAINING_INTERVAL_HOURS` | C4 | `24` | Fréquence du pipeline (heures) |
| `MIN_TRAINING_SAMPLES` | C4 | `50` | Seuil minimum avant entraînement |

---

## Jeux de données supportés

Le pipeline d'entraînement supporte les datasets audio standards pour la SER. Le module `features.py` inclut une fonction `extract_label()` qui détecte automatiquement le format :

| Dataset | Format de nommage | Exemples d'émotions |
|---|---|---|
| **RAVDESS** | `03-01-05-01-02-01-24.wav` | 8 émotions |
| **CREMA-D** | `1001_DFA_ANG_XX.wav` | 6 émotions |
| **TESS** | `OAF_back_angry.wav` | 7 émotions |
| **SAVEE** | `DC_a01.wav` | 7 émotions |

Pour pré-charger un dataset officiel dans V1, organiser les fichiers en sous-dossiers par émotion :
```bash
V1/data/dataset/
  angry/  disgust/  fear/  happy/  neutral/  sad/  surprise/
```

---

*Projet développé dans le cadre de la formation ESGI · Python 3.12 requis · Licence MIT*

##  Reconnaissance des Émotions Vocales (Speech Emotion Recognition)

Ce projet a pour objectif de réaliser une Reconnaissance des Émotions à partir de la voix (SER) en utilisant des techniques de traitement du signal audio et le framework distribué PySpark.

## ⚙️ Pré-requis

- Python 3.12
- Java 17 (required for PySpark)
- Pas obligatoire, mais Conda est recommandé

### Installation de Java 17 (WSL / Ubuntu)

```
sudo apt update
sudo apt install openjdk-17-jdk
```

Vérifier l’installation :
```
java -version
```
Vous devez obtenir :
```
openjdk version "17"
```

## Mise en place de l’environnement

### Création d’un environnement virtuel

Avec conda:
```
conda create -n ser_env python=3.12 -y
conda activate ser_env
```

Si vous n’utilisez pas conda :
```
python3 -m venv ser_env
source ser_env/bin/activate
```

### Installation des dépendances

```
pip install -r requirements.txt
```

### Structure du projet

Le notebook est à la racine du projet. Le dossier qui contient les données, Dataset/, est placé dans le dossier parent du repo. 

## Lancement du projet

Si vous utilisez Jupyter Notebook:

```
pip install jupyter ipykernel
python -m ipykernel install --user --name ser_env --display-name "Python (ser_env)"
jupyter notebook
```

## Architecture Python 3.12 requis!!! mdr

```
Streamlit (front-end)  →  POST /predict  →  FastAPI (backend)
                                               ├── features.py (extraction 185 dims)
                                               ├── scaler.pkl (normalisation)
                                               └── *.keras (modèles Conv1D)
```

---

## Installer toute le projet Python 3.12 requis!!!

### 2. Générer le scaler (si les Parquet sont disponibles)
```bash
python save_scaler.py <chemin_train_features.parquet>
```
> Sans scaler, l'API fonctionne mais les prédictions seront moins fiables (features non normalisées).

### 3. Lancer le backend
```bash
uvicorn backend.app:app --reload --port 8000
```
Ouvrir `http://localhost:8000/docs` pour la doc Swagger.

### 4. Lancer le front-end
```bash
streamlit run front-end/front-test.py
```

### 5. Lancer les tests
```bash
pytest tests/ -v
```


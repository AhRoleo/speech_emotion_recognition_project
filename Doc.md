## Architecture Python 3.12 requis!!! mdr

```
Streamlit (front-end)  →  POST /predict  →  FastAPI (backend)
                                               ├── features.py (extraction 185 dims)
                                               ├── scaler.pkl (normalisation)
                                               └── *.keras (modèles Conv1D)
```

---

## Comment utiliser? Python 3.12 requis!!! mdr

### 1. Installer les dépendances
```bash
cd "c:\Users\joane\Desktop\ESGI4\Spark Core\with"
pip install -r requirements.txt
```

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

"""
Tests pour l'API FastAPI SER.

Utilise le TestClient de FastAPI (httpx) pour tester les endpoints
sans avoir besoin de lancer le serveur.
"""

import sys
from pathlib import Path

import pytest

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from backend.app import app
from features import generate_test_wav


@pytest.fixture
def client():
    """Fixture fournissant un TestClient FastAPI."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests pour GET /health."""

    def test_health_returns_200(self, client):
        """Le healthcheck doit retourner un statut 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok(self, client):
        """Le healthcheck doit indiquer status=ok."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_contains_model_counts(self, client):
        """Le healthcheck doit contenir les compteurs de modèles."""
        response = client.get("/health")
        data = response.json()
        assert "models_loaded" in data
        assert "models_available" in data


class TestModelsEndpoint:
    """Tests pour GET /models."""

    def test_models_returns_200(self, client):
        """La liste des modèles doit retourner un statut 200."""
        response = client.get("/models")
        assert response.status_code == 200

    def test_models_returns_list(self, client):
        """La réponse doit contenir une liste de modèles."""
        response = client.get("/models")
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_models_contain_keras_model(self, client):
        """La liste doit contenir le modèle ser_conv1d_model.keras."""
        response = client.get("/models")
        data = response.json()
        file_names = [m["file_name"] for m in data["models"]]
        assert "ser_conv1d_model.keras" in file_names


class TestPredictEndpoint:
    """Tests pour POST /predict."""

    def test_predict_returns_200(self, client):
        """Une prédiction valide doit retourner un statut 200."""
        wav_bytes = generate_test_wav(duration=1.0, freq=440.0)
        response = client.post(
            "/predict",
            files={"audio": ("test.wav", wav_bytes, "audio/wav")},
            data={"model_name": "ser_conv1d_model.keras"},
        )
        assert response.status_code == 200

    def test_predict_returns_emotion(self, client):
        """La prédiction doit contenir une émotion principale."""
        wav_bytes = generate_test_wav(duration=1.0, freq=440.0)
        response = client.post(
            "/predict",
            files={"audio": ("test.wav", wav_bytes, "audio/wav")},
            data={"model_name": "ser_conv1d_model.keras"},
        )
        data = response.json()
        assert "main_emotion" in data
        assert "confidence" in data
        assert "probabilities" in data

    def test_predict_confidence_range(self, client):
        """La confiance doit être entre 0 et 1."""
        wav_bytes = generate_test_wav(duration=1.0, freq=440.0)
        response = client.post(
            "/predict",
            files={"audio": ("test.wav", wav_bytes, "audio/wav")},
            data={"model_name": "ser_conv1d_model.keras"},
        )
        data = response.json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_probabilities_sum_to_one(self, client):
        """Les probabilités doivent sommer approximativement à 1."""
        wav_bytes = generate_test_wav(duration=1.0, freq=440.0)
        response = client.post(
            "/predict",
            files={"audio": ("test.wav", wav_bytes, "audio/wav")},
            data={"model_name": "ser_conv1d_model.keras"},
        )
        data = response.json()
        total = sum(p["probability"] for p in data["probabilities"])
        assert abs(total - 1.0) < 0.01

    def test_predict_invalid_model_returns_404(self, client):
        """Un modèle inexistant doit retourner un statut 404."""
        wav_bytes = generate_test_wav(duration=1.0, freq=440.0)
        response = client.post(
            "/predict",
            files={"audio": ("test.wav", wav_bytes, "audio/wav")},
            data={"model_name": "non_existent_model.keras"},
        )
        assert response.status_code == 404

    def test_predict_empty_audio_returns_400(self, client):
        """Un fichier audio vide doit retourner un statut 400."""
        response = client.post(
            "/predict",
            files={"audio": ("test.wav", b"", "audio/wav")},
            data={"model_name": "ser_conv1d_model.keras"},
        )
        assert response.status_code == 400

"""
Tests unitaires pour le module d'extraction de features.

Objectifs :
- Vérifier que extract_features fonctionne correctement
- Tester la robustesse face aux erreurs (audio vide, invalide…)
- Vérifier la cohérence des features (dimensions, types, valeurs)
- Tester extract_label pour différents datasets (RAVDESS, CREMA-D, TESS, SAVEE)
"""

# ============================================================
# 1) IMPORTS & CONFIGURATION
# ============================================================

import math
import numpy as np
import pytest

from features import (
    FEATURE_DIM,
    extract_features,
    extract_label,
    generate_test_wav,
)


# ============================================================
# 2) TESTS DE LA FONCTION extract_features
# ============================================================

class TestExtractFeatures:
    """Tests unitaires pour la fonction extract_features."""

    # ------------------------------------------------------------
    # 2.1 Vérification des dimensions
    # ------------------------------------------------------------
    def test_returns_correct_dimensions(self):
        """Le vecteur de features doit avoir exactement 185 dimensions."""
        wav_bytes = generate_test_wav(duration=1.0, freq=440.0)
        features = extract_features(wav_bytes)
        assert len(features) == FEATURE_DIM

    # ------------------------------------------------------------
    # 2.2 Vérification absence de NaN
    # ------------------------------------------------------------
    def test_no_nan_values(self):
        """Le vecteur ne doit contenir aucun NaN."""
        wav_bytes = generate_test_wav(duration=1.0, freq=440.0)
        features = extract_features(wav_bytes)
        for i, f in enumerate(features):
            assert not math.isnan(f), f"NaN trouvé à l'index {i}"

    # ------------------------------------------------------------
    # 2.3 Vérification absence de valeurs infinies
    # ------------------------------------------------------------
    def test_no_inf_values(self):
        """Le vecteur ne doit contenir aucune valeur infinie."""
        wav_bytes = generate_test_wav(duration=1.0, freq=440.0)
        features = extract_features(wav_bytes)
        for i, f in enumerate(features):
            assert not math.isinf(f), f"Inf trouvé à l'index {i}"

    # ------------------------------------------------------------
    # 2.4 Test sur un signal silencieux
    # ------------------------------------------------------------
    def test_silence_returns_valid_features(self):
        """Un signal silencieux doit retourner un vecteur valide de 185 dims."""
        wav_bytes = generate_test_wav(duration=1.0, freq=0.0)
        features = extract_features(wav_bytes)
        assert len(features) == FEATURE_DIM
        # L'entropie d'énergie pour un silence doit être 0
        entropy = features[184]
        assert entropy == 0.0

    # ------------------------------------------------------------
    # 2.5 Test sur un audio très court
    # ------------------------------------------------------------
    def test_short_audio(self):
        """Un audio très court doit retourner un vecteur de 185 dims."""
        wav_bytes = generate_test_wav(duration=0.2, freq=440.0)
        features = extract_features(wav_bytes)
        assert len(features) == FEATURE_DIM

    # ------------------------------------------------------------
    # 2.6 Deux fréquences différentes → features différentes
    # ------------------------------------------------------------
    def test_different_frequencies_produce_different_features(self):
        """Deux fréquences différentes doivent produire des features différentes."""
        wav_low = generate_test_wav(duration=1.0, freq=220.0)
        wav_high = generate_test_wav(duration=1.0, freq=880.0)
        features_low = extract_features(wav_low)
        features_high = extract_features(wav_high)
        assert features_low != features_high

    # ------------------------------------------------------------
    # 2.7 Vérification du type des features
    # ------------------------------------------------------------
    def test_features_are_floats(self):
        """Toutes les features doivent être des float."""
        wav_bytes = generate_test_wav(duration=1.0, freq=440.0)
        features = extract_features(wav_bytes)
        for i, f in enumerate(features):
            assert isinstance(f, float), f"Feature {i} n'est pas un float: {type(f)}"

    # ------------------------------------------------------------
    # 2.8 Audio invalide → vecteur nul
    # ------------------------------------------------------------
    def test_invalid_audio_returns_zero_vector(self):
        """Un fichier audio invalide doit retourner un vecteur nul."""
        features = extract_features(b"not_a_valid_wav_file")
        assert len(features) == FEATURE_DIM
        assert all(f == 0.0 for f in features)

    # ------------------------------------------------------------
    # 2.9 Audio vide → vecteur nul
    # ------------------------------------------------------------
    def test_empty_bytes_returns_zero_vector(self):
        """Des bytes vides doivent retourner un vecteur nul."""
        features = extract_features(b"")
        assert len(features) == FEATURE_DIM
        assert all(f == 0.0 for f in features)

    # ------------------------------------------------------------
    # 2.10 Vérification de la section MFCC
    # ------------------------------------------------------------
    def test_mfcc_section(self):
        """Les 40 premières valeurs (MFCC) doivent être non nulles pour un vrai signal."""
        wav_bytes = generate_test_wav(duration=1.0, freq=440.0)
        features = extract_features(wav_bytes)
        mfcc_section = features[0:40]
        assert any(f != 0.0 for f in mfcc_section)


# ============================================================
# 3) TESTS DE LA FONCTION extract_label
# ============================================================

class TestExtractLabel:
    """Tests unitaires pour la fonction extract_label."""

    # ------------------------------------------------------------
    # 3.1 Tests RAVDESS
    # ------------------------------------------------------------
    def test_ravdess_angry(self):
        """RAVDESS : code 05 = angry."""
        assert extract_label("/dataset/03-01-05-01-02-01-24.wav") == "angry"

    def test_ravdess_happy(self):
        """RAVDESS : code 03 = happy."""
        assert extract_label("/dataset/03-01-03-01-02-01-24.wav") == "happy"

    def test_ravdess_sad(self):
        """RAVDESS : code 04 = sad."""
        assert extract_label("/dataset/03-01-04-01-02-01-24.wav") == "sad"

    def test_ravdess_neutral(self):
        """RAVDESS : code 01 = neutral."""
        assert extract_label("/dataset/03-01-01-01-02-01-24.wav") == "neutral"

    def test_ravdess_surprise(self):
        """RAVDESS : code 08 = surprise."""
        assert extract_label("/dataset/03-01-08-01-02-01-24.wav") == "surprise"

    # ------------------------------------------------------------
    # 3.2 Tests CREMA-D
    # ------------------------------------------------------------
    def test_crema_angry(self):
        """CREMA-D : ANG = angry."""
        assert extract_label("/dataset/1001_DFA_ANG_XX.wav") == "angry"

    def test_crema_happy(self):
        """CREMA-D : HAP = happy."""
        assert extract_label("/dataset/1001_DFA_HAP_XX.wav") == "happy"

    def test_crema_sad(self):
        """CREMA-D : SAD = sad."""
        assert extract_label("/dataset/1001_DFA_SAD_XX.wav") == "sad"

    def test_crema_neutral(self):
        """CREMA-D : NEU = neutral."""
        assert extract_label("/dataset/1001_DFA_NEU_XX.wav") == "neutral"

    # ------------------------------------------------------------
    # 3.3 Tests TESS
    # ------------------------------------------------------------
    def test_tess_angry(self):
        """TESS : OAF_back_angry.wav → angry."""
        assert extract_label("/dataset/OAF_back_angry.wav") == "angry"

    def test_tess_happy(self):
        """TESS : OAF_back_happy.wav → happy."""
        assert extract_label("/dataset/OAF_back_happy.wav") == "happy"

    def test_tess_ps(self):
        """TESS : OAF_back_ps.wav → ps."""
        assert extract_label("/dataset/OAF_back_ps.wav") == "ps"

    # ------------------------------------------------------------
    # 3.4 Tests SAVEE
    # ------------------------------------------------------------
    def test_savee_angry(self):
        """SAVEE : DC_a01.wav → angry."""
        assert extract_label("/dataset/DC_a01.wav") == "angry"

    def test_savee_sad(self):
        """SAVEE : DC_sa01.wav → sad."""
        assert extract_label("/dataset/DC_sa01.wav") == "sad"

    def test_savee_surprise(self):
        """SAVEE : DC_su01.wav → surprise."""
        assert extract_label("/dataset/DC_su01.wav") == "surprise"

    # ------------------------------------------------------------
    # 3.5 Format inconnu
    # ------------------------------------------------------------
    def test_unknown_format(self):
        """Un format inconnu doit retourner 'unknown'."""
        assert extract_label("/dataset/random_file.wav") == "unknown"

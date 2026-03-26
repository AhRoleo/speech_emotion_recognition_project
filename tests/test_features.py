"""
Tests unitaires pour le module d'extraction de features.
"""

import math

import numpy as np
import pytest

from features import (
    FEATURE_DIM,
    extract_features,
    extract_label,
    generate_test_wav,
)


# =============================================================================
# Tests de extract_features
# =============================================================================


class TestExtractFeatures:
    """Tests pour la fonction extract_features."""

    def test_returns_correct_dimensions(self):
        """Le vecteur de features doit avoir exactement 185 dimensions."""
        wav_bytes = generate_test_wav(duration=1.0, freq=440.0)
        features = extract_features(wav_bytes)
        assert len(features) == FEATURE_DIM

    def test_no_nan_values(self):
        """Le vecteur ne doit contenir aucun NaN."""
        wav_bytes = generate_test_wav(duration=1.0, freq=440.0)
        features = extract_features(wav_bytes)
        for i, f in enumerate(features):
            assert not math.isnan(f), f"NaN trouvé à l'index {i}"

    def test_no_inf_values(self):
        """Le vecteur ne doit contenir aucune valeur infinie."""
        wav_bytes = generate_test_wav(duration=1.0, freq=440.0)
        features = extract_features(wav_bytes)
        for i, f in enumerate(features):
            assert not math.isinf(f), f"Inf trouvé à l'index {i}"

    def test_silence_returns_valid_features(self):
        """Un signal silencieux doit retourner un vecteur valide de 185 dims."""
        wav_bytes = generate_test_wav(duration=1.0, freq=0.0)
        features = extract_features(wav_bytes)
        assert len(features) == FEATURE_DIM
        # L'entropie d'énergie pour un silence doit être 0
        entropy = features[184]
        assert entropy == 0.0

    def test_short_audio(self):
        """Un audio très court (0.1s) doit retourner un vecteur de 185 dims."""
        wav_bytes = generate_test_wav(duration=0.2, freq=440.0)
        features = extract_features(wav_bytes)
        assert len(features) == FEATURE_DIM

    def test_different_frequencies_produce_different_features(self):
        """Deux fréquences différentes doivent produire des features différentes."""
        wav_low = generate_test_wav(duration=1.0, freq=220.0)
        wav_high = generate_test_wav(duration=1.0, freq=880.0)
        features_low = extract_features(wav_low)
        features_high = extract_features(wav_high)
        # Au moins certaines features doivent différer
        assert features_low != features_high

    def test_features_are_floats(self):
        """Toutes les features doivent être des float."""
        wav_bytes = generate_test_wav(duration=1.0, freq=440.0)
        features = extract_features(wav_bytes)
        for i, f in enumerate(features):
            assert isinstance(f, float), f"Feature {i} n'est pas un float: {type(f)}"

    def test_invalid_audio_returns_zero_vector(self):
        """Un fichier audio invalide doit retourner un vecteur nul."""
        features = extract_features(b"not_a_valid_wav_file")
        assert len(features) == FEATURE_DIM
        assert all(f == 0.0 for f in features)

    def test_empty_bytes_returns_zero_vector(self):
        """Des bytes vides doivent retourner un vecteur nul."""
        features = extract_features(b"")
        assert len(features) == FEATURE_DIM
        assert all(f == 0.0 for f in features)

    def test_mfcc_section(self):
        """Les 40 premières valeurs (MFCC) doivent être non nulles pour un vrai signal."""
        wav_bytes = generate_test_wav(duration=1.0, freq=440.0)
        features = extract_features(wav_bytes)
        mfcc_section = features[0:40]
        # Au moins quelques MFCC doivent être non nuls
        assert any(f != 0.0 for f in mfcc_section)


# =============================================================================
# Tests de extract_label
# =============================================================================


class TestExtractLabel:
    """Tests pour la fonction extract_label."""

    def test_ravdess_angry(self):
        """RAVDESS : code 05 = angry."""
        label = extract_label("/dataset/03-01-05-01-02-01-24.wav")
        assert label == "angry"

    def test_ravdess_happy(self):
        """RAVDESS : code 03 = happy."""
        label = extract_label("/dataset/03-01-03-01-02-01-24.wav")
        assert label == "happy"

    def test_ravdess_sad(self):
        """RAVDESS : code 04 = sad."""
        label = extract_label("/dataset/03-01-04-01-02-01-24.wav")
        assert label == "sad"

    def test_ravdess_neutral(self):
        """RAVDESS : code 01 = neutral."""
        label = extract_label("/dataset/03-01-01-01-02-01-24.wav")
        assert label == "neutral"

    def test_ravdess_surprise(self):
        """RAVDESS : code 08 = surprise."""
        label = extract_label("/dataset/03-01-08-01-02-01-24.wav")
        assert label == "surprise"

    def test_crema_angry(self):
        """CREMA-D : ANG = angry."""
        label = extract_label("/dataset/1001_DFA_ANG_XX.wav")
        assert label == "angry"

    def test_crema_happy(self):
        """CREMA-D : HAP = happy."""
        label = extract_label("/dataset/1001_DFA_HAP_XX.wav")
        assert label == "happy"

    def test_crema_sad(self):
        """CREMA-D : SAD = sad."""
        label = extract_label("/dataset/1001_DFA_SAD_XX.wav")
        assert label == "sad"

    def test_crema_neutral(self):
        """CREMA-D : NEU = neutral."""
        label = extract_label("/dataset/1001_DFA_NEU_XX.wav")
        assert label == "neutral"

    def test_tess_angry(self):
        """TESS : OAF_back_angry.wav → angry."""
        label = extract_label("/dataset/OAF_back_angry.wav")
        assert label == "angry"

    def test_tess_happy(self):
        """TESS : OAF_back_happy.wav → happy."""
        label = extract_label("/dataset/OAF_back_happy.wav")
        assert label == "happy"

    def test_tess_ps(self):
        """TESS : OAF_back_ps.wav → ps."""
        label = extract_label("/dataset/OAF_back_ps.wav")
        assert label == "ps"

    def test_savee_angry(self):
        """SAVEE : DC_a01.wav → angry."""
        label = extract_label("/dataset/DC_a01.wav")
        assert label == "angry"

    def test_savee_sad(self):
        """SAVEE : DC_sa01.wav → sad."""
        label = extract_label("/dataset/DC_sa01.wav")
        assert label == "sad"

    def test_savee_surprise(self):
        """SAVEE : DC_su01.wav → surprise."""
        label = extract_label("/dataset/DC_su01.wav")
        assert label == "surprise"

    def test_unknown_format(self):
        """Un format inconnu doit retourner 'unknown'."""
        label = extract_label("/dataset/random_file.wav")
        assert label == "unknown"

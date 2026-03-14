"""
Module partagé d'extraction de features audio pour le pipeline SER.

Ce module fournit les fonctions de traitement audio utilisées à la fois
par le backend API (inférence) et par les tests unitaires.

Composition du vecteur de features (185 dimensions) :
    [0:40]    MFCC           — 40 coefficients
    [40]      ZCR            — 1 valeur
    [41]      RMS Energy     — 1 valeur
    [42]      Spectral Centr — 1 valeur
    [43]      Spectral Roll  — 1 valeur
    [44:56]   Chroma STFT    — 12 valeurs
    [56:184]  Mel Spectro    — 128 valeurs
    [184]     Entropy Energy — 1 valeur
"""

import io
import os

import librosa
import numpy as np
import soundfile as sf

SAMPLE_RATE = 22050
FEATURE_DIM = 185

# Les 8 classes du modèle Conv1D (ordre alphabétique = ordre LabelEncoder)
EMOTION_LABELS = [
    "angry", "disgust", "fear", "happy",
    "neutral", "ps", "sad", "surprise",
]

# Mapping emoji pour l'affichage front-end
EMOTION_DISPLAY = {
    "angry":    ("Colère",    "😡"),
    "disgust":  ("Dégoût",    "🤢"),
    "fear":     ("Peur",      "😨"),
    "happy":    ("Joie",      "😄"),
    "neutral":  ("Neutre",    "😐"),
    "ps":       ("Surprise+", "😮"),
    "sad":      ("Tristesse", "😢"),
    "surprise": ("Surprise",  "😲"),
}


def extract_features(audio_bytes: bytes) -> list[float]:
    """
    Extrait un vecteur de 185 features (9 caractéristiques Babko)
    à partir de données audio binaires WAV.

    Args:
        audio_bytes: Contenu binaire d'un fichier WAV.

    Returns:
        Liste de 185 floats représentant les features audio.
    """
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)

        # 1. MFCC — 40 coefficients (moyenne temporelle)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)  # (40,)

        # 2. Zero Crossing Rate (moyenne temporelle)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean = np.mean(zcr)

        # 3. RMS Energy (moyenne temporelle)
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)

        # 4. Spectral Centroid (moyenne temporelle)
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)
        sc_mean = np.mean(sc)

        # 5. Spectral Rolloff (moyenne temporelle)
        sro = librosa.feature.spectral_rolloff(y=y, sr=sr)
        sro_mean = np.mean(sro)

        # 6. Chroma STFT — 12 bins (moyenne temporelle)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)  # (12,)

        # 7. Mel Spectrogram — 128 bins (moyenne temporelle)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_mean = np.mean(mel, axis=1)  # (128,)

        # 8. Entropy of Energy (entropie de Shannon)
        frame_length = 2048
        hop_length = 512
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        energy = np.sum(frames ** 2, axis=0)
        energy_sum = np.sum(energy)
        if energy_sum > 0:
            prob = energy / energy_sum
            prob = prob[prob > 0]
            entropy = -np.sum(prob * np.log2(prob))
        else:
            entropy = 0.0

        # Concaténation → vecteur de 185 dimensions
        features = np.concatenate([
            mfcc_mean,       # 40
            [zcr_mean],      # 1
            [rms_mean],      # 1
            [sc_mean],       # 1
            [sro_mean],      # 1
            chroma_mean,     # 12
            mel_mean,        # 128
            [entropy],       # 1
        ])

        return [float(f) for f in features]

    except Exception:
        # En cas d'erreur de décodage, retourner un vecteur nul
        return [0.0] * FEATURE_DIM


def extract_label(path: str) -> str:
    """
    Extrait le label d'émotion à partir du chemin du fichier audio.
    Supporte RAVDESS, CREMA-D, TESS et SAVEE.

    Args:
        path: Chemin complet du fichier audio.

    Returns:
        Nom de l'émotion (str), ou "unknown" si non reconnu.
    """
    filename = os.path.basename(path).lower()

    # --- RAVDESS : ex. "03-01-05-01-02-01-24.wav" ---
    if filename.startswith("03-01") or filename.count("-") >= 5:
        parts = filename.replace(".wav", "").split("-")
        if len(parts) >= 3:
            ravdess_map = {
                "01": "neutral", "02": "neutral",
                "03": "happy",   "04": "sad",
                "05": "angry",   "06": "fear",
                "07": "disgust", "08": "surprise",
            }
            return ravdess_map.get(parts[2], "unknown")

    # --- CREMA-D / TESS ---
    if "_" in filename:
        parts = filename.replace(".wav", "").split("_")

        # TESS : "OAF_back_angry.wav" → parts[2] = émotion
        if len(parts) == 3 and parts[2] in (
            "angry", "disgust", "fear", "happy", "neutral",
            "sad", "surprise", "ps",
        ):
            return parts[2]

        # CREMA-D : "1001_DFA_ANG_XX.wav" → parts[2] = code émotion
        if len(parts) >= 3:
            crema_map = {
                "ANG": "angry", "DIS": "disgust", "FEA": "fear",
                "HAP": "happy", "NEU": "neutral", "SAD": "sad",
            }
            emotion_code = parts[2].upper()
            return crema_map.get(emotion_code, "unknown")

    # --- SAVEE : ex. "DC_a01.wav" ---
    if "_" in filename:
        savee_name = filename.replace(".wav", "")
        emotion_part = savee_name.split("_")[-1]
        emotion_letters = "".join(c for c in emotion_part if c.isalpha())
        savee_map = {
            "a": "angry", "d": "disgust", "f": "fear",
            "h": "happy", "n": "neutral", "sa": "sad",
            "su": "surprise",
        }
        return savee_map.get(emotion_letters, "unknown")

    return "unknown"


def generate_test_wav(duration: float = 1.0, freq: float = 440.0) -> bytes:
    """
    Génère un fichier WAV synthétique en mémoire pour les tests.

    Args:
        duration: Durée en secondes.
        freq: Fréquence du sinus en Hz (0 pour silence).

    Returns:
        Contenu binaire du fichier WAV.
    """
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    if freq > 0:
        y = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    else:
        y = np.zeros_like(t, dtype=np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, y, SAMPLE_RATE, format="WAV")
    return buffer.getvalue()

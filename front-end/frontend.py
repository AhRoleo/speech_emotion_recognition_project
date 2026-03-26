# ============================================================
# 1) IMPORTS & CONFIGURATION DE LA PAGE
# ============================================================

import streamlit as st
import requests
import time
from datetime import datetime

# Configuration générale de la page Streamlit
st.set_page_config(
    page_title="Détecteur d'Émotions Vocales",
    page_icon="🎙️",
    layout="centered"
)

# URL de l'API FastAPI (modifiable dans la sidebar)
API_URL = st.sidebar.text_input(
    "URL de l'API",
    value="http://localhost:8000",
    key="api_url"
)


# ============================================================
# 2) CSS PERSONNALISÉ (design, animations, styles)
# ============================================================

st.markdown(
    """
    <style>
    :root{
        --accent:#6EE7B7;
        --card-bg: rgba(255,255,255,0.06);
        --glass: rgba(255,255,255,0.04);
    }
    .main-bg{
        background: linear-gradient(135deg,#071d2c 0%, #0b2940 50%, #122a3b 100%);
        padding: 18px;
        border-radius: 12px;
    }
    .result-card{
        background: var(--card-bg);
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.45);
        color: white;
    }
    .big-emoji{
        font-size: 96px;
        line-height: 1;
        animation: pulse 1.6s ease-in-out infinite;
        display:block;
        text-align:center;
    }
    @keyframes pulse {
      0% { transform: scale(1); }
      50% { transform: scale(1.08); }
      100% { transform: scale(1); }
    }
    .mini-bar {
        height: 10px;
        background: rgba(255,255,255,0.1);
        border-radius: 999px;
        overflow: hidden;
    }
    .mini-bar > .fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--accent), #34a0a4);
        transition: width 0.7s ease;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3) SESSION STATE (mémoire de session)
# ============================================================

# Historique des prédictions
if "history" not in st.session_state:
    st.session_state.history = []

# Dernier fichier analysé
if "last_file_name" not in st.session_state:
    st.session_state.last_file_name = None

# Liste des modèles disponibles
if "available_models" not in st.session_state:
    st.session_state.available_models = []


# ============================================================
# 4) FONCTIONS POUR COMMUNIQUER AVEC L'API
# ============================================================

def fetch_models():
    """Récupère la liste des modèles disponibles depuis l'API."""
    try:
        resp = requests.get(f"{API_URL}/models", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("models", [])
    except:
        return []
    return []


def predict_audio(audio_bytes: bytes, model_name: str):
    """Envoie un fichier audio à l'API pour obtenir une prédiction."""
    try:
        files = {"audio": ("audio.wav", audio_bytes, "audio/wav")}
        data = {"model_name": model_name}
        resp = requests.post(f"{API_URL}/predict", files=files, data=data, timeout=30)

        if resp.status_code == 200:
            return resp.json()

        st.error(f"❌ Erreur API ({resp.status_code}) : {resp.json().get('detail')}")
        return None

    except requests.exceptions.ConnectionError:
        st.error("❌ Impossible de se connecter à l'API.")
        return None


def check_api_health():
    """Vérifie si l'API FastAPI est en ligne."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        return resp.status_code == 200
    except:
        return False


# ============================================================
# 5) SIDEBAR : INFO + STATUT API + SÉLECTION DU MODÈLE
# ============================================================

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2920/2920349.png", width=80)
    st.title("ℹ️ À propos")
    st.write("Détecteur d'émotions vocales — interface interactive")

    # Vérification API
    api_online = check_api_health()
    if api_online:
        st.success("🟢 API en ligne")
    else:
        st.error("🔴 API hors ligne")

    st.divider()
    st.subheader("🤖 Sélection du modèle")

    # Rafraîchir les modèles
    if st.button("🔄 Rafraîchir les modèles"):
        st.session_state.available_models = fetch_models()

    # Charger automatiquement si API OK
    if not st.session_state.available_models and api_online:
        st.session_state.available_models = fetch_models()

    models = st.session_state.available_models

    if models:
        model_options = {m["name"]: m["file_name"] for m in models}
        selected_model_name = st.selectbox("Modèle", list(model_options.keys()))
        selected_model_file = model_options[selected_model_name]
    else:
        st.warning("Aucun modèle disponible")
        selected_model_file = None

    st.caption("Astuce : charge un fichier audio puis clique sur Analyser.")


# ============================================================
# 6) INTERFACE PRINCIPALE : UPLOAD + ANALYSE
# ============================================================

st.markdown('<div class="main-bg">', unsafe_allow_html=True)
st.title("🎙️ Détecteur d'Émotions Vocales")
st.write("Charge un fichier audio (WAV/MP3), choisis un modèle et lance l'analyse.")

uploaded_file = st.file_uploader("⬆️ Upload audio", type=["wav", "mp3"])

# Bouton Analyser
can_analyze = uploaded_file and selected_model_file and api_online
analyze_btn = st.button("🔍 Analyser", disabled=not can_analyze)


# ============================================================
# 7) TRAITEMENT DE LA PRÉDICTION
# ============================================================

if analyze_btn:
    audio_bytes = uploaded_file.getvalue()

    with st.spinner("✨ Analyse en cours..."):
        result = predict_audio(audio_bytes, selected_model_file)

    if result:
        main_emotion = result["main_emotion"]
        main_label = result["main_label_fr"]
        main_emoji = result["main_emoji"]
        main_prob = result["confidence"]

        # Historique
        st.session_state.history.insert(0, {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file": uploaded_file.name,
            "main": main_label,
            "emoji": main_emoji,
            "confidence": int(main_prob * 100),
            "model": selected_model_file,
        })
        st.session_state.history = st.session_state.history[:10]


        # ============================================================
        # 8) AFFICHAGE DU RÉSULTAT PRINCIPAL
        # ============================================================

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="big-emoji">{main_emoji}</div>', unsafe_allow_html=True)
        st.markdown(f"### <center>{main_label}</center>", unsafe_allow_html=True)
        st.metric("Indice de confiance", f"{int(main_prob * 100)}%")
        st.progress(main_prob)
        st.markdown("</div>", unsafe_allow_html=True)


        # ============================================================
        # 9) AFFICHAGE DES AUTRES ÉMOTIONS
        # ============================================================

        if st.checkbox("Voir toutes les émotions"):
            for ep in result["probabilities"]:
                if ep["emotion"] == main_emotion:
                    continue
                pct = int(ep["probability"] * 100)
                st.markdown(
                    f"""
                    <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                        <div style="width:36px;text-align:center;font-size:20px;">{ep["emoji"]}</div>
                        <div style="flex:1;">
                            <div style="display:flex;justify-content:space-between;">
                                <div style="font-weight:600;color:white;">{ep["label_fr"]}</div>
                                <div style="color: rgba(255,255,255,0.6);font-size:0.85rem;">{pct}%</div>
                            </div>
                            <div class="mini-bar">
                                <div class="fill" style="width:{pct}%;"></div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.audio(uploaded_file)


# ============================================================
# 10) HISTORIQUE DES ANALYSES
# ============================================================

if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📚 Historique des dernières analyses")
    for item in st.session_state.history[:5]:
        st.markdown(
            f"- **{item['time']}** — `{item['file']}` — "
            f"{item['emoji']} **{item['main']}** — {item['confidence']}% "
            f"[{item['model']}]"
        )

st.markdown("</div>", unsafe_allow_html=True)

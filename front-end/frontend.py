import streamlit as st
import requests
import time
from datetime import datetime

# --- CONFIG PAGE ---
st.set_page_config(page_title="Détecteur d'Émotions Vocales", page_icon="🎙️", layout="centered")

# --- Configuration API ---
API_URL = st.sidebar.text_input("URL de l'API", value="http://localhost:8000", key="api_url")

# --- CSS pour l'esthétique et animations ---
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
      0% { transform: scale(1); filter: drop-shadow(0 6px 18px rgba(0,0,0,0.6)); }
      50% { transform: scale(1.08); filter: drop-shadow(0 12px 30px rgba(0,0,0,0.7)); }
      100% { transform: scale(1); }
    }
    .mini-bar {
        height: 10px;
        background: linear-gradient(90deg, rgba(255,255,255,0.18), rgba(255,255,255,0.05));
        border-radius: 999px;
        overflow: hidden;
    }
    .mini-bar > .fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--accent), #34a0a4);
        transition: width 0.7s ease;
    }
    .emoji-btn {
        font-size: 22px;
        padding: 8px 12px;
        border-radius: 10px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.03);
        color: white;
        cursor: pointer;
    }
    .small-muted { color: rgba(255,255,255,0.6); font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session state ---
if "history" not in st.session_state:
    st.session_state.history = []
if "last_file_name" not in st.session_state:
    st.session_state.last_file_name = None
if "available_models" not in st.session_state:
    st.session_state.available_models = []


# --- Fonctions API ---
def fetch_models():
    """Récupère la liste des modèles depuis l'API."""
    try:
        resp = requests.get(f"{API_URL}/models", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("models", [])
    except requests.exceptions.ConnectionError:
        pass
    return []


def predict_audio(audio_bytes: bytes, model_name: str):
    """Envoie un fichier audio à l'API pour prédiction."""
    try:
        files = {"audio": ("audio.wav", audio_bytes, "audio/wav")}
        data = {"model_name": model_name}
        resp = requests.post(f"{API_URL}/predict", files=files, data=data, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        else:
            detail = resp.json().get("detail", "Erreur inconnue")
            st.error(f"❌ Erreur API ({resp.status_code}) : {detail}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Impossible de se connecter à l'API. Vérifiez qu'elle est lancée.")
        return None
    except Exception as e:
        st.error(f"❌ Erreur : {e}")
        return None


def check_api_health():
    """Vérifie si l'API est en ligne."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2920/2920349.png", width=80)
    st.title("ℹ️ À propos")
    st.write("Détecteur d'émotions vocales — version interactive")

    # Statut API
    api_online = check_api_health()
    if api_online:
        st.success("🟢 API en ligne")
    else:
        st.error("🔴 API hors ligne")

    # Sélecteur de modèle
    st.divider()
    st.subheader("🤖 Sélection du modèle")

    if st.button("🔄 Rafraîchir les modèles"):
        st.session_state.available_models = fetch_models()

    if not st.session_state.available_models and api_online:
        st.session_state.available_models = fetch_models()

    models = st.session_state.available_models
    if models:
        model_options = {m["name"]: m["file_name"] for m in models}
        selected_model_name = st.selectbox(
            "Modèle", list(model_options.keys()), key="model_selector"
        )
        selected_model_file = model_options[selected_model_name]
    else:
        st.warning("Aucun modèle disponible")
        selected_model_file = None

    st.divider()
    st.caption("Astuce: charge un fichier audio puis clique sur Analyser.")

# --- MAIN UI ---
st.markdown('<div class="main-bg">', unsafe_allow_html=True)
st.title("🎙️ Détecteur d'Émotions Vocales")
st.markdown(
    "Charge un enregistrement (WAV / MP3), sélectionne un modèle, "
    "clique **Analyser**, puis explore le résultat."
)

uploaded_file = st.file_uploader("⬆️ Upload audio (wav / mp3)", type=["wav", "mp3"])

# Boutons d'analyse
col_upload_left, col_upload_right = st.columns([2, 1])
with col_upload_right:
    can_analyze = (
        uploaded_file is not None and selected_model_file is not None and api_online
    )
    analyze_btn = st.button("� Analyser", use_container_width=True, disabled=not can_analyze)

if analyze_btn and uploaded_file and selected_model_file:
    fname = uploaded_file.name
    st.session_state.last_file_name = fname

    with st.spinner("✨ Extraction de features et inférence en cours..."):
        audio_bytes = uploaded_file.getvalue()
        result = predict_audio(audio_bytes, selected_model_file)

    if result:
        main_emotion = result["main_emotion"]
        main_label = result["main_label_fr"]
        main_emoji = result["main_emoji"]
        main_prob = result["confidence"]

        # Historique
        st.session_state.history.insert(0, {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file": fname,
            "main": main_label,
            "emoji": main_emoji,
            "confidence": round(main_prob * 100),
            "model": selected_model_file,
        })
        st.session_state.history = st.session_state.history[:10]

        # --- AFFICHAGE RÉSULTAT PRINCIPAL ---
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="big-emoji">{main_emoji}</div>', unsafe_allow_html=True)
        st.markdown(f"### <center>{main_label}</center>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align:center' class='small-muted'>Confiance du modèle</p>",
            unsafe_allow_html=True,
        )
        st.metric(label="Indice de confiance", value=f"{int(main_prob * 100)}%")
        st.progress(main_prob)
        st.markdown("</div>", unsafe_allow_html=True)

        # --- AFFICHAGE DES AUTRES ÉMOTIONS ---
        show_others = st.checkbox(
            "Voir toutes les émotions (probabilités)", value=False
        )
        if show_others:
            st.markdown("### Toutes les émotions")
            for ep in result["probabilities"]:
                if ep["emotion"] == main_emotion:
                    continue
                width_pct = int(ep["probability"] * 100)
                st.markdown(
                    f"""
                    <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                        <div style="width:36px;text-align:center;font-size:20px;">{ep["emoji"]}</div>
                        <div style="flex:1;">
                            <div style="display:flex;justify-content:space-between;">
                                <div style="font-weight:600;color:white;">{ep["label_fr"]}</div>
                                <div style="color: rgba(255,255,255,0.6);font-size:0.85rem;">{width_pct}%</div>
                            </div>
                            <div class="mini-bar" aria-hidden="true">
                                <div class="fill" style="width:{width_pct}%;"></div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Audio player
        if uploaded_file is not None:
            st.audio(uploaded_file)

# --- HISTORIQUE ---
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📚 Historique des dernières analyses (session)")
    for item in st.session_state.history[:5]:
        model_tag = f" [{item.get('model', '?')}]" if 'model' in item else ""
        st.markdown(
            f"- **{item['time']}** — `{item['file']}` — "
            f"{item['emoji']} **{item['main']}** — {item['confidence']}%{model_tag}"
        )

st.markdown("</div>", unsafe_allow_html=True)
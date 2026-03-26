import streamlit as st
import time
import random
from datetime import datetime

# --- CONFIG PAGE ---
st.set_page_config(page_title="Détecteur d'Émotions Vocales", page_icon="🎙️", layout="centered")

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

# --- EMOTIONS (utilise la liste complète demandée) ---
EMOTIONS = {
    "anger": ("Colère", "😡"),
    "disgust": ("Dégoût", "🤢"),
    "fear": ("Peur", "😨"),
    "happiness": ("Joie", "😄"),
    "neutral": ("Neutre", "😐"),
    "sadness": ("Tristesse", "😢"),
    "surprise": ("Surprise", "😲")
}

# --- Helpers / simulation de modèle ---
def preprocess_audio(audio_file):
    # placeholder pour vraie extraction MFCC
    time.sleep(0.9)
    return "processed"

def predict_probs(processed_data):
    # simulateur de distribution probabiliste interne
    raw = {k: random.uniform(0.02, 1.0) for k in EMOTIONS.keys()}
    s = sum(raw.values())
    probs = {k: raw[k]/s for k in raw}
    # tri décroissant pour cohérence
    sorted_probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
    return sorted_probs

# --- Session state pour historique ---
if "history" not in st.session_state:
    st.session_state.history = []  # liste de dicts {time, filename, main, conf}

if "last_file_name" not in st.session_state:
    st.session_state.last_file_name = None

# --- SIDEBAR INFO ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2920/2920349.png", width=80)
    st.title("ℹ️ À propos")
    st.write("Détecteur d'émotions vocales — version interactive")
    st.markdown("**Émotions supportées :**")
    for k,(fr,emo) in EMOTIONS.items():
        st.markdown(f"- {emo} {fr}")
    st.divider()
    st.caption("Astuce: charge un fichier audio puis clique sur Analyser.")

# --- MAIN UI ---
st.markdown('<div class="main-bg">', unsafe_allow_html=True)
st.title("🎙️ Détecteur d'Émotions Vocales — Interactive")
st.markdown("Charge un enregistrement (WAV / MP3), clique **Analyser**, puis explore le résultat.")

uploaded_file = st.file_uploader("⬆️ Upload audio (wav / mp3)", type=["wav","mp3"])

# Boutons d'analyse
col_upload_left, col_upload_right = st.columns([2,1])
with col_upload_right:
    analyze_btn = st.button("🚀 Analyser", use_container_width=True, disabled=(uploaded_file is None and st.session_state.last_file_name is None))
    reanalyze_btn = st.button("🔁 Réinitialiser / Nouveau", use_container_width=True)

if reanalyze_btn:
    # clear last upload context
    st.session_state.last_file_name = None
    st.experimental_rerun()

if analyze_btn:
    # affecter un nom de fichier pour l'historique
    fname = uploaded_file.name if uploaded_file is not None else st.session_state.last_file_name or f"audio_{random.randint(1000,9999)}.wav"
    st.session_state.last_file_name = fname

    with st.spinner("✨ Prétraitement et inférence en cours..."):
        processed = preprocess_audio(uploaded_file)
        probs = predict_probs(processed)

    # extraire émotion principale et confiance
    main_key = max(probs, key=probs.get)
    main_prob = probs[main_key]
    main_label, main_emoji = EMOTIONS[main_key]

    # stocker dans l'historique (garder seulement 10 derniers)
    st.session_state.history.insert(0, {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file": fname,
        "main": main_label,
        "emoji": main_emoji,
        "confidence": round(main_prob*100)
    })
    st.session_state.history = st.session_state.history[:10]

    # --- AFFICHAGE RÉSULTAT PRINCIPAL ---
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="big-emoji">{main_emoji}</div>', unsafe_allow_html=True)
    st.markdown(f"### <center>{main_label}</center>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center' class='small-muted'>Confiance du modèle</p>", unsafe_allow_html=True)
    # Affiche la seule valeur chiffrée
    st.metric(label="Indice de confiance", value=f"{int(main_prob*100)}%")
    # Jauge visuelle (progress) uniquement pour la principale
    st.progress(main_prob)
    st.markdown("</div>", unsafe_allow_html=True)

    # Boutons d'options (réanalyser même fichier)
    opt_col1, opt_col2 = st.columns([1,2])
    with opt_col1:
        if st.button("🔍 Réanalyser (même fichier)"):
            # simple re-run de l'analyse: simule nouvel inference
            processed = preprocess_audio(uploaded_file)
            probs = predict_probs(processed)
            # update and rerun to show new result
            st.experimental_rerun()
    with opt_col2:
        st.write("")  # placeholder for alignment

    # --- AFFICHAGE DES AUTRES ÉMOTIONS (sans pourcentages) ---
    show_others = st.checkbox("Voir les autres émotions (visualisation sans pourcentages)", value=False)
    if show_others:
        st.markdown("### Autres émotions (visualisation relative)")
        # construire une simple barre pour chaque émotion (aucun pourcentage affiché)
        for k, (fr, emoji) in EMOTIONS.items():
            if k == main_key:
                continue
            width_pct = int(probs[k] * 100)  # utilisé uniquement pour la largeur visuelle
            # barre stylée via HTML/CSS, sans nombre
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                    <div style="width:36px;text-align:center;font-size:20px;">{emoji}</div>
                    <div style="flex:1;">
                        <div style="display:flex;justify-content:space-between;">
                            <div style="font-weight:600;color:white;">{fr}</div>
                            <div style="color: rgba(255,255,255,0.45);font-size:0.85rem;">&nbsp;</div>
                        </div>
                        <div class="mini-bar" aria-hidden="true">
                            <div class="fill" style="width:{width_pct}%;"></div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Afficher audio player pour rejouer l'enregistrement s'il y a un fichier
    if uploaded_file is not None:
        st.audio(uploaded_file)

# --- HISTORIQUE SIMPLE ---
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📚 Historique des dernières analyses (session)")
    for item in st.session_state.history[:3]:
        st.markdown(f"- **{item['time']}** — `{item['file']}` — {item['emoji']} **{item['main']}** — {item['confidence']}%")

st.markdown("</div>", unsafe_allow_html=True)
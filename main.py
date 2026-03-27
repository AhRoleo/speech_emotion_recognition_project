import subprocess
import sys
import time
from pathlib import Path

# Couleurs pour le terminal
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


def lancer_services():
    """Lance l'API FastAPI et le Front-End Streamlit en parallèle."""

    print(f"{BOLD} Démarrage de l'application SER (Speech Emotion Recognition)...{RESET}\n")
    # 1. Obtenir les chemins absolus (peu importe d'où le script est lancé)
    racine_projet = Path(__file__).parent.absolute()

    # 2. Lancement du Backend (API FastAPI)
    print(f"[{BLUE}BACKEND{RESET}] Lancement de l'API FastAPI sur le port 8000...")
    process_api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=racine_projet
    )

    # On attend 3 secondes pour laisser le temps à l'API de charger les modèles (.pkl et .keras)
    time.sleep(3)

    # 3. Lancement du Frontend (Streamlit)
    print(f"[{GREEN}FRONTEND{RESET}] Lancement de l'interface Streamlit sur le port 8501...")
    process_front = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "front-end/frontend.py", "--server.port", "8501"],
        cwd=racine_projet
    )

    print(f"\n{BOLD} Tout est lancé !{RESET}")
    print(f"    Interface Utilisateur (Frontend) : {GREEN}http://localhost:8501{RESET}")
    print(f"    Documentation API (Swagger)    : {BLUE}http://localhost:8000/docs{RESET}")
    print(f"\n{BOLD}(Appuyez sur Ctrl+C dans ce terminal pour tout fermer proprement.){RESET}\n")

    try:
        # Garde le script principal ouvert pendant que les deux sous-processus tournent
        process_api.wait()
        process_front.wait()
    except KeyboardInterrupt:
        # Si vous faites Ctrl+C, on ferme proprement l'API et Streamlit
        print(f"\n{BOLD} Arrêt des services demandé...{RESET}")
        process_api.terminate()
        process_front.terminate()
        print("Toutes les connexions ont été fermées avec succès. Bye ! 👋")


if __name__ == "__main__":
    lancer_services()

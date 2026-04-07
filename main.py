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

    print(f"{BOLD} Démarrage de l'application SER (Speech Emotion Recognition)...{RESET}\n")

    racine_projet = Path(__file__).parent.absolute()


    print(f"[{BLUE}BACKEND{RESET}] Lancement de l'API FastAPI sur le port 8000...")
    process_api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=racine_projet
    )

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

        process_api.wait()
        process_front.wait()
    except KeyboardInterrupt:
        print(f"\n{BOLD} Arrêt des services demandé...{RESET}")
        process_api.terminate()
        process_front.terminate()
        print("Toutes les connexions ont été fermées avec succès. Bye ! 👋")


if __name__ == "__main__":
    lancer_services()

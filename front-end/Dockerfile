# ── C1 — Frontend Streamlit ───────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

COPY front-end/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY front-end/frontend.py ./front-end/frontend.py

EXPOSE 8501

CMD ["streamlit", "run", "front-end/frontend.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]

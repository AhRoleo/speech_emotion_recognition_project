import os
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import traceback

from processing.process_data import run_processing

app = FastAPI()

# Statut global du traitement
# Possible states: idle, running, completed, failed
processing_status = {
    "state": "idle",
    "last_error": None,
}

class ProcessResponse(BaseModel):
    message: str
    state: str

DATASET_DIR = os.environ.get("DATASET_DIR", "Dataset/")
PROCESSED_DATA_DIR = DATASET_DIR + "processed"

REQUIRED_FILES = [
    "train_features.parquet",
    "test_features.parquet",
]

def processed_data_ready() -> bool:
    processed_dir = Path(PROCESSED_DATA_DIR)
    if not processed_dir.exists():
        return False
    return all((processed_dir / filename).exists() for filename in REQUIRED_FILES)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "processing",
        "state": processing_status["state"],
    }

@app.get("/ready")
def ready():
    if not processed_data_ready():
        raise HTTPException(
            status_code=503,
            detail="Processed data not ready"
        )
    return {
        "status": "ready",
        "processed_data_ready": True,
    }

@app.get("/status")
def status():
    return processing_status

def _process_job():
    global processing_status
    try:
        processing_status["state"] = "running"
        processing_status["last_error"] = None
        run_processing()
        processing_status["state"] = "completed"
    except Exception:
        processing_status["state"] = "failed"
        processing_status["last_error"] = traceback.format_exc()

@app.post("/process", response_model=ProcessResponse)
def process(background_tasks: BackgroundTasks):
    if processing_status["state"] == "running":
        return ProcessResponse(
            message="Processing already running",
            state="running",
        )

    background_tasks.add_task(_process_job)

    return ProcessResponse(
        message="Processing started",
        state="running",
    )
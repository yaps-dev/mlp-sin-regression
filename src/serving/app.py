import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
import asyncio
from src.serving import drift_monitor
from src.serving.drift_monitor import drift_monitor_loop, load_reference
from src.serving import metrics  # noqa: F401

# Konfiguracja
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sine-sidecar")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "../../artifacts/sine_model_v1.joblib"))
MODEL_METADATA_PATH = Path(os.getenv("MODEL_METADATA_PATH", "../../artifacts/sine_model_v1_metadata.json"))

# Pydantic schemas (walidacja request/response)
class PredictRequest(BaseModel):
    x: float = Field(..., description="Wartosc wejsciowa dla y = A*sin(B*x)")

class BatchPredictRequest(BaseModel):
    x: list[float] = Field(..., min_length=1, description="Lista wartosci x")

class Prediction(BaseModel):
    x: float
    y: float

class PredictResponse(BaseModel):
    prediction: Prediction

class BatchPredictResponse(BaseModel):
    predictions: list[Prediction]
    count: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# Stan aplikacji - prosty kontener na zaladowany model
class ModelState:
    model = None
    metadata = None

state = ModelState()

# Lifespan: model wczytujemy raz, przy starcie aplikacji
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Wczytuje model z: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        logger.error(f"Plik modelu nie istnieje: {MODEL_PATH}")
        raise FileNotFoundError(f"Brak pliku modelu: {MODEL_PATH}")

    logger.info(f"Wczytuje metadane z: {MODEL_METADATA_PATH}")
    if not MODEL_METADATA_PATH.exists():
        logger.error(f"Plik z metadanymi modelu nie istnieje: {MODEL_METADATA_PATH}")
        raise FileNotFoundError(f"Brak pliku z metadanymi modelu: {MODEL_METADATA_PATH}")

    state.model = joblib.load(MODEL_PATH)
    state.metadata = json.loads(MODEL_METADATA_PATH.read_text())

    drift_monitor.REFERENCE_DATA = load_reference()

    # włączenie zadania monitoringu
    task = asyncio.create_task(drift_monitor_loop())

    yield

    # Gracefull shutdown
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # cleanup po SIGTERM
    logger.info("Zwalniam model")
    state.model = None
    state.metadata = None

# Aplikacja FastAPI
app = FastAPI(
    title="Sine Model Sidecar",
    description="Inferencja modelu MLPRegressor (y = A*sin(B*x))",
    version="1.0.0",
    lifespan=lifespan,
)
Instrumentator().instrument(app).expose(app)

@app.get("/metadata")
async def getMetadata():
    return state.metadata

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Predykcja dla pojedynczego x."""
    if state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model niezaladowany",
        )

    X = np.array([[req.x]])
    y_pred = float(state.model.predict(X)[0])

    logger.info(f"predict: x={req.x:.4f} -> y={y_pred:.4f}")

    return PredictResponse(
        prediction=Prediction(x=req.x, y=y_pred)
    )

@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    """Predykcja dla listy x - efektywniejsza niz n*POST /predict."""
    if state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model niezaladowany",
        )

    X = np.array(req.x).reshape(-1, 1)
    y_pred = state.model.predict(X)

    logger.info(f"predict_batch: n={len(req.x)}")

    return BatchPredictResponse(
        predictions=[
            Prediction(x=float(x), y=float(y))
            for x, y in zip(req.x, y_pred)
        ],
        count=len(req.x)
    )
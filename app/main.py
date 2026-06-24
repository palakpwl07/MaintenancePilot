from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import time

from app.schemas import SensorReading, PredictionResponse, HealthResponse
from app.model import model_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    print("Loading model from MLflow registry...")
    model_manager.load()
    yield
    print("Shutting down.")

app = FastAPI(
    title="MaintenancePilot",
    description=(
        "Predictive maintenance API for industrial compressors. "
        "Predicts failure risk 48 hours in advance using sensor data "
        "from the Air Production Unit. "
        "Trained on MetroPT-3 dataset — real operational data from a Portuguese metro system."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """System health check — confirms model is loaded and returns model metadata."""
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status":         "healthy",
        "model_loaded":   True,
        "model_version":  model_manager.config["model_version"],
        "model_type":     model_manager.config["model_type"],
        "threshold":      model_manager.config["threshold"],
        "cv_mean_ap":     model_manager.config["cv_mean_ap"],
        "cv_mean_recall": model_manager.config["cv_mean_recall"],
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: SensorReading):
    """
    Predict compressor failure risk for a single sensor window.

    Input: 41 engineered features (raw sensors + rolling statistics).
    Output: failure risk score, HIGH_RISK/NORMAL prediction, and interpretation.

    Threshold is selected to minimise business cost where FN cost = 10x FP cost.
    A missed failure costs significantly more than an unnecessary inspection.
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(request.features) != 41:
        raise HTTPException(
            status_code=422,
            detail=f"Expected 41 features, got {len(request.features)}"
        )

    try:
        result = model_manager.predict(request.features, request.threshold_override)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info", tags=["System"])
def model_info():
    """Returns full model metadata including CV performance and feature list."""
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_name":        model_manager.config["model_name"],
        "model_version":     model_manager.config["model_version"],
        "model_type":        model_manager.config["model_type"],
        "threshold":         model_manager.config["threshold"],
        "fn_cost_multiplier": model_manager.config["fn_cost"],
        "cv_mean_ap":        model_manager.config["cv_mean_ap"],
        "cv_mean_recall":    model_manager.config["cv_mean_recall"],
        "prediction_horizon": "48 hours",
        "feature_count":     len(model_manager.config["feature_cols"]),
        "features":          model_manager.config["feature_cols"],
        "training_data":     "MetroPT-3 — Portuguese metro APU compressor, Feb-Aug 2020",
        "failure_events":    4,
    }
from pydantic import BaseModel, Field
from typing import List

class SensorReading(BaseModel):
    """Single prediction request — one window of sensor readings."""
    features: List[float] = Field(
        ...,
        description="41 engineered features in order: raw sensors + rolling statistics",
        min_items=41,
        max_items=41
    )
    threshold_override: float | None = Field(
        default=None,
        description="Optional threshold override. Uses business-cost threshold if not provided.",
        ge=0.0,
        le=1.0
    )

class PredictionResponse(BaseModel):
    failure_risk_score: float
    prediction:         str
    threshold_used:     float
    model_version:      int
    model_type:         str
    interpretation:     str

class HealthResponse(BaseModel):
    status:         str
    model_loaded:   bool
    model_version:  int
    model_type:     str
    threshold:      float
    cv_mean_ap:     float
    cv_mean_recall: float
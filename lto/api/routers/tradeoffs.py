"""Tradeoff prediction and query endpoints."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lto.api.deps import get_model
from lto.schemas import JobParameters

router = APIRouter()


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request body for tradeoff prediction."""
    na: float = Field(..., ge=0.1, le=0.55)
    wavelength_nm: float = Field(13.5)
    dose_mj_cm2: float = Field(..., ge=1.0, le=100.0)
    sigma: float = Field(0.8, ge=0.1, le=1.0)
    resist_thickness_nm: float = Field(30.0, ge=5.0, le=200.0)
    grid_size_nm: float = Field(1.0, ge=0.1, le=10.0)
    use_ai_surrogate: bool = True
    job_class: int = Field(1, ge=1, le=20)


class ScoreResponse(BaseModel):
    score: float
    ci_low: float
    ci_high: float


class PredictResponse(BaseModel):
    job_id: str
    parameters: Dict[str, Any]
    predictions: Dict[str, ScoreResponse]
    uncertainty: Dict[str, Any]
    constraints: Dict[str, Any]
    feature_importance: Dict[str, float]
    model_version: str
    inference_time_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/predict", response_model=PredictResponse)
async def predict_tradeoffs(request: PredictRequest):
    """Predict tradeoffs for given parameters (no simulation).

    This runs the ML model to predict tradeoff health scores for a
    set of lithography parameters, without actually running a simulation.
    """
    model = get_model()

    # Convert request to JobParameters
    params = JobParameters(
        na=request.na,
        wavelength_nm=request.wavelength_nm,
        dose_mj_cm2=request.dose_mj_cm2,
        sigma=request.sigma,
        resist_thickness_nm=request.resist_thickness_nm,
        grid_size_nm=request.grid_size_nm,
        use_ai_surrogate=request.use_ai_surrogate,
        job_class=request.job_class,
    )

    try:
        prediction = model.predict(params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return PredictResponse(
        job_id=prediction.job_id,
        parameters={
            "na": params.na,
            "wavelength_nm": params.wavelength_nm,
            "dose_mj_cm2": params.dose_mj_cm2,
            "sigma": params.sigma,
            "resist_thickness_nm": params.resist_thickness_nm,
            "grid_size_nm": params.grid_size_nm,
            "use_ai_surrogate": params.use_ai_surrogate,
        },
        predictions={
            name: ScoreResponse(
                score=sci.score,
                ci_low=sci.ci_low,
                ci_high=sci.ci_high,
            )
            for name, sci in prediction.predictions.items()
        },
        uncertainty={
            "high_uncertainty": prediction.uncertainty.high_uncertainty,
            "recommend_physics_simulation": prediction.uncertainty.recommend_physics_simulation,
            "confidence_level": prediction.uncertainty.confidence_level.value,
        },
        constraints={
            "all_satisfied": prediction.constraints.all_satisfied,
            "violations": prediction.constraints.violations,
        },
        feature_importance=prediction.feature_importance,
        model_version=prediction.model_version,
        inference_time_ms=prediction.inference_time_ms,
    )


@router.get("/history")
async def tradeoff_history():
    """Query historical tradeoff data (placeholder for V0.2)."""
    return {
        "message": "Historical tradeoff queries coming in V0.2",
        "data": [],
    }

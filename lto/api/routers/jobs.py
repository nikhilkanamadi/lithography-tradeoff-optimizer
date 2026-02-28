"""Job submission and management endpoints."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lto.api.deps import get_model
from lto.schemas import JobParameters
from lto.simulator.synthetic import SyntheticSimulator

router = APIRouter()

# Module-level simulator instance
_simulator = SyntheticSimulator(seed=None)


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class SubmitJobRequest(BaseModel):
    """Request body for job submission."""
    na: float = Field(..., ge=0.1, le=0.55)
    wavelength_nm: float = Field(13.5)
    dose_mj_cm2: float = Field(..., ge=1.0, le=100.0)
    sigma: float = Field(0.8, ge=0.1, le=1.0)
    resist_thickness_nm: float = Field(30.0, ge=5.0, le=200.0)
    grid_size_nm: float = Field(1.0, ge=0.1, le=10.0)
    use_ai_surrogate: bool = True
    pattern_complexity: str = "moderate"
    job_class: int = Field(1, ge=1, le=20)
    submitted_by: str = "api_user"


class JobResultResponse(BaseModel):
    job_id: str
    status: str
    parameters: Dict[str, Any]
    outputs: Dict[str, Any]
    tradeoff_signals: Dict[str, float]
    ml_prediction: Dict[str, Any]
    compute_stats: Dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/submit", response_model=JobResultResponse)
async def submit_job(request: SubmitJobRequest):
    """Submit a simulation job with pre-flight tradeoff prediction.

    This endpoint:
    1. Validates parameters
    2. Runs pre-flight ML tradeoff prediction
    3. Executes the simulation
    4. Returns results + tradeoff report
    """
    model = get_model()

    # Build job parameters
    params = JobParameters(
        na=request.na,
        wavelength_nm=request.wavelength_nm,
        dose_mj_cm2=request.dose_mj_cm2,
        sigma=request.sigma,
        resist_thickness_nm=request.resist_thickness_nm,
        grid_size_nm=request.grid_size_nm,
        use_ai_surrogate=request.use_ai_surrogate,
        pattern_complexity=request.pattern_complexity,
        job_class=request.job_class,
        submitted_by=request.submitted_by,
    )

    # Pre-flight prediction
    prediction = model.predict(params)

    # Run simulation
    try:
        result = _simulator.run_job(params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

    return JobResultResponse(
        job_id=result.job_id,
        status="completed",
        parameters={
            "na": params.na,
            "wavelength_nm": params.wavelength_nm,
            "dose_mj_cm2": params.dose_mj_cm2,
            "sigma": params.sigma,
            "resist_thickness_nm": params.resist_thickness_nm,
            "grid_size_nm": params.grid_size_nm,
            "use_ai_surrogate": params.use_ai_surrogate,
            "pattern_complexity": params.pattern_complexity.value
                if hasattr(params.pattern_complexity, "value")
                else params.pattern_complexity,
        },
        outputs={
            "resolution_nm": result.outputs.resolution_nm,
            "depth_of_focus_nm": result.outputs.depth_of_focus_nm,
            "pattern_fidelity": result.outputs.pattern_fidelity,
            "compute_time_s": result.outputs.compute_time_s,
            "accuracy_vs_physics": result.outputs.accuracy_vs_physics,
            "yield_prediction": result.outputs.yield_prediction,
        },
        tradeoff_signals={
            "speed_vs_accuracy": result.tradeoff_signals.speed_vs_accuracy,
            "resolution_vs_dof": result.tradeoff_signals.resolution_vs_dof,
            "cost_vs_fidelity": result.tradeoff_signals.cost_vs_fidelity,
            "surrogate_reliability": result.tradeoff_signals.surrogate_reliability,
            "yield_risk": result.tradeoff_signals.yield_risk,
            "overall_health": result.tradeoff_signals.overall_health,
        },
        ml_prediction={
            "predictions": {
                name: {"score": sci.score, "ci_low": sci.ci_low, "ci_high": sci.ci_high}
                for name, sci in prediction.predictions.items()
            },
            "uncertainty": prediction.uncertainty.confidence_level.value,
            "constraints_satisfied": prediction.constraints.all_satisfied,
            "violations": prediction.constraints.violations,
        },
        compute_stats={
            "wall_time_s": result.compute_stats.wall_time_s,
            "cpu_time_s": result.compute_stats.cpu_time_s,
            "peak_memory_mb": result.compute_stats.peak_memory_mb,
        },
    )


@router.get("/{job_id}")
async def get_job(job_id: str):
    """Get job status and results (placeholder for V0.2 with DB integration)."""
    return {
        "message": f"Job lookup for {job_id} coming in V0.2 with database integration",
        "job_id": job_id,
    }

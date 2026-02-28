"""Job pipeline flow — the main orchestration path.

Implements the ARCHITECTURE.md state machine:
SUBMITTED → VALIDATING → PREFLIGHT → QUEUED → RUNNING → EVALUATING → COMPLETED

Each step is a Prefect task with retries, logging, and metrics emission.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

from lto.schemas import (
    AlertSeverity,
    JobParameters,
    JobStatus,
    SimulationResult,
    TradeoffPrediction,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@task(name="validate_parameters", retries=1)
def validate_parameters(params: JobParameters) -> JobParameters:
    """Validate and enrich job parameters."""
    logger.info(f"[{params.job_id}] Validating parameters...")

    # Pydantic already validates ranges, but we add domain-specific checks
    if params.wavelength_nm == 13.5 and params.na > 0.55:
        raise ValueError(f"NA={params.na} exceeds EUV limit of 0.55")

    if params.pattern_complexity.value == "extreme" and params.grid_size_nm > 2.0:
        logger.warning(
            f"[{params.job_id}] Grid size {params.grid_size_nm}nm too coarse "
            f"for extreme patterns — results may be inaccurate"
        )

    logger.info(f"[{params.job_id}] Validation passed ✓")
    return params


@task(name="preflight_prediction", retries=2)
def preflight_prediction(params: JobParameters) -> TradeoffPrediction:
    """Run pre-flight ML prediction to assess tradeoff health."""
    from lto.ml.engine import TradeoffModel

    logger.info(f"[{params.job_id}] Running pre-flight prediction...")

    # Try loading model, fall back to training
    model = _get_or_train_model()
    prediction = model.predict(params)

    logger.info(
        f"[{params.job_id}] Pre-flight: overall confidence={prediction.uncertainty.confidence_level.value}, "
        f"constraints_ok={prediction.constraints.all_satisfied}"
    )

    return prediction


@task(name="check_preflight_gate")
def check_preflight_gate(
    params: JobParameters,
    prediction: TradeoffPrediction,
) -> bool:
    """Gate: should we proceed with simulation?

    Blocks the job if:
    - yield_risk > 0.8 (CRITICAL)
    - All constraints violated
    - Model recommends blocking
    """
    if prediction.recommend_block:
        logger.warning(
            f"[{params.job_id}] ⚠️ Pre-flight BLOCKED: {prediction.block_reason}"
        )
        return False

    if prediction.uncertainty.recommend_physics_simulation and params.use_ai_surrogate:
        logger.info(
            f"[{params.job_id}] Uncertainty high — recommending physics simulation"
        )

    return True


@task(name="run_simulation", retries=1)
def run_simulation(params: JobParameters) -> SimulationResult:
    """Execute the simulation."""
    from lto.simulator.synthetic import SyntheticSimulator

    logger.info(f"[{params.job_id}] Running simulation...")
    sim = SyntheticSimulator()
    result = sim.run_job(params)
    logger.info(
        f"[{params.job_id}] Simulation complete: "
        f"resolution={result.outputs.resolution_nm}nm, "
        f"yield={result.outputs.yield_prediction:.3f}"
    )
    return result


@task(name="evaluate_results")
def evaluate_results(
    result: SimulationResult,
    prediction: TradeoffPrediction,
) -> Dict[str, Any]:
    """Compare simulation results with pre-flight prediction.

    Evaluates prediction accuracy and flags discrepancies.
    """
    report = {
        "job_id": result.job_id,
        "status": "completed",
        "simulation_health": result.tradeoff_signals.overall_health,
        "predicted_vs_actual": {},
        "discrepancies": [],
    }

    # Compare predicted vs actual tradeoff signals
    actual = {
        "speed_vs_accuracy": result.tradeoff_signals.speed_vs_accuracy,
        "resolution_vs_dof": result.tradeoff_signals.resolution_vs_dof,
        "cost_vs_fidelity": result.tradeoff_signals.cost_vs_fidelity,
        "surrogate_reliability": result.tradeoff_signals.surrogate_reliability,
        "yield_risk": result.tradeoff_signals.yield_risk,
    }

    for name, actual_val in actual.items():
        if name in prediction.predictions:
            predicted_val = prediction.predictions[name].score
            diff = abs(predicted_val - actual_val)
            report["predicted_vs_actual"][name] = {
                "predicted": predicted_val,
                "actual": actual_val,
                "diff": round(diff, 4),
            }
            if diff > 0.1:
                report["discrepancies"].append(
                    f"{name}: predicted={predicted_val:.3f}, actual={actual_val:.3f} (Δ={diff:.3f})"
                )

    if report["discrepancies"]:
        logger.warning(
            f"[{result.job_id}] Prediction discrepancies: {report['discrepancies']}"
        )

    return report


@task(name="check_alerts")
def check_alerts(result: SimulationResult, report: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Check if any alerts should be fired based on results."""
    alerts = []

    if result.tradeoff_signals.yield_risk > 0.7:
        alerts.append({
            "severity": AlertSeverity.CRITICAL.value,
            "type": "high_yield_risk",
            "message": f"Yield risk {result.tradeoff_signals.yield_risk:.3f} exceeds threshold",
            "job_id": result.job_id,
        })

    if result.tradeoff_signals.overall_health < 0.4:
        alerts.append({
            "severity": AlertSeverity.WARN.value,
            "type": "low_health",
            "message": f"Overall health {result.tradeoff_signals.overall_health:.3f} below threshold",
            "job_id": result.job_id,
        })

    if report.get("discrepancies"):
        alerts.append({
            "severity": AlertSeverity.INFO.value,
            "type": "prediction_discrepancy",
            "message": f"Prediction discrepancies detected: {len(report['discrepancies'])} targets",
            "job_id": result.job_id,
        })

    return alerts


# ---------------------------------------------------------------------------
# Main Flow
# ---------------------------------------------------------------------------

@flow(name="lto-job-pipeline", log_prints=True)
def job_pipeline(
    na: float,
    dose_mj_cm2: float,
    wavelength_nm: float = 13.5,
    sigma: float = 0.8,
    resist_thickness_nm: float = 30.0,
    grid_size_nm: float = 1.0,
    use_ai_surrogate: bool = True,
    pattern_complexity: str = "moderate",
    job_class: int = 1,
    submitted_by: str = "prefect",
) -> Dict[str, Any]:
    """Main job pipeline flow.

    Orchestrates: validate → preflight → gate → simulate → evaluate → alert
    """
    # Build parameters
    params = JobParameters(
        na=na,
        wavelength_nm=wavelength_nm,
        dose_mj_cm2=dose_mj_cm2,
        sigma=sigma,
        resist_thickness_nm=resist_thickness_nm,
        grid_size_nm=grid_size_nm,
        use_ai_surrogate=use_ai_surrogate,
        pattern_complexity=pattern_complexity,
        job_class=job_class,
        submitted_by=submitted_by,
    )

    # 1. Validate
    validated_params = validate_parameters(params)

    # 2. Pre-flight prediction
    prediction = preflight_prediction(validated_params)

    # 3. Gate check
    proceed = check_preflight_gate(validated_params, prediction)

    if not proceed:
        return {
            "job_id": params.job_id,
            "status": "blocked",
            "reason": prediction.block_reason,
            "prediction": prediction.model_dump(),
        }

    # 4. Simulate
    result = run_simulation(validated_params)

    # 5. Evaluate
    report = evaluate_results(result, prediction)

    # 6. Check alerts
    alerts = check_alerts(result, report)

    # Create Prefect artifact
    create_markdown_artifact(
        key=f"job-report-{params.job_id}",
        markdown=_format_report_markdown(report, alerts),
        description=f"Job report for {params.job_id}",
    )

    return {
        "job_id": params.job_id,
        "status": "completed",
        "report": report,
        "alerts": alerts,
        "simulation": {
            "resolution_nm": result.outputs.resolution_nm,
            "depth_of_focus_nm": result.outputs.depth_of_focus_nm,
            "yield_prediction": result.outputs.yield_prediction,
            "overall_health": result.tradeoff_signals.overall_health,
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_cached_model = None


def _get_or_train_model():
    """Get or train the ML model (with caching)."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    from pathlib import Path

    from lto.ml.engine import TradeoffModel

    model_path = Path("models/tradeoff_v1.pkl")
    if model_path.exists():
        _cached_model = TradeoffModel.load(model_path)
    else:
        _cached_model = TradeoffModel(use_ensemble=False)
        _cached_model.train_from_simulator(n_samples=5000)
        _cached_model.save("tradeoff_v1")

    return _cached_model


def _format_report_markdown(report: Dict, alerts: list) -> str:
    """Format job report as markdown for Prefect artifact."""
    lines = [
        f"# Job Report: {report['job_id']}",
        f"",
        f"**Status:** {report['status']}",
        f"**Health:** {report.get('simulation_health', 'N/A')}",
        f"",
        f"## Predicted vs Actual",
    ]
    for name, vals in report.get("predicted_vs_actual", {}).items():
        lines.append(f"- **{name}**: predicted={vals['predicted']:.3f}, actual={vals['actual']:.3f}")

    if alerts:
        lines.append(f"\n## Alerts ({len(alerts)})")
        for a in alerts:
            lines.append(f"- [{a['severity']}] {a['message']}")

    return "\n".join(lines)

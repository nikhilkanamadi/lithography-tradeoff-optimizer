"""Pydantic data contracts for the LTO platform.

These schemas define the interface between ALL components.
No component should use raw dicts — any data crossing a boundary
uses one of these contracts.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PatternComplexity(str, enum.Enum):
    """Classification of mask pattern complexity."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXTREME = "extreme"


class JobPriority(str, enum.Enum):
    """Job scheduling priority."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class JobStatus(str, enum.Enum):
    """Job lifecycle states."""
    SUBMITTED = "submitted"
    VALIDATING = "validating"
    PREFLIGHT = "preflight"
    QUEUED = "queued"
    RUNNING = "running"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    REJECTED = "rejected"
    FAILED = "failed"
    ANOMALY_FLAGGED = "anomaly_flagged"
    ENGINEER_REVIEW = "engineer_review"
    CLEARED = "cleared"
    ESCALATED = "escalated"


class AlertSeverity(str, enum.Enum):
    """Alert severity levels."""
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ModelStatus(str, enum.Enum):
    """ML model lifecycle status."""
    TRAINING = "training"
    SHADOW = "shadow"
    PRODUCTION = "production"
    RETIRED = "retired"


class ConfidenceLevel(str, enum.Enum):
    """Uncertainty classification."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Input Schemas
# ---------------------------------------------------------------------------

class JobParameters(BaseModel):
    """Input parameters for a lithography simulation job.

    Validated ranges come from physical constraints of EUV/DUV lithography.
    """

    job_id: str = Field(
        default_factory=lambda: f"job_{uuid4().hex[:8]}",
        description="Unique job identifier",
    )
    na: float = Field(
        ..., ge=0.1, le=0.55,
        description="Numerical aperture — lens light-gathering ability",
    )
    wavelength_nm: float = Field(
        13.5,
        description="Illumination wavelength in nm. EUV=13.5, ArF=193, KrF=248",
    )
    dose_mj_cm2: float = Field(
        ..., ge=1.0, le=100.0,
        description="Exposure dose in mJ/cm²",
    )
    sigma: float = Field(
        0.8, ge=0.1, le=1.0,
        description="Partial coherence factor",
    )
    resist_thickness_nm: float = Field(
        30.0, ge=5.0, le=200.0,
        description="Photoresist layer thickness in nm",
    )
    grid_size_nm: float = Field(
        1.0, ge=0.1, le=10.0,
        description="Simulation grid resolution in nm",
    )
    pattern_complexity: PatternComplexity = PatternComplexity.MODERATE
    use_ai_surrogate: bool = True
    job_class: int = Field(
        1, ge=1, le=20,
        description="Job type classifier for routing and thresholds",
    )
    priority: JobPriority = JobPriority.NORMAL
    constraint_profile: str = "default"
    submitted_by: str = "system"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"json_schema_extra": {
        "examples": [{
            "na": 0.33,
            "wavelength_nm": 13.5,
            "dose_mj_cm2": 15.2,
            "sigma": 0.8,
            "grid_size_nm": 1.0,
            "use_ai_surrogate": True,
        }]
    }}


# ---------------------------------------------------------------------------
# Output Schemas
# ---------------------------------------------------------------------------

class SimulationOutputs(BaseModel):
    """Raw physical outputs from a simulation run."""

    resolution_nm: float = Field(..., description="Minimum feature resolution in nm")
    depth_of_focus_nm: float = Field(..., description="Depth of focus in nm")
    pattern_fidelity: float = Field(
        ..., ge=0.0, le=1.0,
        description="Pattern transfer accuracy score",
    )
    compute_time_s: float = Field(..., ge=0.0, description="Wall-clock simulation time")
    accuracy_vs_physics: float = Field(
        ..., ge=0.0, le=1.0,
        description="Accuracy compared to full physics simulation",
    )
    yield_prediction: float = Field(
        ..., ge=0.0, le=1.0,
        description="Predicted wafer yield",
    )


class TradeoffSignals(BaseModel):
    """Core tradeoff scores — the heart of LTO.

    Each score represents the health of a specific tradeoff dimension.
    0.0 = dangerous / bad tradeoff, 1.0 = excellent / optimal tradeoff.
    """

    speed_vs_accuracy: float = Field(..., ge=0.0, le=1.0)
    resolution_vs_dof: float = Field(..., ge=0.0, le=1.0)
    cost_vs_fidelity: float = Field(..., ge=0.0, le=1.0)
    surrogate_reliability: float = Field(..., ge=0.0, le=1.0)
    yield_risk: float = Field(
        ..., ge=0.0, le=1.0,
        description="0=no risk, 1=high risk (inverted from health)",
    )
    overall_health: float = Field(..., ge=0.0, le=1.0)


class ComputeStats(BaseModel):
    """Compute resource usage for a simulation job."""

    wall_time_s: float
    cpu_time_s: float
    peak_memory_mb: float
    gpu_utilization: Optional[float] = None


class SimulationResult(BaseModel):
    """Complete output of a simulation job — the master record."""

    job_id: str
    parameters: JobParameters
    outputs: SimulationOutputs
    tradeoff_signals: TradeoffSignals
    compute_stats: ComputeStats
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    simulator_version: str = "synthetic-v0.1"
    adapter_type: str = "synthetic"


# ---------------------------------------------------------------------------
# ML Prediction Schemas
# ---------------------------------------------------------------------------

class ScoreWithCI(BaseModel):
    """A predicted score with 95% confidence interval."""

    score: float = Field(..., ge=0.0, le=1.0)
    ci_low: float = Field(..., ge=0.0, le=1.0)
    ci_high: float = Field(..., ge=0.0, le=1.0)


class UncertaintyInfo(BaseModel):
    """Uncertainty quantification result."""

    high_uncertainty: bool = False
    recommend_physics_simulation: bool = False
    confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH
    reason: Optional[str] = None


class ConstraintResult(BaseModel):
    """Constraint evaluation result."""

    all_satisfied: bool = True
    violations: list[str] = Field(default_factory=list)


class TradeoffPrediction(BaseModel):
    """ML model output for a tradeoff prediction."""

    job_id: str
    predictions: Dict[str, ScoreWithCI]
    uncertainty: UncertaintyInfo
    constraints: ConstraintResult
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    model_version: str
    inference_time_ms: float
    recommend_block: bool = False
    block_reason: Optional[str] = None
    alternative_params: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Alert Schema
# ---------------------------------------------------------------------------

class AlertInfo(BaseModel):
    """An alert fired by the system."""

    alert_id: str = Field(default_factory=lambda: f"alert_{uuid4().hex[:8]}")
    fired_at: datetime = Field(default_factory=datetime.utcnow)
    severity: AlertSeverity
    job_id: Optional[str] = None
    alert_type: str
    message: str
    recommendation: Optional[str] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    was_actionable: Optional[bool] = None


# ---------------------------------------------------------------------------
# API Response Schemas
# ---------------------------------------------------------------------------

class TradeoffReport(BaseModel):
    """Human-readable tradeoff report for a completed job."""

    job_id: str
    timestamp: datetime
    parameters: JobParameters
    tradeoff_signals: TradeoffSignals
    status_summary: str
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    recommendations: list[str] = Field(default_factory=list)


class SystemHealth(BaseModel):
    """Overall system health snapshot."""

    overall_score: float = Field(..., ge=0.0, le=1.0)
    active_jobs: int
    jobs_per_minute: float
    tradeoff_scores: Dict[str, float]
    alerts_active: int
    model_version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

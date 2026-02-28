"""SQLAlchemy ORM models mapped to the LTO database schema.

Four core tables:
  - simulation_jobs: Master record for every simulation run
  - tradeoff_predictions: ML predictions (pre-flight and post-hoc)
  - alerts: Alert lifecycle records
  - model_registry: ML model versioning and deployment tracking
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Double, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all LTO models."""
    pass


class SimulationJobRecord(Base):
    """Master record for a simulation job."""

    __tablename__ = "simulation_jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    submitted_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    submitted_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    job_class: Mapped[int | None] = mapped_column(Integer, nullable=True)
    adapter_type: Mapped[str] = mapped_column(String(32), nullable=False)
    simulator_version: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Parameters (denormalized for query performance)
    na: Mapped[float | None] = mapped_column(Double, nullable=True)
    wavelength_nm: Mapped[float | None] = mapped_column(Double, nullable=True)
    dose_mj_cm2: Mapped[float | None] = mapped_column(Double, nullable=True)
    sigma: Mapped[float | None] = mapped_column(Double, nullable=True)
    resist_thickness_nm: Mapped[float | None] = mapped_column(Double, nullable=True)
    grid_size_nm: Mapped[float | None] = mapped_column(Double, nullable=True)
    use_ai_surrogate: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    pattern_complexity: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # Raw outputs
    resolution_nm: Mapped[float | None] = mapped_column(Double, nullable=True)
    depth_of_focus_nm: Mapped[float | None] = mapped_column(Double, nullable=True)
    pattern_fidelity: Mapped[float | None] = mapped_column(Double, nullable=True)
    compute_time_s: Mapped[float | None] = mapped_column(Double, nullable=True)

    # Tradeoff signals
    speed_vs_accuracy: Mapped[float | None] = mapped_column(Double, nullable=True)
    resolution_vs_dof: Mapped[float | None] = mapped_column(Double, nullable=True)
    cost_vs_fidelity: Mapped[float | None] = mapped_column(Double, nullable=True)
    surrogate_reliability: Mapped[float | None] = mapped_column(Double, nullable=True)
    yield_risk: Mapped[float | None] = mapped_column(Double, nullable=True)
    overall_health: Mapped[float | None] = mapped_column(Double, nullable=True)


class PredictionRecord(Base):
    """ML prediction record — pre-flight or post-hoc."""

    __tablename__ = "tradeoff_predictions"

    prediction_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    predicted_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    prediction_type: Mapped[str | None] = mapped_column(String(16), nullable=True)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)

    # Predictions with confidence intervals
    pred_speed_vs_accuracy: Mapped[float | None] = mapped_column(Double, nullable=True)
    ci_speed_low: Mapped[float | None] = mapped_column(Double, nullable=True)
    ci_speed_high: Mapped[float | None] = mapped_column(Double, nullable=True)
    pred_yield_risk: Mapped[float | None] = mapped_column(Double, nullable=True)
    ci_yield_low: Mapped[float | None] = mapped_column(Double, nullable=True)
    ci_yield_high: Mapped[float | None] = mapped_column(Double, nullable=True)
    pred_overall_health: Mapped[float | None] = mapped_column(Double, nullable=True)

    # Uncertainty flags
    high_uncertainty: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    recommend_physics: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    inference_time_ms: Mapped[float | None] = mapped_column(Double, nullable=True)


class AlertRecord(Base):
    """Alert lifecycle record."""

    __tablename__ = "alerts"

    alert_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    fired_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    job_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    alert_type: Mapped[str] = mapped_column(String(64), nullable=False)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    recommendation: Mapped[str | None] = mapped_column(Text, nullable=True)
    acknowledged_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    was_actionable: Mapped[bool | None] = mapped_column(Boolean, nullable=True)


class ModelRecord(Base):
    """ML model registry entry."""

    __tablename__ = "model_registry"

    model_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    deployed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    retired_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    training_samples: Mapped[int | None] = mapped_column(Integer, nullable=True)
    overall_accuracy: Mapped[float | None] = mapped_column(Double, nullable=True)
    critical_accuracy: Mapped[float | None] = mapped_column(Double, nullable=True)
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)

"""Repository pattern for data access.

Provides high-level operations over the database, abstracting away
SQLAlchemy session management from business logic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from lto.schemas import SimulationResult
from lto.storage.models import AlertRecord, ModelRecord, PredictionRecord, SimulationJobRecord


class SimulationRepository:
    """Repository for simulation job records."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_result(self, result: SimulationResult) -> SimulationJobRecord:
        """Persist a simulation result to the database."""
        record = SimulationJobRecord(
            job_id=result.job_id,
            submitted_at=result.timestamp,
            completed_at=datetime.utcnow(),
            status="completed",
            submitted_by=result.parameters.submitted_by,
            job_class=result.parameters.job_class,
            adapter_type=result.adapter_type,
            simulator_version=result.simulator_version,
            # Parameters
            na=result.parameters.na,
            wavelength_nm=result.parameters.wavelength_nm,
            dose_mj_cm2=result.parameters.dose_mj_cm2,
            sigma=result.parameters.sigma,
            resist_thickness_nm=result.parameters.resist_thickness_nm,
            grid_size_nm=result.parameters.grid_size_nm,
            use_ai_surrogate=result.parameters.use_ai_surrogate,
            pattern_complexity=result.parameters.pattern_complexity.value,
            # Outputs
            resolution_nm=result.outputs.resolution_nm,
            depth_of_focus_nm=result.outputs.depth_of_focus_nm,
            pattern_fidelity=result.outputs.pattern_fidelity,
            compute_time_s=result.outputs.compute_time_s,
            # Tradeoff signals
            speed_vs_accuracy=result.tradeoff_signals.speed_vs_accuracy,
            resolution_vs_dof=result.tradeoff_signals.resolution_vs_dof,
            cost_vs_fidelity=result.tradeoff_signals.cost_vs_fidelity,
            surrogate_reliability=result.tradeoff_signals.surrogate_reliability,
            yield_risk=result.tradeoff_signals.yield_risk,
            overall_health=result.tradeoff_signals.overall_health,
        )
        self.session.add(record)
        await self.session.flush()
        return record

    async def get_by_id(self, job_id: str) -> Optional[SimulationJobRecord]:
        """Retrieve a simulation job by ID."""
        stmt = select(SimulationJobRecord).where(SimulationJobRecord.job_id == job_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_recent(self, limit: int = 50) -> list[SimulationJobRecord]:
        """Get the most recent simulation jobs."""
        stmt = (
            select(SimulationJobRecord)
            .order_by(SimulationJobRecord.submitted_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count(self) -> int:
        """Count total simulation jobs."""
        from sqlalchemy import func
        stmt = select(func.count()).select_from(SimulationJobRecord)
        result = await self.session.execute(stmt)
        return result.scalar_one()

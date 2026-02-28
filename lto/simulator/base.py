"""Abstract base class for all simulator adapters.

Every simulator — synthetic, PROLITH, cuLitho — implements this interface.
The Intelligence Layer only speaks to this interface, never to a concrete adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from lto.schemas import JobParameters, SimulationResult


class AdapterHealth:
    """Health status of a simulator adapter."""

    def __init__(self, healthy: bool, message: str = "OK", details: dict | None = None):
        self.healthy = healthy
        self.message = message
        self.details = details or {}

    def __repr__(self) -> str:
        status = "HEALTHY" if self.healthy else "UNHEALTHY"
        return f"AdapterHealth({status}: {self.message})"


class MetricSnapshot:
    """A point-in-time metric reading during simulation execution."""

    def __init__(self, metric_name: str, value: float, timestamp: float, labels: dict | None = None):
        self.metric_name = metric_name
        self.value = value
        self.timestamp = timestamp
        self.labels = labels or {}


class SimulatorInterface(ABC):
    """Abstract interface all simulators must implement.

    Design decision (ADR-001): Contract-based adapters ensure the core
    system never imports an adapter directly. Adding a new simulator
    requires only implementing this interface.
    """

    @abstractmethod
    def run_job(self, params: JobParameters) -> SimulationResult:
        """Execute one simulation job and return results."""
        ...

    @abstractmethod
    async def stream_metrics(self) -> AsyncIterator[MetricSnapshot]:
        """Stream real-time metrics during job execution."""
        ...

    @abstractmethod
    def health_check(self) -> AdapterHealth:
        """Return adapter health status."""
        ...

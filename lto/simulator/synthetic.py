"""Synthetic lithography simulator.

Generates physics-inspired simulation data representing the core tradeoffs
in computational lithography. Acts as a stand-in for PROLITH/cuLitho during
development, and as a data augmentation tool in production.

The synthetic data has realistic correlations between parameters and outputs,
with controlled noise to represent real-world variability.
"""

from __future__ import annotations

import math
import time
from typing import AsyncIterator

import numpy as np

from lto.schemas import (
    ComputeStats,
    JobParameters,
    SimulationOutputs,
    SimulationResult,
    TradeoffSignals,
)
from lto.simulator.base import AdapterHealth, MetricSnapshot, SimulatorInterface
from lto.simulator.models.optical import (
    aerial_image_contrast,
    depth_of_focus,
    resolution,
)


class SyntheticSimulator(SimulatorInterface):
    """Physics-inspired synthetic lithography simulator.

    Generates realistic data by computing outputs from first-principles
    optical equations and adding calibrated noise to represent process
    variability.
    """

    VERSION = "synthetic-v0.1"

    def __init__(self, noise_scale: float = 0.02, seed: int | None = None):
        """
        Args:
            noise_scale: Standard deviation of Gaussian noise (fraction of signal).
                         0.02 = 2% noise, realistic for modern litho tools.
            seed: Random seed for reproducibility.
        """
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # SimulatorInterface implementation
    # ------------------------------------------------------------------

    def run_job(self, params: JobParameters) -> SimulationResult:
        """Execute one synthetic simulation job."""
        t_start = time.perf_counter()

        # 1. Compute raw physical outputs
        outputs = self._compute_outputs(params)

        # 2. Compute tradeoff signals from outputs + parameters
        signals = self._compute_tradeoff_signals(params, outputs)

        # 3. Package compute stats
        wall_time = time.perf_counter() - t_start
        compute = ComputeStats(
            wall_time_s=round(wall_time, 4),
            cpu_time_s=round(wall_time * 0.95, 4),
            peak_memory_mb=round(50 + self.rng.normal(0, 5), 1),
        )

        return SimulationResult(
            job_id=params.job_id,
            parameters=params,
            outputs=outputs,
            tradeoff_signals=signals,
            compute_stats=compute,
            simulator_version=self.VERSION,
            adapter_type="synthetic",
        )

    async def stream_metrics(self) -> AsyncIterator[MetricSnapshot]:
        """Stream mock metrics (no real simulation running)."""
        yield MetricSnapshot(
            metric_name="synthetic_heartbeat",
            value=1.0,
            timestamp=time.time(),
        )

    def health_check(self) -> AdapterHealth:
        """Synthetic adapter is always healthy."""
        return AdapterHealth(
            healthy=True,
            message="Synthetic simulator operational",
            details={"version": self.VERSION, "noise_scale": self.noise_scale},
        )

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def generate_batch(self, n: int, randomize_params: bool = True) -> list[SimulationResult]:
        """Generate a batch of simulation results.

        Args:
            n: Number of jobs to generate.
            randomize_params: If True, randomly sample parameters.
                              If False, use default parameters for all jobs.

        Returns:
            List of SimulationResult objects.
        """
        results = []
        for _ in range(n):
            if randomize_params:
                params = self._random_params()
            else:
                params = JobParameters(na=0.33, dose_mj_cm2=15.0)
            results.append(self.run_job(params))
        return results

    def generate_dataframe(self, n: int) -> "pd.DataFrame":
        """Generate simulation data as a flat pandas DataFrame.

        Useful for ML training — flattens all nested structures into columns.
        """
        import pandas as pd

        results = self.generate_batch(n, randomize_params=True)
        rows = []
        for r in results:
            row = {
                "job_id": r.job_id,
                # Parameters
                "na": r.parameters.na,
                "wavelength_nm": r.parameters.wavelength_nm,
                "dose_mj_cm2": r.parameters.dose_mj_cm2,
                "sigma": r.parameters.sigma,
                "resist_thickness_nm": r.parameters.resist_thickness_nm,
                "grid_size_nm": r.parameters.grid_size_nm,
                "use_ai_surrogate": r.parameters.use_ai_surrogate,
                "pattern_complexity": r.parameters.pattern_complexity.value,
                "job_class": r.parameters.job_class,
                # Outputs
                "resolution_nm": r.outputs.resolution_nm,
                "depth_of_focus_nm": r.outputs.depth_of_focus_nm,
                "pattern_fidelity": r.outputs.pattern_fidelity,
                "compute_time_s": r.outputs.compute_time_s,
                "accuracy_vs_physics": r.outputs.accuracy_vs_physics,
                "yield_prediction": r.outputs.yield_prediction,
                # Tradeoff signals (targets for ML)
                "speed_vs_accuracy": r.tradeoff_signals.speed_vs_accuracy,
                "resolution_vs_dof": r.tradeoff_signals.resolution_vs_dof,
                "cost_vs_fidelity": r.tradeoff_signals.cost_vs_fidelity,
                "surrogate_reliability": r.tradeoff_signals.surrogate_reliability,
                "yield_risk": r.tradeoff_signals.yield_risk,
                "overall_health": r.tradeoff_signals.overall_health,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Private: Physics computation
    # ------------------------------------------------------------------

    def _compute_outputs(self, params: JobParameters) -> SimulationOutputs:
        """Compute physical outputs from input parameters."""

        # Resolution from Rayleigh criterion
        res_nm = resolution(params.wavelength_nm, params.na)

        # Depth of focus — inversely related to NA²
        dof_nm = depth_of_focus(params.wavelength_nm, params.na)

        # Pattern fidelity — depends on dose, grid size, and contrast
        # Higher dose + finer grid → better fidelity
        dose_factor = self._sigmoid_normalize(params.dose_mj_cm2, center=15.0, scale=5.0)
        grid_factor = self._sigmoid_normalize(1.0 / params.grid_size_nm, center=1.0, scale=0.5)
        contrast = aerial_image_contrast(
            params.na, params.sigma, params.wavelength_nm, pitch_nm=res_nm * 2
        )
        fidelity = 0.4 * dose_factor + 0.3 * grid_factor + 0.3 * contrast
        fidelity = self._add_noise(fidelity)

        # Compute time — scales with grid fineness and pattern complexity
        complexity_mul = {"simple": 0.5, "moderate": 1.0, "complex": 2.0, "extreme": 4.0}
        base_time = (1.0 / params.grid_size_nm) ** 2
        compute_time = base_time * complexity_mul[params.pattern_complexity.value]
        if not params.use_ai_surrogate:
            compute_time *= 10.0  # Full physics is ~10x slower
        compute_time = max(0.1, self._add_noise(compute_time, scale=0.1))

        # Accuracy vs physics — surrogate accuracy degrades with parameter extremes
        if params.use_ai_surrogate:
            # Surrogate accuracy: high in well-explored regions, lower at extremes
            na_deviation = abs(params.na - 0.33) / 0.22  # 0=center, 1=edge of range
            dose_deviation = abs(params.dose_mj_cm2 - 20.0) / 20.0
            accuracy = 0.995 - 0.03 * na_deviation - 0.02 * dose_deviation
            accuracy = self._add_noise(accuracy, scale=0.005)
        else:
            accuracy = 1.0  # Physics simulation is ground truth

        # Yield prediction — function of DoF margin and pattern fidelity
        dof_margin = min(1.0, dof_nm / 60.0)  # 60nm is comfortable DoF
        yield_pred = 0.5 * dof_margin + 0.3 * fidelity + 0.2 * accuracy
        yield_pred = self._add_noise(yield_pred)

        return SimulationOutputs(
            resolution_nm=round(self._add_noise(res_nm, scale=0.01), 2),
            depth_of_focus_nm=round(self._add_noise(dof_nm, scale=0.01), 2),
            pattern_fidelity=round(self._clamp(fidelity), 4),
            compute_time_s=round(compute_time, 3),
            accuracy_vs_physics=round(self._clamp(accuracy), 4),
            yield_prediction=round(self._clamp(yield_pred), 4),
        )

    def _compute_tradeoff_signals(
        self, params: JobParameters, outputs: SimulationOutputs
    ) -> TradeoffSignals:
        """Compute tradeoff health scores from parameters + outputs.

        Each score is 0–1 where higher = healthier tradeoff.
        """

        # Speed vs Accuracy: good when compute time is low AND accuracy is high
        speed_norm = self._sigmoid_normalize(
            1.0 / max(0.01, outputs.compute_time_s), center=0.5, scale=0.3
        )
        speed_vs_accuracy = 0.4 * speed_norm + 0.6 * outputs.accuracy_vs_physics

        # Resolution vs DoF: good when both resolution and DoF are acceptable
        res_quality = self._sigmoid_normalize(1.0 / outputs.resolution_nm, center=0.1, scale=0.05)
        dof_quality = min(1.0, outputs.depth_of_focus_nm / 50.0)
        resolution_vs_dof = 0.5 * res_quality + 0.5 * dof_quality

        # Cost vs Fidelity: good when fidelity is high per unit compute
        efficiency = outputs.pattern_fidelity / max(0.01, outputs.compute_time_s)
        cost_vs_fidelity = self._sigmoid_normalize(efficiency, center=0.5, scale=0.3)

        # Surrogate reliability: based on accuracy and whether surrogate is being used
        if params.use_ai_surrogate:
            surrogate_reliability = outputs.accuracy_vs_physics
        else:
            surrogate_reliability = 1.0

        # Yield risk: 0=no risk, 1=high risk
        # Risk increases with low DoF, low fidelity, or low accuracy
        yield_risk = 1.0 - outputs.yield_prediction

        # Overall health: weighted combination
        overall = (
            0.25 * speed_vs_accuracy
            + 0.25 * resolution_vs_dof
            + 0.20 * cost_vs_fidelity
            + 0.15 * surrogate_reliability
            + 0.15 * (1.0 - yield_risk)
        )

        return TradeoffSignals(
            speed_vs_accuracy=round(self._clamp(speed_vs_accuracy), 4),
            resolution_vs_dof=round(self._clamp(resolution_vs_dof), 4),
            cost_vs_fidelity=round(self._clamp(cost_vs_fidelity), 4),
            surrogate_reliability=round(self._clamp(surrogate_reliability), 4),
            yield_risk=round(self._clamp(yield_risk), 4),
            overall_health=round(self._clamp(overall), 4),
        )

    # ------------------------------------------------------------------
    # Private: Utilities
    # ------------------------------------------------------------------

    def _random_params(self) -> JobParameters:
        """Generate random but physically plausible job parameters."""
        wavelength = self.rng.choice([13.5, 193.0, 248.0], p=[0.5, 0.35, 0.15])

        # NA range depends on wavelength
        if wavelength == 13.5:  # EUV
            na = self.rng.uniform(0.25, 0.55)
        elif wavelength == 193.0:  # ArF
            na = self.rng.uniform(0.3, 0.55)
        else:  # KrF
            na = self.rng.uniform(0.3, 0.5)

        complexities = list(c.value for c in __import__("lto.schemas", fromlist=["PatternComplexity"]).PatternComplexity)
        complexity = self.rng.choice(complexities)

        return JobParameters(
            na=round(na, 3),
            wavelength_nm=wavelength,
            dose_mj_cm2=round(self.rng.uniform(5.0, 40.0), 1),
            sigma=round(self.rng.uniform(0.3, 0.95), 2),
            resist_thickness_nm=round(self.rng.uniform(10.0, 100.0), 1),
            grid_size_nm=round(self.rng.choice([0.5, 1.0, 2.0, 5.0]), 1),
            pattern_complexity=complexity,
            use_ai_surrogate=bool(self.rng.choice([True, False], p=[0.7, 0.3])),
            job_class=int(self.rng.integers(1, 15)),
        )

    def _add_noise(self, value: float, scale: float | None = None) -> float:
        """Add Gaussian noise to a value."""
        s = scale if scale is not None else self.noise_scale
        return value * (1.0 + self.rng.normal(0, s))

    @staticmethod
    def _sigmoid_normalize(x: float, center: float = 0.0, scale: float = 1.0) -> float:
        """Map a value to [0, 1] using a sigmoid function."""
        z = (x - center) / max(scale, 1e-8)
        return 1.0 / (1.0 + math.exp(-z))

    @staticmethod
    def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        """Clamp a value to [lo, hi]."""
        return max(lo, min(hi, value))

"""Uncertainty quantifier — combines multiple uncertainty sources.

Three sources of uncertainty (per ARCHITECTURE.md):
1. Aleatory (data noise) — from training data variance
2. Epistemic (model uncertainty) — from GP standard deviation
3. Input distribution shift — from drift detection

The quantifier decides whether to trust the ensemble prediction
or recommend a full physics simulation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from lto.schemas import ConfidenceLevel, UncertaintyInfo

logger = logging.getLogger(__name__)


@dataclass
class UncertaintySources:
    """Detailed breakdown of uncertainty sources."""

    aleatory: float = 0.0
    epistemic: float = 0.0
    input_shift: float = 0.0
    combined: float = 0.0
    per_target: dict[str, float] = field(default_factory=dict)


class UncertaintyQuantifier:
    """Multi-source uncertainty quantifier.

    Combines GP standard deviations (epistemic), residual variance (aleatory),
    and distribution shift signals to produce a holistic uncertainty estimate.
    """

    def __init__(
        self,
        epistemic_threshold: float = 0.10,
        combined_threshold: float = 0.15,
        critical_targets: list[str] | None = None,
    ):
        self.epistemic_threshold = epistemic_threshold
        self.combined_threshold = combined_threshold
        self.critical_targets = critical_targets or ["yield_risk", "surrogate_reliability"]
        self._residual_std: dict[str, float] = {}

    def set_residual_statistics(self, residuals: dict[str, np.ndarray]) -> None:
        """Set aleatory uncertainty from training residuals.

        Args:
            residuals: Dict of target name → residual array (actual - predicted).
        """
        self._residual_std = {
            name: float(np.std(vals)) for name, vals in residuals.items()
        }

    def quantify(
        self,
        gp_stds: dict[str, float],
        predictions: dict[str, float],
        input_shift_score: float = 0.0,
    ) -> tuple[UncertaintyInfo, UncertaintySources]:
        """Quantify uncertainty from all sources.

        Args:
            gp_stds: GP standard deviations per target.
            predictions: Ensemble predictions per target.
            input_shift_score: How far the input is from training distribution (0=normal).

        Returns:
            Tuple of (UncertaintyInfo for API, UncertaintySources for diagnostics).
        """
        # Epistemic: max GP std across targets
        epistemic = max(gp_stds.values()) if gp_stds else 0.0

        # Aleatory: average residual std
        aleatory = np.mean(list(self._residual_std.values())) if self._residual_std else 0.02

        # Combined
        combined = np.sqrt(epistemic ** 2 + aleatory ** 2 + input_shift_score ** 2)

        # Decision logic
        high_uncertainty = combined > self.combined_threshold
        recommend_physics = False
        reason = None

        # Check critical targets specifically
        for target in self.critical_targets:
            if target in gp_stds and gp_stds[target] > self.epistemic_threshold:
                high_uncertainty = True
                recommend_physics = True
                reason = f"High uncertainty on critical target '{target}': std={gp_stds[target]:.3f}"
                break

        if not reason and high_uncertainty:
            reason = f"Combined uncertainty ({combined:.3f}) exceeds threshold ({self.combined_threshold})"

        # Confidence level
        if combined < 0.05:
            confidence = ConfidenceLevel.HIGH
        elif combined < self.combined_threshold:
            confidence = ConfidenceLevel.MEDIUM
        else:
            confidence = ConfidenceLevel.LOW

        info = UncertaintyInfo(
            high_uncertainty=high_uncertainty,
            recommend_physics_simulation=recommend_physics,
            confidence_level=confidence,
            reason=reason,
        )

        sources = UncertaintySources(
            aleatory=float(aleatory),
            epistemic=float(epistemic),
            input_shift=float(input_shift_score),
            combined=float(combined),
            per_target=gp_stds,
        )

        return info, sources

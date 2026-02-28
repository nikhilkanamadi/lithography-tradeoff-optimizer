"""Population Stability Index (PSI) based drift detector.

Monitors feature distributions over time and triggers retraining
when significant distribution shift is detected. Per ARCHITECTURE.md,
PSI > 0.2 = significant drift requiring action.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Drift detection results."""

    psi_scores: dict[str, float] = field(default_factory=dict)
    drifted_features: list[str] = field(default_factory=list)
    requires_retraining: bool = False
    overall_psi: float = 0.0
    message: str = "No drift detected"


class DriftDetector:
    """PSI-based distribution drift detector.

    Compares current batch feature distributions against the
    training distribution. Uses 10-bin PSI histogram comparison.
    """

    PSI_THRESHOLD = 0.2  # Per ARCHITECTURE.md

    def __init__(self, n_bins: int = 10, feature_names: list[str] | None = None):
        self.n_bins = n_bins
        self.feature_names = feature_names or []
        self._reference_bins: dict[str, np.ndarray] = {}
        self._reference_edges: dict[str, np.ndarray] = {}
        self._fitted = False

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> None:
        """Fit on training data distribution.

        Args:
            X: Training feature matrix (n_samples, n_features).
            feature_names: Names for each feature column.
        """
        if feature_names:
            self.feature_names = feature_names
        elif not self.feature_names:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        for i, name in enumerate(self.feature_names):
            counts, edges = np.histogram(X[:, i], bins=self.n_bins)
            # Add small epsilon to avoid division by zero
            freqs = (counts + 1e-8) / (counts.sum() + 1e-8 * len(counts))
            self._reference_bins[name] = freqs
            self._reference_edges[name] = edges

        self._fitted = True
        logger.info(f"Drift detector fitted on {X.shape[0]} samples, {X.shape[1]} features")

    def detect(self, X: np.ndarray) -> DriftReport:
        """Check for drift against reference distribution.

        Args:
            X: New feature matrix (n_samples, n_features).

        Returns:
            DriftReport with per-feature PSI scores and overall assessment.
        """
        if not self._fitted:
            raise RuntimeError("DriftDetector must be fitted first.")

        psi_scores = {}
        drifted = []

        for i, name in enumerate(self.feature_names):
            if i >= X.shape[1]:
                break
            psi = self._compute_psi(
                self._reference_bins[name],
                self._reference_edges[name],
                X[:, i],
            )
            psi_scores[name] = float(psi)
            if psi > self.PSI_THRESHOLD:
                drifted.append(name)

        overall = float(np.mean(list(psi_scores.values()))) if psi_scores else 0.0
        requires_retraining = len(drifted) > 0 or overall > self.PSI_THRESHOLD

        message = (
            f"Drift detected in {len(drifted)} features: {drifted}"
            if drifted else "No significant drift detected"
        )

        return DriftReport(
            psi_scores=psi_scores,
            drifted_features=drifted,
            requires_retraining=requires_retraining,
            overall_psi=overall,
            message=message,
        )

    def _compute_psi(
        self,
        reference_freqs: np.ndarray,
        reference_edges: np.ndarray,
        new_values: np.ndarray,
    ) -> float:
        """Compute PSI between reference and new distributions.

        PSI = Σ (P_new - P_ref) * ln(P_new / P_ref)
        """
        counts, _ = np.histogram(new_values, bins=reference_edges)
        new_freqs = (counts + 1e-8) / (counts.sum() + 1e-8 * len(counts))

        psi = np.sum(
            (new_freqs - reference_freqs) * np.log(new_freqs / reference_freqs)
        )
        return max(0.0, float(psi))

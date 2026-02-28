"""Gaussian Process uncertainty model.

Uses scikit-learn's GaussianProcessRegressor to provide calibrated
uncertainty estimates for tradeoff predictions. This is the safety
layer — when the GP is unsure, LTO recommends running a full physics
simulation instead of trusting the surrogate.

GP is intentionally trained on fewer samples (subsampled) because
GP scales O(n³). We use a sparse subset + RBF + WhiteKernel.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

logger = logging.getLogger(__name__)


class GPUncertaintyModel:
    """Gaussian Process model for uncertainty quantification.

    Provides mean predictions AND calibrated standard deviations.
    The std deviation is the key signal: when it's high, the model
    is operating outside its confident region.
    """

    VERSION = "gp-v0.1"
    MAX_TRAIN_SAMPLES = 2000  # O(n³) constraint

    def __init__(
        self,
        target_names: list[str] | None = None,
        n_restarts: int = 3,
    ):
        self.target_names = target_names or [
            "speed_vs_accuracy",
            "resolution_vs_dof",
            "cost_vs_fidelity",
            "surrogate_reliability",
            "yield_risk",
        ]
        self.n_restarts = n_restarts
        self.models: dict[str, GaussianProcessRegressor] = {}
        self._trained = False

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_samples: int | None = None,
    ) -> dict[str, float]:
        """Train GP models on (potentially subsampled) data.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target matrix (n_samples, n_targets).
            max_samples: Maximum training samples (default: MAX_TRAIN_SAMPLES).

        Returns:
            Per-target log-marginal-likelihood.
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        max_n = max_samples or self.MAX_TRAIN_SAMPLES
        if len(X) > max_n:
            logger.info(f"Subsampling GP training data: {len(X)} → {max_n}")
            idx = np.random.RandomState(42).choice(len(X), max_n, replace=False)
            X = X[idx]
            y = y[idx]

        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01)

        metrics = {}
        for i, name in enumerate(self.target_names):
            logger.info(f"Training GP for target: {name}")
            gp = GaussianProcessRegressor(
                kernel=kernel.clone_with_theta(kernel.theta),
                n_restarts_optimizer=self.n_restarts,
                normalize_y=True,
                random_state=42,
            )
            gp.fit(X, y[:, i])
            lml = gp.log_marginal_likelihood_value_
            metrics[name] = float(lml)
            logger.info(f"  {name} — Log-Marginal-Likelihood: {lml:.4f}")
            self.models[name] = gp

        self._trained = True
        return metrics

    def predict(self, X: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Predict with uncertainty estimates.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Tuple of (means_dict, stds_dict):
                means_dict: target name → predicted mean values
                stds_dict: target name → predicted standard deviations
        """
        if not self._trained:
            raise RuntimeError("GP model must be trained before prediction.")

        means = {}
        stds = {}
        for name in self.target_names:
            mean, std = self.models[name].predict(X, return_std=True)
            means[name] = np.clip(mean, 0.0, 1.0)
            stds[name] = std

        return means, stds

    def predict_single(
        self, X: np.ndarray
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Predict for a single sample with uncertainty.

        Returns:
            Tuple of (means_dict, stds_dict) with single float values.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        means_batch, stds_batch = self.predict(X)
        means = {name: float(vals[0]) for name, vals in means_batch.items()}
        stds = {name: float(vals[0]) for name, vals in stds_batch.items()}
        return means, stds

    def is_high_uncertainty(self, stds: dict[str, float], threshold: float = 0.1) -> bool:
        """Check if any target exceeds uncertainty threshold.

        Args:
            stds: Dict of target name → standard deviation.
            threshold: Std threshold above which uncertainty is flagged.

        Returns:
            True if any target has std > threshold.
        """
        return any(s > threshold for s in stds.values())

    def save(self, path: str | Path) -> None:
        """Save GP models."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "models": self.models,
            "target_names": self.target_names,
            "version": self.VERSION,
        }
        joblib.dump(state, path)
        logger.info(f"GP model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "GPUncertaintyModel":
        """Load GP models."""
        state = joblib.load(path)
        instance = cls(target_names=state["target_names"])
        instance.models = state["models"]
        instance._trained = True
        logger.info(f"GP model loaded from {path}")
        return instance

    @property
    def is_trained(self) -> bool:
        return self._trained

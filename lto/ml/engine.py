"""TradeoffModel — main inference engine (V0.2: ensemble edition).

Wraps the ensemble (XGBoost + MLP + GP) with feature engineering,
prediction, uncertainty quantification, and constraint checking.
This is the single entry point the rest of the system uses.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from lto.ml.drift.detector import DriftDetector
from lto.ml.ensemble import EnsembleModel
from lto.ml.features.engineering import (
    FeatureScaler,
    extract_features,
    get_feature_columns,
    get_target_columns,
)
from lto.ml.models.xgboost_model import XGBoostTradeoffModel
from lto.ml.uncertainty.quantifier import UncertaintyQuantifier
from lto.schemas import (
    ConfidenceLevel,
    ConstraintResult,
    JobParameters,
    ScoreWithCI,
    TradeoffPrediction,
    UncertaintyInfo,
)

logger = logging.getLogger(__name__)


class TradeoffModel:
    """Main tradeoff prediction engine.

    V0.2 upgrade: Uses full ensemble (XGBoost + MLP + GP) with
    calibrated uncertainty from GP standard deviations.

    Usage:
        model = TradeoffModel()
        model.train_from_simulator(n_samples=5000)
        prediction = model.predict(job_params)
    """

    VERSION = "v0.2"

    def __init__(self, model_dir: str | Path = "models", use_ensemble: bool = True):
        self.model_dir = Path(model_dir)
        self.use_ensemble = use_ensemble
        self.ensemble: Optional[EnsembleModel] = None
        self.xgboost_only: Optional[XGBoostTradeoffModel] = None
        self.scaler = FeatureScaler()
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.drift_detector = DriftDetector()
        self._trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_from_dataframe(self, df: pd.DataFrame) -> dict:
        """Train the model from a simulator-generated DataFrame."""
        logger.info(f"Training on {len(df)} samples (ensemble={self.use_ensemble})...")

        # Extract features
        features_df = extract_features(df)
        X = self.scaler.fit_transform(features_df)

        # Extract targets
        target_cols = get_target_columns()
        y = df[target_cols].values

        # Fit drift detector on training features
        self.drift_detector.fit(X, feature_names=list(features_df.columns))

        if self.use_ensemble:
            self.ensemble = EnsembleModel()
            metrics = self.ensemble.train(X, y)
        else:
            self.xgboost_only = XGBoostTradeoffModel()
            xgb_metrics = self.xgboost_only.train(X, y)
            metrics = {"xgboost": xgb_metrics}

        self._trained = True
        logger.info(f"Training complete. Ensemble={self.use_ensemble}")
        return metrics

    def train_from_simulator(self, n_samples: int = 5000, seed: int = 42) -> dict:
        """Generate data from SyntheticSimulator and train."""
        from lto.simulator.synthetic import SyntheticSimulator

        sim = SyntheticSimulator(seed=seed)
        df = sim.generate_dataframe(n_samples)
        return self.train_from_dataframe(df)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, params: JobParameters) -> TradeoffPrediction:
        """Predict tradeoff scores for given job parameters."""
        if not self._trained:
            raise RuntimeError("TradeoffModel must be trained before prediction.")

        t_start = time.perf_counter()

        # Build single-row DataFrame
        row_df = pd.DataFrame([{
            "na": params.na,
            "wavelength_nm": params.wavelength_nm,
            "dose_mj_cm2": params.dose_mj_cm2,
            "sigma": params.sigma,
            "resist_thickness_nm": params.resist_thickness_nm,
            "grid_size_nm": params.grid_size_nm,
            "use_ai_surrogate": params.use_ai_surrogate,
            "pattern_complexity": params.pattern_complexity.value,
            "job_class": params.job_class,
        }])

        features_df = extract_features(row_df)
        X = self.scaler.transform(features_df)

        if self.use_ensemble and self.ensemble:
            raw_preds, gp_stds, uncertainty_flags = self.ensemble.predict_single(X)

            # Build confidence intervals from GP stds
            predictions = {}
            for name, score in raw_preds.items():
                std = gp_stds.get(name, 0.05)
                margin = 1.96 * std  # 95% CI
                predictions[name] = ScoreWithCI(
                    score=round(score, 4),
                    ci_low=round(max(0.0, score - margin), 4),
                    ci_high=round(min(1.0, score + margin), 4),
                )

            # Uncertainty quantification
            uncertainty, sources = self.uncertainty_quantifier.quantify(
                gp_stds=gp_stds,
                predictions=raw_preds,
            )
        else:
            # Fallback: XGBoost only
            raw_preds = self.xgboost_only.predict_single(X)
            predictions = {}
            for name, score in raw_preds.items():
                margin = 0.05
                predictions[name] = ScoreWithCI(
                    score=round(score, 4),
                    ci_low=round(max(0.0, score - margin), 4),
                    ci_high=round(min(1.0, score + margin), 4),
                )
            gp_stds = {}
            avg_score = np.mean(list(raw_preds.values()))
            high_unc = avg_score < 0.3 or any(v < 0.2 for v in raw_preds.values())
            uncertainty = UncertaintyInfo(
                high_uncertainty=high_unc,
                recommend_physics_simulation=high_unc and params.use_ai_surrogate,
                confidence_level=(
                    ConfidenceLevel.LOW if high_unc
                    else ConfidenceLevel.MEDIUM if avg_score < 0.5
                    else ConfidenceLevel.HIGH
                ),
            )

        # Constraint checking
        violations = []
        raw_vals = {name: sci.score for name, sci in predictions.items()}
        if raw_vals.get("yield_risk", 0) > 0.8:
            violations.append("Yield risk exceeds 0.8 — CRITICAL threshold")
        if raw_vals.get("speed_vs_accuracy", 1) < 0.4:
            violations.append("Speed vs accuracy below 0.4 — degraded tradeoff")
        if raw_vals.get("surrogate_reliability", 1) < 0.95 and params.use_ai_surrogate:
            violations.append("Surrogate reliability below 0.95 — consider physics simulation")

        constraints = ConstraintResult(
            all_satisfied=len(violations) == 0,
            violations=violations,
        )

        inference_ms = (time.perf_counter() - t_start) * 1000

        return TradeoffPrediction(
            job_id=params.job_id,
            predictions=predictions,
            uncertainty=uncertainty,
            constraints=constraints,
            model_version=self.VERSION,
            inference_time_ms=round(inference_ms, 2),
            recommend_block=len(violations) > 0 and raw_vals.get("yield_risk", 0) > 0.8,
            block_reason=violations[0] if violations else None,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, name: str = "tradeoff_v1") -> Path:
        """Save the complete model to disk."""
        model_path = self.model_dir / f"{name}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "ensemble": self.ensemble,
            "xgboost_only": self.xgboost_only,
            "scaler": self.scaler,
            "drift_detector": self.drift_detector,
            "use_ensemble": self.use_ensemble,
            "version": self.VERSION,
        }
        joblib.dump(state, model_path)
        logger.info(f"TradeoffModel saved to {model_path}")
        return model_path

    @classmethod
    def load(cls, path: str | Path) -> "TradeoffModel":
        """Load a complete model from disk."""
        state = joblib.load(path)
        use_ensemble = state.get("use_ensemble", False)
        instance = cls(use_ensemble=use_ensemble)
        instance.ensemble = state.get("ensemble")
        instance.xgboost_only = state.get("xgboost_only")
        instance.scaler = state["scaler"]
        instance.drift_detector = state.get("drift_detector", DriftDetector())
        instance._trained = True
        version = state.get("version", "v0.1")
        logger.info(f"TradeoffModel loaded from {path} (version: {version})")
        return instance

    @property
    def is_trained(self) -> bool:
        return self._trained

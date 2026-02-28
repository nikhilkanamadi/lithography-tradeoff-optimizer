"""XGBoost tradeoff model component.

500-tree gradient boosting model that predicts 5 tradeoff scores
from lithography simulation parameters. This is the V0.1 baseline model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class XGBoostTradeoffModel:
    """XGBoost multi-output tradeoff predictor.

    Trains one XGBRegressor per target (5 tradeoff scores).
    Uses early stopping on validation set to prevent overfitting.
    """

    VERSION = "xgb-v0.1"

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        target_names: list[str] | None = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.target_names = target_names or [
            "speed_vs_accuracy",
            "resolution_vs_dof",
            "cost_vs_fidelity",
            "surrogate_reliability",
            "yield_risk",
        ]
        self.models: dict[str, xgb.XGBRegressor] = {}
        self._trained = False

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_fraction: float = 0.15,
    ) -> dict[str, float]:
        """Train the model on feature matrix X and target matrix y.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target matrix of shape (n_samples, n_targets).
            eval_fraction: Fraction of data for validation/early stopping.

        Returns:
            Dictionary of per-target RMSE on validation set.
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        assert y.shape[1] == len(self.target_names), (
            f"Expected {len(self.target_names)} targets, got {y.shape[1]}"
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=eval_fraction, random_state=42
        )

        metrics = {}
        for i, name in enumerate(self.target_names):
            logger.info(f"Training XGBoost model for target: {name}")

            model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                early_stopping_rounds=30,
                eval_metric="rmse",
            )

            model.fit(
                X_train,
                y_train[:, i],
                eval_set=[(X_val, y_val[:, i])],
                verbose=False,
            )

            # Evaluate
            val_pred = model.predict(X_val)
            rmse = float(np.sqrt(np.mean((val_pred - y_val[:, i]) ** 2)))
            metrics[name] = rmse
            logger.info(f"  {name} — Val RMSE: {rmse:.4f}")

            self.models[name] = model

        self._trained = True
        return metrics

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Predict tradeoff scores for input features.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Dictionary mapping target name → predicted values array.
        """
        if not self._trained:
            raise RuntimeError("Model must be trained before prediction.")

        predictions = {}
        for name in self.target_names:
            preds = self.models[name].predict(X)
            predictions[name] = np.clip(preds, 0.0, 1.0)
        return predictions

    def predict_single(self, X: np.ndarray) -> dict[str, float]:
        """Predict tradeoff scores for a single sample.

        Args:
            X: Feature array of shape (1, n_features) or (n_features,).

        Returns:
            Dictionary mapping target name → single predicted float value.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        batch_preds = self.predict(X)
        return {name: float(vals[0]) for name, vals in batch_preds.items()}

    def feature_importance(self) -> dict[str, dict[str, float]]:
        """Get feature importance for each target.

        Returns:
            Dict of {target_name: {feature_index: importance_score}}.
        """
        if not self._trained:
            raise RuntimeError("Model must be trained first.")

        result = {}
        for name, model in self.models.items():
            importance = model.feature_importances_
            result[name] = {str(i): float(v) for i, v in enumerate(importance)}
        return result

    def save(self, path: str | Path) -> None:
        """Save the model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "models": self.models,
            "target_names": self.target_names,
            "version": self.VERSION,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
        }
        joblib.dump(state, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostTradeoffModel":
        """Load a model from disk."""
        state = joblib.load(path)
        instance = cls(
            n_estimators=state["n_estimators"],
            max_depth=state["max_depth"],
            learning_rate=state["learning_rate"],
            target_names=state["target_names"],
        )
        instance.models = state["models"]
        instance._trained = True
        logger.info(f"Model loaded from {path} (version: {state['version']})")
        return instance

    @property
    def is_trained(self) -> bool:
        return self._trained

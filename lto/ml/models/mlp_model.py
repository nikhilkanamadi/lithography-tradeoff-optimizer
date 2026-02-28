"""PyTorch MLP tradeoff model component.

A 4-layer fully connected neural network that predicts 5 tradeoff scores.
Architecture: Input(16) → 256 → 128 → 64 → 5 (with BatchNorm, Dropout, GELU).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class TradeoffMLP(nn.Module):
    """Multi-layer perceptron for tradeoff score prediction."""

    def __init__(self, input_dim: int = 16, output_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.15),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(64, output_dim),
            nn.Sigmoid(),  # Outputs bounded to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPTradeoffModel:
    """Wrapper around TradeoffMLP for training, inference, and persistence.

    Handles training loop, early stopping, and serialization.
    Designed to be a drop-in parallel to XGBoostTradeoffModel.
    """

    VERSION = "mlp-v0.1"

    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 5,
        lr: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 256,
        patience: int = 20,
        target_names: list[str] | None = None,
        device: str | None = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.target_names = target_names or [
            "speed_vs_accuracy",
            "resolution_vs_dof",
            "cost_vs_fidelity",
            "surrogate_reliability",
            "yield_risk",
        ]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[TradeoffMLP] = None
        self._trained = False

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_fraction: float = 0.15,
    ) -> dict[str, float]:
        """Train the MLP on feature matrix X and target matrix y.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target matrix (n_samples, n_targets).
            eval_fraction: Fraction for validation.

        Returns:
            Dictionary of per-target RMSE on validation set.
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Split
        n_val = int(len(X) * eval_fraction)
        indices = np.random.RandomState(42).permutation(len(X))
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        X_train = torch.FloatTensor(X[train_idx]).to(self.device)
        y_train = torch.FloatTensor(y[train_idx]).to(self.device)
        X_val = torch.FloatTensor(X[val_idx]).to(self.device)
        y_val = torch.FloatTensor(y[val_idx]).to(self.device)

        # Create model
        self.model = TradeoffMLP(
            input_dim=X.shape[1], output_dim=y.shape[1]
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        criterion = nn.MSELoss()

        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = criterion(val_pred, y_val).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 50 == 0:
                logger.info(
                    f"  Epoch {epoch + 1}: train_loss={epoch_loss / len(train_loader):.6f}, "
                    f"val_loss={val_loss:.6f}"
                )

        # Load best weights
        if best_state:
            self.model.load_state_dict(best_state)
        self.model.to(self.device)

        # Compute per-target RMSE
        self.model.eval()
        with torch.no_grad():
            val_pred = self.model(X_val).cpu().numpy()
        y_val_np = y_val.cpu().numpy()

        metrics = {}
        for i, name in enumerate(self.target_names):
            rmse = float(np.sqrt(np.mean((val_pred[:, i] - y_val_np[:, i]) ** 2)))
            metrics[name] = rmse
            logger.info(f"  MLP {name} — Val RMSE: {rmse:.4f}")

        self._trained = True
        return metrics

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Predict tradeoff scores.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Dictionary mapping target name → predicted values array.
        """
        if not self._trained or self.model is None:
            raise RuntimeError("MLP model must be trained before prediction.")

        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X).to(self.device)
            preds = self.model(x_tensor).cpu().numpy()

        return {
            name: np.clip(preds[:, i], 0.0, 1.0)
            for i, name in enumerate(self.target_names)
        }

    def predict_single(self, X: np.ndarray) -> dict[str, float]:
        """Predict for a single sample."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        batch_preds = self.predict(X)
        return {name: float(vals[0]) for name, vals in batch_preds.items()}

    def save(self, path: str | Path) -> None:
        """Save model state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict() if self.model else None,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "target_names": self.target_names,
            "version": self.VERSION,
        }, path)
        logger.info(f"MLP model saved to {path}")

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "MLPTradeoffModel":
        """Load model state."""
        state = torch.load(path, map_location=device or "cpu", weights_only=False)
        instance = cls(
            input_dim=state["input_dim"],
            output_dim=state["output_dim"],
            target_names=state["target_names"],
            device=device,
        )
        instance.model = TradeoffMLP(state["input_dim"], state["output_dim"])
        instance.model.load_state_dict(state["model_state"])
        instance.model.to(instance.device)
        instance._trained = True
        logger.info(f"MLP model loaded from {path}")
        return instance

    @property
    def is_trained(self) -> bool:
        return self._trained

"""Shared dependencies for API routers.

Avoids circular imports by keeping the model state separate from main.py.
"""

from __future__ import annotations

from lto.ml.engine import TradeoffModel

# Global model instance — set during app lifespan
_model: TradeoffModel | None = None


def set_model(model: TradeoffModel) -> None:
    """Set the global model instance (called from lifespan)."""
    global _model
    _model = model


def get_model() -> TradeoffModel:
    """Get the global loaded TradeoffModel. Used as dependency in routers."""
    if _model is None:
        raise RuntimeError("Model not loaded. Server not fully started.")
    return _model

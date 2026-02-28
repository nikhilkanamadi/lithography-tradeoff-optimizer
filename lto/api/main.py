"""FastAPI application for the LTO platform.

This is the main entry point for the REST API.

Run with:
    uvicorn lto.api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from lto.api.deps import set_model
from lto.ml.engine import TradeoffModel

logger = logging.getLogger(__name__)

# Frontend static directory
FRONTEND_DIR = Path(__file__).parent.parent / "frontend" / "static"


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — load model at startup."""
    model_path = Path("models/tradeoff_v1.pkl")
    if model_path.exists():
        logger.info(f"Loading model from {model_path}")
        model = TradeoffModel.load(model_path)
    else:
        logger.warning(
            f"Model file not found at {model_path}. "
            "Training a fresh model with 5000 samples..."
        )
        model = TradeoffModel()
        model.train_from_simulator(n_samples=5000)
        model.save("tradeoff_v1")
        logger.info("Model trained and saved.")

    set_model(model)
    logger.info("LTO API ready.")
    yield
    logger.info("LTO API shutting down.")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LTO — Lithography Tradeoff Optimizer",
    description=(
        "Tradeoff observability platform for computational lithography. "
        "Predict, monitor, and manage tradeoffs in semiconductor manufacturing."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Import routers AFTER app is created (no circular dependency now)
from lto.api.routers import health, jobs, tradeoffs  # noqa: E402

app.include_router(health.router, prefix="/api/v1/health", tags=["Health"])
app.include_router(jobs.router, prefix="/api/v1/jobs", tags=["Jobs"])
app.include_router(tradeoffs.router, prefix="/api/v1/tradeoffs", tags=["Tradeoffs"])


@app.get("/", include_in_schema=False)
async def root():
    """Serve the dashboard frontend."""
    return FileResponse(str(FRONTEND_DIR / "index.html"))

"""Health check endpoints."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter

router = APIRouter()


@router.get("")
async def health_check():
    """System health summary."""
    return {
        "status": "healthy",
        "service": "lto",
        "version": "0.1.0",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "api": "healthy",
            "model": "loaded",
            "simulator": "available",
        },
    }


@router.get("/components")
async def component_health():
    """Per-component health status."""
    return {
        "api": {"status": "healthy", "uptime_s": 0},
        "ml_model": {"status": "loaded", "version": "v0.1"},
        "simulator": {"status": "available", "type": "synthetic"},
        "database": {"status": "configured", "type": "postgresql"},
    }

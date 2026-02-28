# LTO — Lithography Tradeoff Optimizer

> Tradeoff observability platform for computational lithography — predict, monitor, and optimize semiconductor manufacturing tradeoffs.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![License](https://img.shields.io/badge/license-MIT-purple)

## Overview

LTO provides an end-to-end platform for managing the critical tradeoffs in semiconductor lithography:

- **5 Tradeoff Dimensions** — Speed vs Accuracy, Resolution vs Depth of Focus, Cost vs Fidelity, Surrogate Reliability, Yield Risk
- **Ensemble ML Engine** — XGBoost + MLP (PyTorch) + Gaussian Process with calibrated 95% confidence intervals
- **Physics Simulator** — Synthetic lithography simulator modeling resolution, DoF, pattern fidelity, and yield
- **Real-time Dashboard** — Premium dark-themed UI with lithography machine backgrounds, Chart.js visualizations, and live API integration
- **Alert Engine** — 6 configurable rules with cooldowns, deduplication, and multi-channel dispatch
- **Prefect Orchestration** — Job pipeline: validate → ML preflight → gate check → simulate → evaluate → alert

## Quick Start

```bash
# Clone
git clone https://github.com/<your-username>/lithography-tradeoff-optimizer.git
cd lithography-tradeoff-optimizer

# Setup
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run
uvicorn lto.api.main:app --reload --port 8000

# Open dashboard
open http://localhost:8000
```

## Architecture

```
lto/
├── api/            # FastAPI REST API (health, jobs, tradeoffs)
├── ml/             # Ensemble ML engine
│   ├── models/     # XGBoost, MLP (PyTorch), Gaussian Process
│   ├── ensemble.py # Weighted ensemble aggregation
│   ├── drift/      # PSI-based drift detection
│   └── uncertainty/ # Multi-source uncertainty quantification
├── simulator/      # Synthetic lithography physics simulator
├── storage/        # PostgreSQL + SQLAlchemy models
├── orchestration/  # Prefect job pipeline flows
├── alerts/         # Alert engine + Slack channel
├── dashboard/      # Streamlit analytics dashboard
└── frontend/       # Static HTML/CSS/JS premium dashboard
    └── static/
        ├── index.html   # Single-page app (6 scrolling sections)
        ├── styles.css   # Dark glassmorphism design system
        ├── app.js       # Chart.js + API integration
        └── images/      # Lithography machine backgrounds
```

## Dashboard

The frontend is a single-page application inspired by [on.energy](https://www.on.energy/):

- **Hero** — Full-bleed EUV lithography machine background with floating stat cards
- **Tradeoff Overview** — Radar chart + bar chart against precision optics background
- **Predictor** — Interactive parameter sliders → real-time ensemble predictions
- **Submit Job** — Full pipeline: simulation outputs + tradeoff signals + ML confidence
- **Alerts** — 6 alert rule cards with severity badges
- **Health** — Live component status with performance counters

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | System health check |
| POST | `/api/v1/tradeoffs/predict` | Ensemble tradeoff prediction |
| POST | `/api/v1/jobs/submit` | Submit simulation job |
| GET | `/docs` | Interactive API documentation |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Uvicorn, SQLAlchemy |
| ML | XGBoost, PyTorch, scikit-learn |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Database | PostgreSQL |
| Orchestration | Prefect |
| Monitoring | Prometheus, Grafana |
| Infrastructure | Docker Compose |

## Testing

```bash
pytest tests/ -v
# 32 tests passing
```

## License

MIT

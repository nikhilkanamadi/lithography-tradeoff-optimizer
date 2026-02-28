# ARCHITECTURE.md
# LTO — Lithography Tradeoff Optimizer
## Technical Architecture Reference

> **Audience:** Engineers building on or contributing to LTO.
> **Scope:** High-level system design, low-level component design, data contracts, technology decisions, and operational patterns.

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Low-Level Architecture](#3-low-level-architecture)
4. [Open Source Technology Stack](#4-open-source-technology-stack)
5. [Data Architecture](#5-data-architecture)
6. [ML System Design](#6-ml-system-design)
7. [Orchestration Design](#7-orchestration-design)
8. [API Design](#8-api-design)
9. [Observability Design](#9-observability-design)
10. [Security & Deployment](#10-security--deployment)
11. [ADRs — Architecture Decision Records](#11-adrs--architecture-decision-records)

---

## 1. Design Principles

Every architectural decision in LTO flows from five non-negotiable principles.

### P1 — Domain Awareness Over Generic Monitoring

LTO is not a general-purpose monitoring tool. It understands semiconductor physics. It knows that a 1.5% accuracy drop in a surrogate model during EUV simulation has completely different implications than a 1.5% CPU spike. All components are designed with this domain specificity in mind.

### P2 — Fail Safe, Not Fail Fast

In a general software system, failing fast is good engineering. In a semiconductor context, the cost of a false negative (missing a real problem) vastly exceeds the cost of a false positive (unnecessary alert). LTO is calibrated to be conservative — it flags uncertainty rather than making confident wrong predictions.

### P3 — Modular Adapters, Stable Core

Real customers will have PROLITH, cuLitho, custom simulators, or some combination. The core ML, orchestration, and observability logic must be simulator-agnostic. All simulator-specific code lives in adapter modules. The core never imports an adapter directly.

### P4 — Observable Itself

A tradeoff observability system that is not itself observable is a contradiction. LTO exposes its own health metrics, model performance metrics, and pipeline health through the same observability stack it provides to customers.

### P5 — Replace Components Without Rewiring

DuckDB today, PostgreSQL tomorrow. Streamlit today, React tomorrow. Every component boundary is defined by a data contract (schema), not implementation. Replacing a component means implementing the same contract with a new technology.

---

## 2. High-Level Architecture

### System Context Diagram

```
╔══════════════════════════════════════════════════════════════════════╗
║                        EXTERNAL WORLD                                ║
║                                                                      ║
║  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  ║
║  │  PROLITH    │  │  cuLitho    │  │  GPU Cluster│  │  Wafer    │  ║
║  │  Simulator  │  │  Platform   │  │  (NVIDIA    │  │  Yield    │  ║
║  │  (KLA)      │  │  (NVIDIA)   │  │   H100s)    │  │  Database │  ║
║  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬─────┘  ║
╚═════════╪════════════════╪════════════════╪════════════════╪════════╝
          │                │                │                │
          └────────────────┴────────────────┴────────────────┘
                                    │
                    ════════════════╪════════════════
                           LTO SYSTEM BOUNDARY
                    ════════════════╪════════════════
                                    │
                         ┌──────────▼──────────┐
                         │                     │
                         │   LTO PLATFORM      │
                         │                     │
                         │  ┌───────────────┐  │
                         │  │  Ingestion    │  │
                         │  │  Layer        │  │
                         │  └───────┬───────┘  │
                         │          │          │
                         │  ┌───────▼───────┐  │
                         │  │  Intelligence │  │
                         │  │  Layer        │  │
                         │  └───────┬───────┘  │
                         │          │          │
                         │  ┌───────▼───────┐  │
                         │  │  Action       │  │
                         │  │  Layer        │  │
                         │  └───────────────┘  │
                         └──────────┬──────────┘
                                    │
          ┌─────────────────────────┼──────────────────────────┐
          │                         │                          │
   ┌──────▼──────┐          ┌───────▼──────┐         ┌────────▼──────┐
   │  Engineers  │          │  Managers /  │         │  External     │
   │  Dashboard  │          │  Executives  │         │  Systems      │
   │             │          │  Dashboard   │         │  (API)        │
   └─────────────┘          └──────────────┘         └───────────────┘
```

### Three-Layer Model

**Layer 1 — Ingestion Layer**
Receives data from all external sources. Normalizes it into LTO's internal schema. Buffers during high-load periods. Validates all incoming data against contracts before passing downstream. Nothing upstream of this layer is LTO's concern; nothing downstream of this layer knows what kind of simulator produced the data.

**Layer 2 — Intelligence Layer**
The computational core. Runs ML inference, computes tradeoff scores, detects drift, evaluates constraint violations, and maintains model health. All decisions are made here.

**Layer 3 — Action Layer**
Translates Intelligence Layer outputs into human-readable or machine-readable actions: alerts, dashboard updates, API responses, retraining triggers.

---

### Component Interaction Map

```
                    INGESTION LAYER
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Adapter Pool                    Ingestion API          │
│  ┌──────────┐  ┌──────────┐     ┌────────────────────┐ │
│  │ PROLITH  │  │ cuLitho  │────▶│  FastAPI           │ │
│  │ Adapter  │  │ Adapter  │     │  /ingest/job        │ │
│  └──────────┘  └──────────┘     │  /ingest/metrics    │ │
│  ┌──────────┐  ┌──────────┐     │  /ingest/yield      │ │
│  │ Synthetic│  │ Custom   │────▶└────────┬───────────┘ │
│  │ Adapter  │  │ Adapter  │              │             │
│  └──────────┘  └──────────┘     ┌────────▼───────────┐ │
│                                 │  Redis Streams     │ │
│                                 │  (message buffer)  │ │
│                                 └────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                         │
                    INTELLIGENCE LAYER
┌────────────────────────▼────────────────────────────────┐
│                                                         │
│  ┌─────────────────┐    ┌──────────────────────────┐   │
│  │  Simulation     │    │  ML Tradeoff Engine      │   │
│  │  Data Store     │───▶│                          │   │
│  │  (DuckDB)       │    │  ┌────────────────────┐  │   │
│  └─────────────────┘    │  │  Feature Engineer  │  │   │
│                         │  └────────┬───────────┘  │   │
│  ┌─────────────────┐    │           │              │   │
│  │  Time Series    │    │  ┌────────▼───────────┐  │   │
│  │  Store          │───▶│  │  Ensemble Model    │  │   │
│  │  (Prometheus +  │    │  │  XGBoost + MLP +   │  │   │
│  │   InfluxDB)     │    │  │  Gaussian Process  │  │   │
│  └─────────────────┘    │  └────────┬───────────┘  │   │
│                         │           │              │   │
│  ┌─────────────────┐    │  ┌────────▼───────────┐  │   │
│  │  Model Store    │───▶│  │  Uncertainty       │  │   │
│  │  (filesystem +  │    │  │  Quantification    │  │   │
│  │   metadata DB)  │    │  └────────┬───────────┘  │   │
│  └─────────────────┘    └──────────┼───────────────┘   │
│                                    │                    │
│                         ┌──────────▼──────────────────┐ │
│                         │  Orchestration Engine       │ │
│                         │  (Prefect + Ray)            │ │
│                         └──────────┬──────────────────┘ │
└────────────────────────────────────┼────────────────────┘
                         │
                    ACTION LAYER
┌────────────────────────▼────────────────────────────────┐
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Alert       │  │  Dashboard   │  │  Decision    │  │
│  │  Engine      │  │  Server      │  │  API         │  │
│  │              │  │  (Streamlit/ │  │  (FastAPI)   │  │
│  │  Alertmanager│  │   Grafana)   │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Low-Level Architecture

### 3.1 Simulation Data Generator — Internal Design

```python
# Module: lto/simulator/

lto/simulator/
├── __init__.py
├── base.py              # Abstract base class: SimulatorInterface
├── synthetic.py         # Physics-inspired synthetic simulator
├── prolith_adapter.py   # PROLITH COM/API adapter
├── culitho_adapter.py   # cuLitho metrics adapter
├── models/
│   ├── optical.py       # Optical physics models (Rayleigh, Abbe)
│   ├── resist.py        # Resist chemistry models
│   └── process.py       # Process window models
└── schemas.py           # Pydantic data contracts
```

**SimulatorInterface (base.py)**
```python
class SimulatorInterface(ABC):
    """
    All simulators — synthetic, PROLITH, cuLitho — implement this.
    The Intelligence Layer only speaks to this interface.
    """

    @abstractmethod
    def run_job(self, params: JobParameters) -> SimulationResult:
        """Execute one simulation job and return results."""
        ...

    @abstractmethod
    def stream_metrics(self) -> AsyncIterator[MetricSnapshot]:
        """Stream real-time metrics during job execution."""
        ...

    @abstractmethod
    def health_check(self) -> AdapterHealth:
        """Return adapter health status."""
        ...
```

**Physics Models (models/optical.py)**
```python
# Rayleigh resolution limit
def resolution(wavelength_nm: float, na: float, k1: float = 0.25) -> float:
    return k1 * wavelength_nm / na

# Depth of focus (Rayleigh criterion)
def depth_of_focus(wavelength_nm: float, na: float, k2: float = 0.5) -> float:
    return k2 * wavelength_nm / (na ** 2)

# Modulation Transfer Function (simplified)
def mtf(spatial_frequency: float, na: float, wavelength_nm: float) -> float:
    cutoff = 2 * na / wavelength_nm
    if spatial_frequency > cutoff:
        return 0.0
    return (2/π) * (arccos(spatial_frequency/cutoff)
                    - (spatial_frequency/cutoff)
                    * sqrt(1 - (spatial_frequency/cutoff)**2))
```

**Data Contract — JobParameters (schemas.py)**
```python
class JobParameters(BaseModel):
    job_id: str = Field(default_factory=lambda: f"job_{uuid4().hex[:8]}")
    na: float = Field(..., ge=0.1, le=0.55, description="Numerical aperture")
    wavelength_nm: float = Field(13.5, description="EUV=13.5, ArF=193, KrF=248")
    dose_mj_cm2: float = Field(..., ge=1.0, le=100.0)
    sigma: float = Field(0.8, ge=0.1, le=1.0, description="Partial coherence")
    resist_thickness_nm: float = Field(30.0, ge=5.0, le=200.0)
    grid_size_nm: float = Field(1.0, ge=0.1, le=10.0)
    pattern_complexity: PatternComplexity = PatternComplexity.MODERATE
    use_ai_surrogate: bool = True
    job_class: int = Field(1, ge=1, le=20, description="Job type classifier")
    priority: JobPriority = JobPriority.NORMAL
    constraint_profile: str = "default"
    submitted_by: str = "system"
    metadata: Dict[str, Any] = {}

class SimulationResult(BaseModel):
    job_id: str
    parameters: JobParameters
    outputs: SimulationOutputs
    tradeoff_signals: TradeoffSignals
    compute_stats: ComputeStats
    timestamp: datetime
    simulator_version: str
    adapter_type: str

class TradeoffSignals(BaseModel):
    speed_vs_accuracy: float = Field(..., ge=0.0, le=1.0)
    resolution_vs_dof: float = Field(..., ge=0.0, le=1.0)
    cost_vs_fidelity: float = Field(..., ge=0.0, le=1.0)
    surrogate_reliability: float = Field(..., ge=0.0, le=1.0)
    yield_risk: float = Field(..., ge=0.0, le=1.0)
    overall_health: float = Field(..., ge=0.0, le=1.0)
```

---

### 3.2 ML Tradeoff Engine — Internal Design

```
lto/ml/
├── __init__.py
├── engine.py            # TradeoffModel — main inference class
├── ensemble.py          # Ensemble coordinator
├── models/
│   ├── xgboost_model.py # Gradient boosting component
│   ├── mlp_model.py     # PyTorch MLP component
│   └── gp_model.py      # Gaussian Process uncertainty model
├── features/
│   ├── engineering.py   # Feature extraction + transforms
│   └── importance.py    # SHAP-based feature importance
├── uncertainty/
│   ├── quantifier.py    # Uncertainty aggregation
│   └── calibration.py   # Platt scaling calibration
├── drift/
│   ├── detector.py      # Statistical drift detection
│   └── retrainer.py     # Online retraining trigger
└── evaluation/
    ├── metrics.py        # Domain-specific evaluation metrics
    └── validator.py      # Production readiness validator
```

**Ensemble Inference Flow**

```
Input: JobParameters (12 features)
           │
           ▼
┌─────────────────────────┐
│  Feature Engineering    │
│                         │
│  physics_features()     │  Derive resolution, DoF, MTF
│  interaction_features() │  NA*dose, wavelength/NA, etc.
│  normalize()            │  StandardScaler per feature
└────────────┬────────────┘
             │
     ┌───────┴────────────┐
     │                    │
     ▼                    ▼
┌─────────┐          ┌─────────────────────────────┐
│ XGBoost │          │  PyTorch MLP                │
│ Model   │          │                             │
│         │          │  Input(12) → Dense(256)     │
│ 500     │          │  → BatchNorm → ReLU         │
│ trees   │          │  → Dropout(0.3)             │
│         │          │  → Dense(128) → ReLU        │
│ Outputs │          │  → Dense(64) → ReLU         │
│ 5 trade-│          │  → Dense(5 outputs)         │
│ off     │          │                             │
│ scores  │          │  Outputs: 5 tradeoff scores │
└────┬────┘          └─────────────┬───────────────┘
     │                             │
     └──────────┬──────────────────┘
                │
                ▼
┌───────────────────────────────┐
│  Gaussian Process Regressor   │
│  (Uncertainty layer)          │
│                               │
│  Input: XGBoost + MLP outputs │
│  Output: Prediction + 95% CI  │
│  + flag: high_uncertainty     │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  Ensemble Aggregation         │
│                               │
│  weights = [0.35, 0.40, 0.25] │ (XGBoost, MLP, GP)
│  final = weighted_average()   │
│  ci = gp_confidence_interval  │
│                               │
│  if uncertainty > threshold:  │
│    flag recommend_physics=True│
└───────────────────────────────┘
                │
                ▼
         TradeoffPrediction
```

**Drift Detection (drift/detector.py)**

```python
class DriftDetector:
    """
    Uses Population Stability Index (PSI) — the industry standard
    for detecting distribution shift in production ML models.
    PSI < 0.1  → No drift
    PSI 0.1-0.2 → Moderate drift, monitor closely
    PSI > 0.2  → Significant drift, retrain required
    """

    def compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        buckets: int = 10
    ) -> float:
        ...

    def check_all_features(
        self,
        reference_window: DataFrame,
        current_window: DataFrame
    ) -> DriftReport:
        """
        Returns per-feature PSI scores and overall drift verdict.
        """
        ...

    def check_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray
    ) -> PredictionDriftReport:
        """
        Monitors output distribution drift independently of input drift.
        """
        ...
```

---

### 3.3 Orchestration Engine — Internal Design

```
lto/orchestration/
├── __init__.py
├── flows/
│   ├── job_pipeline.py      # Main job execution flow
│   ├── monitoring.py        # Continuous monitoring flow
│   ├── retraining.py        # Model retraining flow
│   └── audit.py             # Compliance audit flow
├── tasks/
│   ├── validation.py        # Parameter validation tasks
│   ├── preflight.py         # Pre-flight tradeoff check tasks
│   ├── simulation.py        # Simulation execution tasks
│   ├── evaluation.py        # Result evaluation tasks
│   └── alerting.py          # Alert dispatch tasks
├── scheduler/
│   ├── job_queue.py         # Priority job queue
│   └── resource_manager.py  # GPU/CPU resource allocation
└── state/
    ├── job_state.py          # Job lifecycle state machine
    └── system_state.py       # System-wide health state
```

**Job Lifecycle State Machine**

```
                    SUBMITTED
                        │
                        ▼
                  VALIDATING ──────────────────────── REJECTED
                        │                           (param error)
                        ▼
                  PREFLIGHT ───────────────────────── BLOCKED
                        │                    (tradeoff violation)
                        ▼
                  QUEUED
                        │
                   (resource
                   available)
                        │
                        ▼
                  RUNNING ────────────────────────── FAILED
                        │                       (sim error)
                        ▼
                  EVALUATING
                        │
                 ┌───────┴────────┐
                 │                │
                 ▼                ▼
           COMPLETED         ANOMALY_FLAGGED
                                  │
                                  ▼
                             ENGINEER_REVIEW
                                  │
                       ┌──────────┴──────────┐
                       │                     │
                       ▼                     ▼
                  CLEARED              ESCALATED
```

**Core Prefect Flow (flows/job_pipeline.py)**

```python
@flow(name="lto-job-pipeline", retries=2, retry_delay_seconds=30)
async def job_pipeline_flow(job_params: JobParameters) -> JobResult:

    # 1. Validate parameters against schema and physical constraints
    validated = await validate_parameters(job_params)

    # 2. Pre-flight ML tradeoff prediction
    prediction = await preflight_tradeoff_check(validated)

    # 3. Gate: should we run this job?
    if prediction.recommend_block:
        await dispatch_alert(
            level=AlertLevel.WARN,
            job_id=job_params.job_id,
            reason=prediction.block_reason,
            recommendation=prediction.alternative_params
        )
        return JobResult.blocked(prediction)

    # 4. Execute simulation
    sim_result = await run_simulation(validated)

    # 5. Evaluate actual vs predicted tradeoffs
    evaluation = await evaluate_tradeoffs(
        predicted=prediction,
        actual=sim_result.tradeoff_signals
    )

    # 6. Alert if actual tradeoffs violated constraints
    if evaluation.has_violations:
        await dispatch_alert(
            level=AlertLevel.CRITICAL,
            job_id=job_params.job_id,
            violations=evaluation.violations
        )

    # 7. Store results and feed training buffer
    await store_results(sim_result, evaluation)
    await update_training_buffer(sim_result)

    return JobResult.completed(sim_result, evaluation)
```

---

### 3.4 API Design

```
lto/api/
├── main.py              # FastAPI app + lifespan
├── routers/
│   ├── jobs.py          # Job submission + status
│   ├── tradeoffs.py     # Tradeoff query + history
│   ├── models.py        # ML model management
│   ├── alerts.py        # Alert management
│   └── health.py        # System health
├── websockets/
│   └── streams.py       # Real-time metric streams
├── middleware/
│   ├── auth.py          # API key authentication
│   └── rate_limit.py    # Rate limiting
└── schemas/             # Request/response models
```

**REST API Endpoints**

```
POST   /api/v1/jobs/submit           Submit a new simulation job
GET    /api/v1/jobs/{job_id}         Get job status and results
GET    /api/v1/jobs/{job_id}/report  Get full tradeoff report
DELETE /api/v1/jobs/{job_id}         Cancel a queued job

POST   /api/v1/tradeoffs/predict     Predict tradeoffs for parameters (no simulation)
GET    /api/v1/tradeoffs/history     Query historical tradeoff data
GET    /api/v1/tradeoffs/trends      Tradeoff trend analysis

GET    /api/v1/models/current        Current deployed model info
POST   /api/v1/models/retrain        Trigger manual retraining
GET    /api/v1/models/performance    Model accuracy metrics

GET    /api/v1/alerts/active         List active unacknowledged alerts
POST   /api/v1/alerts/{id}/ack       Acknowledge an alert
GET    /api/v1/alerts/history        Alert history with resolution

GET    /api/v1/health                System health summary
GET    /api/v1/health/components     Per-component health

WS     /ws/metrics                   Real-time metric stream
WS     /ws/alerts                    Real-time alert stream
```

**Example Response — /api/v1/tradeoffs/predict**

```json
{
  "job_id": "pred_a3f8c2d1",
  "parameters": {
    "na": 0.33,
    "wavelength_nm": 13.5,
    "dose_mj_cm2": 15.2,
    "grid_size_nm": 1.0,
    "use_ai_surrogate": true
  },
  "predictions": {
    "speed_vs_accuracy": { "score": 0.84, "ci_low": 0.79, "ci_high": 0.89 },
    "resolution_vs_dof": { "score": 0.71, "ci_low": 0.65, "ci_high": 0.77 },
    "cost_vs_fidelity":  { "score": 0.76, "ci_low": 0.71, "ci_high": 0.81 },
    "surrogate_reliability": { "score": 0.91, "ci_low": 0.87, "ci_high": 0.95 },
    "yield_risk":        { "score": 0.12, "ci_low": 0.08, "ci_high": 0.17 },
    "overall_health":    { "score": 0.81, "ci_low": 0.76, "ci_high": 0.86 }
  },
  "uncertainty": {
    "high_uncertainty": false,
    "recommend_physics_simulation": false,
    "confidence_level": "HIGH"
  },
  "constraints": {
    "all_satisfied": true,
    "violations": []
  },
  "feature_importance": {
    "na": 0.31,
    "dose_mj_cm2": 0.24,
    "grid_size_nm": 0.18,
    "sigma": 0.14,
    "resist_thickness_nm": 0.13
  },
  "model_version": "v2.3",
  "inference_time_ms": 4.2
}
```

---

## 4. Open Source Technology Stack

### Full Stack Reference

```
┌────────────────────────────────────────────────────────────────┐
│ LAYER               TECHNOLOGY          VERSION   LICENSE       │
├────────────────────────────────────────────────────────────────┤
│ Language            Python              3.11+     PSF           │
├────────────────────────────────────────────────────────────────┤
│ SIMULATION                                                      │
│ Physics modeling    NumPy               1.26+     BSD           │
│ Scientific compute  SciPy               1.11+     BSD           │
│ Data manipulation   Pandas              2.1+      BSD           │
├────────────────────────────────────────────────────────────────┤
│ ML / AI                                                         │
│ Deep learning       PyTorch             2.1+      BSD           │
│ Gradient boosting   XGBoost             2.0+      Apache 2.0    │
│ Classical ML        scikit-learn        1.3+      BSD           │
│ Feature importance  SHAP                0.43+     MIT           │
│ Experiment tracking MLflow              2.8+      Apache 2.0    │
│ Hyperparameter opt  Optuna              3.4+      MIT           │
├────────────────────────────────────────────────────────────────┤
│ ORCHESTRATION                                                   │
│ Workflow engine     Prefect             3.0+      Apache 2.0    │
│ Distributed compute Ray                 2.8+      Apache 2.0    │
│ Task scheduling     APScheduler         3.10+     MIT           │
├────────────────────────────────────────────────────────────────┤
│ API                                                             │
│ REST framework      FastAPI             0.104+    MIT           │
│ ASGI server         Uvicorn             0.24+     BSD           │
│ Validation          Pydantic            2.5+      MIT           │
│ Auth                python-jose         3.3+      MIT           │
├────────────────────────────────────────────────────────────────┤
│ MESSAGING                                                       │
│ Message queue       Redis (Streams)     7.2+      BSD           │
│ Python client       redis-py            5.0+      MIT           │
├────────────────────────────────────────────────────────────────┤
│ STORAGE                                                         │
│ Analytics DB        DuckDB              0.9+      MIT           │
│ Time series         InfluxDB            2.7+      MIT           │
│ Metrics scraping    Prometheus          2.47+     Apache 2.0    │
│ Object storage      MinIO               (self-hosted S3)Apache  │
├────────────────────────────────────────────────────────────────┤
│ OBSERVABILITY                                                   │
│ Dashboards (MVP)    Streamlit           1.28+     Apache 2.0    │
│ Dashboards (prod)   Grafana             10.2+     AGPLv3        │
│ Alerting            Alertmanager        0.26+     Apache 2.0    │
│ Tracing             OpenTelemetry       1.21+     Apache 2.0    │
├────────────────────────────────────────────────────────────────┤
│ INFRASTRUCTURE                                                  │
│ Containerization    Docker              24+       Apache 2.0    │
│ Orchestration       Docker Compose      2.23+     Apache 2.0    │
│ (future) K8s        Kubernetes          1.28+     Apache 2.0    │
├────────────────────────────────────────────────────────────────┤
│ TESTING                                                         │
│ Test framework      pytest              7.4+      MIT           │
│ Mocking             pytest-mock         3.12+     MIT           │
│ Coverage            pytest-cov          4.1+      MIT           │
│ Property testing    Hypothesis          6.88+     MPL 2.0       │
│ Load testing        Locust              2.18+     MIT           │
└────────────────────────────────────────────────────────────────┘
```

### Technology Decision Rationale

**Why DuckDB over PostgreSQL?**
Computational lithography workloads are almost entirely analytical — batch inserts and complex aggregate queries over millions of simulation records. DuckDB is 10–100x faster for this workload pattern than row-oriented PostgreSQL, requires zero server setup, and runs embedded in the Python process. At the scale where DuckDB becomes a bottleneck (billions of records, concurrent writes from many nodes), a migration to ClickHouse or PostgreSQL is straightforward given our contract-based design.

**Why Prefect over Airflow?**
Airflow is battle-tested but requires a significant operational footprint and its DAG model is awkward for dynamic, parameter-driven workflows. Prefect's Python-native flow definitions are easier to write, test, and maintain. Its native async support is important for our high-throughput streaming use cases. Prefect Cloud is optional — we can self-host the server entirely.

**Why Ray for parallel compute?**
Ray's actor model maps cleanly onto our simulation job execution pattern. A simulation job is an actor that holds state (parameters, partial results) across multiple task calls. Ray handles fault tolerance, resource management, and cluster scaling transparently. Unlike Dask (which is array-oriented) or Celery (which is task-queue-oriented), Ray's general actor model fits our workload well.

**Why scikit-learn for Gaussian Process?**
The GP in our ensemble is used specifically for uncertainty quantification, not for primary prediction. scikit-learn's `GaussianProcessRegressor` is well-validated, easily calibrated, and interpretable. It does not scale to millions of training points — but it doesn't need to. The GP runs on a carefully selected subset of training data (representative samples, not all data). For larger scale, a sparse GP approximation (GPyTorch) is a documented upgrade path.

**Why SHAP for feature importance?**
SHAP (SHapley Additive exPlanations) provides consistent, theoretically grounded feature importance that works across all model types in our ensemble. This is not just for model debugging — in a semiconductor context, being able to tell an engineer *which* parameter is driving a tradeoff violation is directly actionable information. SHAP makes our model explainable to domain experts.

---

## 5. Data Architecture

### Storage Topology

```
                   ┌─────────────────────────────────────┐
                   │  DATA STORES                        │
                   │                                     │
                   │  ┌─────────────┐                    │
                   │  │  DuckDB     │  Simulation jobs,  │
                   │  │             │  tradeoff history, │
                   │  │  lto.duckdb │  model predictions │
                   │  └─────────────┘                    │
                   │                                     │
                   │  ┌─────────────┐                    │
                   │  │  Prometheus │  Real-time metrics │
                   │  │             │  (15s resolution,  │
                   │  │             │  30d retention)    │
                   │  └─────────────┘                    │
                   │                                     │
                   │  ┌─────────────┐                    │
                   │  │  InfluxDB   │  Long-term metrics │
                   │  │             │  (1m resolution,   │
                   │  │             │  2y retention)     │
                   │  └─────────────┘                    │
                   │                                     │
                   │  ┌─────────────┐                    │
                   │  │  MinIO /    │  Model artifacts,  │
                   │  │  Filesystem │  training datasets,│
                   │  │             │  audit exports     │
                   │  └─────────────┘                    │
                   └─────────────────────────────────────┘
```

### Core DuckDB Schema

```sql
-- Simulation jobs — master record
CREATE TABLE simulation_jobs (
    job_id          VARCHAR PRIMARY KEY,
    submitted_at    TIMESTAMP NOT NULL,
    completed_at    TIMESTAMP,
    status          VARCHAR NOT NULL,  -- see JobState enum
    submitted_by    VARCHAR,
    job_class       INTEGER,
    adapter_type    VARCHAR NOT NULL,  -- synthetic, prolith, culitho
    simulator_version VARCHAR,
    -- Parameters (denormalized for query performance)
    na              DOUBLE,
    wavelength_nm   DOUBLE,
    dose_mj_cm2     DOUBLE,
    sigma           DOUBLE,
    resist_thickness_nm DOUBLE,
    grid_size_nm    DOUBLE,
    use_ai_surrogate BOOLEAN,
    pattern_complexity VARCHAR,
    -- Raw outputs
    resolution_nm   DOUBLE,
    depth_of_focus_nm DOUBLE,
    pattern_fidelity DOUBLE,
    compute_time_s  DOUBLE,
    -- Tradeoff signals
    speed_vs_accuracy DOUBLE,
    resolution_vs_dof DOUBLE,
    cost_vs_fidelity  DOUBLE,
    surrogate_reliability DOUBLE,
    yield_risk        DOUBLE,
    overall_health    DOUBLE
);

-- ML predictions (pre-flight and post-hoc)
CREATE TABLE tradeoff_predictions (
    prediction_id   VARCHAR PRIMARY KEY,
    job_id          VARCHAR REFERENCES simulation_jobs(job_id),
    predicted_at    TIMESTAMP NOT NULL,
    prediction_type VARCHAR,  -- preflight, post_hoc, api_query
    model_version   VARCHAR NOT NULL,
    -- Predictions with confidence intervals
    pred_speed_vs_accuracy DOUBLE,
    ci_speed_low    DOUBLE,
    ci_speed_high   DOUBLE,
    pred_yield_risk DOUBLE,
    ci_yield_low    DOUBLE,
    ci_yield_high   DOUBLE,
    pred_overall_health DOUBLE,
    -- Uncertainty flags
    high_uncertainty BOOLEAN,
    recommend_physics BOOLEAN,
    inference_time_ms DOUBLE
);

-- Alerts
CREATE TABLE alerts (
    alert_id        VARCHAR PRIMARY KEY,
    fired_at        TIMESTAMP NOT NULL,
    acknowledged_at TIMESTAMP,
    resolved_at     TIMESTAMP,
    severity        VARCHAR NOT NULL,  -- INFO, WARN, CRITICAL, EMERGENCY
    job_id          VARCHAR,
    alert_type      VARCHAR NOT NULL,
    message         TEXT,
    recommendation  TEXT,
    acknowledged_by VARCHAR,
    was_actionable  BOOLEAN  -- engineer feedback
);

-- Model registry
CREATE TABLE model_registry (
    model_id        VARCHAR PRIMARY KEY,
    version         VARCHAR NOT NULL,
    trained_at      TIMESTAMP NOT NULL,
    deployed_at     TIMESTAMP,
    retired_at      TIMESTAMP,
    status          VARCHAR,  -- shadow, production, retired
    training_samples INTEGER,
    overall_accuracy DOUBLE,
    critical_accuracy DOUBLE,
    artifact_path   VARCHAR NOT NULL
);
```

---

## 6. ML System Design

### Model Versioning and Deployment Protocol

```
NEW DATA ACCUMULATED
        │
        ▼ (every 24h or on drift alert)
RETRAIN CANDIDATE MODEL
        │
        ▼
OFFLINE EVALUATION
  ├─ Accuracy on held-out test set ≥ 97.5%?
  ├─ Accuracy on critical feature jobs ≥ 98.5%?
  ├─ Uncertainty calibration within bounds?
  └─ Performance vs current production model?
        │
   ALL PASS?
        │
    YES ▼                    NO → DISCARD + LOG
SHADOW DEPLOYMENT
  ├─ New model runs in parallel with production
  ├─ Predictions logged but NOT used for decisions
  ├─ Minimum shadow period: 200 jobs
  └─ Shadow accuracy matches offline evaluation?
        │
   ALL PASS?
        │
    YES ▼                    NO → EXTEND SHADOW PERIOD
PROMOTE TO PRODUCTION
  ├─ Atomic swap — zero downtime
  ├─ Previous model becomes rollback candidate
  ├─ Rollback candidate retained for 30 days
  └─ Announcement to alert channel
```

### Uncertainty Quantification Details

The system uses a three-source uncertainty approach:

**Source 1 — Aleatory Uncertainty (data noise)**
Captured by training the MLP with MC Dropout. 100 forward passes with dropout active at inference time. Variance across passes = aleatory uncertainty.

**Source 2 — Epistemic Uncertainty (model uncertainty)**
Captured by the Gaussian Process regressor. GP variance is highest in regions of parameter space with sparse training data — exactly the cases where we should not trust the surrogate.

**Source 3 — Input Distribution Uncertainty**
If the incoming job parameters are outside the training distribution (as measured by Mahalanobis distance from training data centroid), uncertainty is elevated regardless of model confidence.

Final uncertainty = weighted combination of all three sources. Threshold tuning is done on validation data to minimize false alarms while maintaining zero missed critical cases.

---

## 7. Orchestration Design

### Prefect Flow Topology

```
                    ┌─────────────────┐
                    │  External       │
                    │  Trigger        │
                    │  (API / Cron /  │
                    │   Event)        │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────┐ ┌────────────────┐ ┌────────────────┐
│ job_pipeline_   │ │ monitoring_    │ │ retraining_    │
│ flow            │ │ flow           │ │ flow           │
│                 │ │                │ │                │
│ Triggered per   │ │ Runs every     │ │ Triggered by   │
│ job submission  │ │ 30 seconds     │ │ drift detector │
│                 │ │                │ │ or daily cron  │
│ Tasks:          │ │ Tasks:         │ │                │
│ - validate      │ │ - collect_gpu  │ │ Tasks:         │
│ - preflight     │ │ - compute_     │ │ - collect_data │
│ - run_sim       │ │   health       │ │ - retrain      │
│ - evaluate      │ │ - check_thres  │ │ - evaluate     │
│ - store         │ │ - alert_if_red │ │ - shadow       │
│ - alert_if_bad  │ │ - update_dash  │ │ - promote      │
└─────────────────┘ └────────────────┘ └────────────────┘
```

### Resource Management with Ray

```python
# GPU-aware job scheduling
@ray.remote(num_gpus=0.25)  # 4 jobs share one GPU
class SimulationWorker:
    def run(self, params: JobParameters) -> SimulationResult:
        ...

# CPU-only ML inference (fast, no GPU needed for inference)
@ray.remote(num_cpus=2)
class MLInferenceWorker:
    def predict(self, params: JobParameters) -> TradeoffPrediction:
        ...

# Resource pool configuration
ray.init(
    num_cpus=16,
    num_gpus=4,
    object_store_memory=8 * 1024**3  # 8 GB shared object store
)
```

---

## 8. API Design

### WebSocket Real-Time Stream Protocol

```
Client connects to: ws://lto-host/ws/metrics

Server pushes MetricFrame every 5 seconds:
{
  "timestamp": "2024-01-15T10:23:45Z",
  "frame_type": "metrics",
  "system_health": 0.84,
  "active_jobs": 47,
  "jobs_per_minute": 12.3,
  "tradeoff_scores": {
    "speed_vs_accuracy": { "current": 0.82, "trend": "stable" },
    "yield_risk": { "current": 0.11, "trend": "improving" },
    "compute_efficiency": { "current": 0.76, "trend": "degrading" }
  },
  "alerts_active": 1,
  "gpu_utilization": 0.73
}

Server pushes AlertFrame on alert events:
{
  "timestamp": "2024-01-15T10:24:01Z",
  "frame_type": "alert",
  "alert_id": "alert_f3a2b1c0",
  "severity": "WARN",
  "job_id": "job_4821",
  "message": "Surrogate drift detected: 3.2% (threshold: 1.5%)",
  "recommendation": "Force physics simulation for job class 5"
}
```

---

## 9. Observability Design

### What LTO Observes About Itself

LTO is instrumented with OpenTelemetry. The following metrics are exported to Prometheus:

```
# Simulation pipeline
lto_jobs_submitted_total          Counter  Total jobs submitted
lto_jobs_completed_total          Counter  Total jobs completed
lto_jobs_blocked_total            Counter  Jobs blocked by preflight
lto_job_duration_seconds          Histogram  End-to-end job time
lto_simulation_duration_seconds   Histogram  Simulation-only time

# ML model
lto_model_inference_duration_ms   Histogram  Inference time per prediction
lto_model_accuracy_score          Gauge    Current model accuracy vs ground truth
lto_model_uncertainty_ratio       Gauge    Fraction of predictions flagged high-uncertainty
lto_model_drift_psi               Gauge    Population Stability Index (drift metric)

# Tradeoff health
lto_tradeoff_health_score         Gauge    labels=[tradeoff_type]
lto_constraint_violations_total   Counter  labels=[constraint_type, severity]
lto_yield_risk_current            Gauge    Current yield risk estimate

# System
lto_api_request_duration_seconds  Histogram  labels=[endpoint, method]
lto_redis_queue_depth             Gauge    Messages waiting in queue
lto_alert_fired_total             Counter  labels=[severity, alert_type]
```

---

## 10. Security & Deployment

### Local Development

```bash
# Everything in Docker Compose
docker compose up -d

# Services exposed:
# LTO API:       http://localhost:8000
# Streamlit:     http://localhost:8501
# Prefect UI:    http://localhost:4200
# Grafana:       http://localhost:3000
# Prometheus:    http://localhost:9090
```

### Production Deployment Topology

```
KUBERNETES CLUSTER
┌────────────────────────────────────────────────────┐
│                                                    │
│  Namespace: lto-prod                               │
│                                                    │
│  ┌──────────────┐  ┌──────────────┐               │
│  │  lto-api     │  │  lto-worker  │               │
│  │  Deployment  │  │  Deployment  │               │
│  │  (3 replicas)│  │  (Ray cluster│               │
│  └──────────────┘  │   autoscale) │               │
│                    └──────────────┘               │
│  ┌──────────────┐  ┌──────────────┐               │
│  │  prefect-    │  │  lto-dash    │               │
│  │  server      │  │  (Streamlit  │               │
│  └──────────────┘  │   or Grafana)│               │
│                    └──────────────┘               │
│  ┌──────────────────────────────────┐             │
│  │  Persistent Volumes              │             │
│  │  PVC: duckdb-data (100GB)        │             │
│  │  PVC: influxdb-data (500GB)      │             │
│  │  PVC: model-artifacts (50GB)     │             │
│  └──────────────────────────────────┘             │
└────────────────────────────────────────────────────┘
```

### On-Premise for Enterprise Customers

All LTO services are packaged as container images with no external dependencies. Deployment inside a customer's air-gapped data center requires:

1. Container image export via `docker save`
2. Helm chart or Docker Compose file
3. Persistent storage provisioning (any POSIX filesystem)
4. Network access to customer's simulator (PROLITH/cuLitho) only

No cloud connectivity required. No customer data leaves the premises.

---

## 11. ADRs — Architecture Decision Records

### ADR-001 — Use contract-based adapters for all simulators

**Status:** Accepted

**Context:** We need to support synthetic simulation (development), PROLITH (KLA customers), and cuLitho (NVIDIA ecosystem customers) from day one.

**Decision:** All simulator integrations are implemented as adapters that satisfy `SimulatorInterface`. The core system never imports an adapter directly — it receives an adapter instance via dependency injection.

**Consequences:** Adding a new simulator requires only implementing one interface. The core system is never modified. Testing is straightforward — inject a mock adapter.

---

### ADR-002 — Gaussian Process for uncertainty, not Bayesian Neural Networks

**Status:** Accepted

**Context:** We need calibrated uncertainty estimates for safety-critical semiconductor decisions.

**Decision:** Use scikit-learn GaussianProcessRegressor as the uncertainty layer in our ensemble, rather than more complex Bayesian Neural Networks or Deep Ensembles.

**Consequences:** GP provides well-understood, theoretically grounded uncertainty estimates. It does not scale beyond ~10,000 training points — but our use case (uncertainty over parameter space, not over all data) works well within this limit. We document the upgrade path to GPyTorch sparse GPs for future scale.

---

### ADR-003 — DuckDB for primary data store

**Status:** Accepted

**Context:** Our primary query pattern is analytical — complex aggregations over millions of simulation records. We need zero-ops setup for MVP.

**Decision:** DuckDB as primary analytical store. SQLite for configuration and small metadata. No PostgreSQL in V1.

**Consequences:** Extremely fast analytical queries, zero server management. Limit: single-writer (we use Redis queue to serialize writes). Documented migration path to ClickHouse at 100M+ records.

---

### ADR-004 — Streamlit for MVP dashboard, Grafana for production

**Status:** Accepted

**Context:** We need a dashboard quickly for demos and pilots. We also need a production-grade solution.

**Decision:** Build MVP dashboards in Streamlit (fast to iterate, Python-native). Document migration to Grafana (or custom React) for production deployments requiring scale, SSO, and advanced embedding.

**Consequences:** Some duplicate effort in the Streamlit → Grafana migration. Accepted as the cost of fast MVP iteration.

---

*Document Version 1.0 — February 2026*
*LTO Technical Architecture Reference*
*For latest version, see repository root ARCHITECTURE.md*

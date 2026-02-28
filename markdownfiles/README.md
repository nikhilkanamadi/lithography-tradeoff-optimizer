# ⚡ LTO — Lithography Tradeoff Optimizer

<div align="center">

```
██╗  ████████╗ ██████╗
██║  ╚══██╔══╝██╔═══██╗
██║     ██║   ██║   ██║
██║     ██║   ██║   ██║
███████╗██║   ╚██████╔╝
╚══════╝╚═╝    ╚═════╝
Lithography Tradeoff Optimizer
```

**The world's first open-source tradeoff observability platform for computational lithography.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Prefect](https://img.shields.io/badge/Prefect-3.0-1749CC?style=flat-square)](https://prefect.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-Apache_2.0-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active_Development-orange?style=flat-square)]()

[Why LTO?](#-why-lto) · [Use Cases](#-use-cases) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [Components](#-components) · [Roadmap](#-roadmap)

</div>

---

## 🔭 The Problem Nobody Is Talking About

Semiconductor manufacturing at 2nm and below is the most complex industrial process humanity has ever attempted.

At its heart sits **computational lithography** — the software that tells light how to print circuit patterns onto silicon wafers. Tools like NVIDIA's cuLitho run on massive GPU clusters, making thousands of optimization decisions per second.

Every single decision is a **tradeoff**.

> *How much accuracy do we sacrifice for speed?*
> *When is the AI surrogate safe to trust?*
> *Which jobs need full physics simulation vs approximation?*
> *When is the system drifting toward dangerous territory?*

**These tradeoffs are invisible.** Engineers discover problems *after* the fact — after wasted compute, after degraded yield, after decisions made without full information.

A single bad lithography decision on a leading-edge node can corrupt an entire wafer batch worth **millions of dollars**.

**LTO makes tradeoffs visible, continuous, and actionable — before damage happens.**

---

## 💡 Why LTO?

```
BEFORE LTO                          AFTER LTO
────────────────────────────────    ────────────────────────────────
Engineers discover problems          Problems flagged before jobs run
after expensive wafer runs           
                                     
Tradeoff decisions made by          Tradeoff health scores visible
intuition and experience            in real time across all jobs

Audit is manual, periodic,          Continuous automated audit with
and incomplete                      confidence quantification

Management has zero visibility      Executive dashboard shows system
into system tradeoff health         health in plain language

Model drift goes undetected         Drift detection triggers retraining
until yield degrades                before production impact
```

### What Makes LTO Different

Unlike generic monitoring tools (Grafana, Datadog, Prometheus), LTO is **domain-aware**. It understands that a drop in simulation accuracy means something entirely different from a CPU spike. It knows the physics relationships between NA, wavelength, resolution, and depth of focus. It translates raw metrics into **semiconductor-meaningful tradeoff signals**.

---

## 🏭 Use Cases

### Use Case 1 — The $50M Wafer Run You Almost Ruined

**Scenario:** A leading-edge foundry is running night-shift lithography simulations for a new 2nm tape-out. A junior engineer configures a job with aggressive parameters — high NA, fine grid, AI surrogate enabled — to hit a deadline.

**Without LTO:** The job runs. The surrogate model has silently drifted 3.2% from physics ground truth due to a resist recipe change two weeks prior. Patterns print 6nm off target on 40% of critical features. 180 wafers are ruined before morning shift discovers the problem.

**With LTO:**
```
[22:47:03] WARN  Job #4821 — Pre-flight tradeoff check
           Surrogate accuracy drift: 3.2% (threshold: 1.5%)
           Cause: Resist parameter change detected 14 days ago
           Recommendation: Force physics simulation for this job type
           Estimated additional compute time: 4.2 hours
           Estimated yield risk if ignored: HIGH (0.83 score)

[22:47:03] ACTION  Job #4821 paused pending engineer approval
[22:47:09] SLACK   #litho-alerts → @on-call-engineer
```

The engineer reviews, confirms the recommendation, forces physics simulation. **180 wafers saved. $50M protected.**

---

### Use Case 2 — The GPU Cluster Burning Money While Nobody Watches

**Scenario:** A semiconductor company is paying $800K/month for GPU cluster time to run computational lithography workloads. Nobody knows if the cluster is being used efficiently.

**Without LTO:** Engineers run jobs at maximum fidelity because they're not sure what they can safely reduce. The cluster runs at 40% average utilization but 100% billing.

**With LTO:**
```
COMPUTE EFFICIENCY REPORT — Week of Jan 15
────────────────────────────────────────────
Total jobs run:              4,847
Jobs at max fidelity:        4,847  (100%)
Jobs where max fidelity      2,341  (48%)  ← these could run at 70% fidelity
was provably unnecessary:           with <0.1% accuracy impact

Estimated compute waste:     $312,000 this week
Recommended action:          Enable adaptive fidelity for job classes
                             3, 7, 11, 14
Projected monthly saving:    $890,000
```

**With adaptive fidelity enabled based on LTO recommendations:** Same quality outputs, 48% lower compute cost.

---

### Use Case 3 — The Model That Silently Stopped Working

**Scenario:** A chip company has deployed an ML surrogate model that runs 80x faster than physics simulation. It was validated 6 months ago. Nobody has checked it since.

**Without LTO:** The model has been gradually drifting as process conditions change. Its accuracy on the most critical pattern types is now 94.2% — well below the 98.5% production threshold. Nobody knows.

**With LTO:**
```
[MODEL DRIFT ALERT — January 14]

Surrogate model v2.3 accuracy degradation detected

Accuracy vs. physics ground truth:
  Overall:              97.8%  ✓  (threshold: 97.0%)
  Critical features:    94.2%  ✗  (threshold: 98.5%) ← VIOLATION
  Dense arrays:         96.1%  ✓
  Isolated lines:       99.1%  ✓

Degradation began:      ~December 28
Likely cause:           Resist batch change on Dec 26
Affected job types:     Classes 2, 5, 9 (critical feature layers)

Recommended action:     Retrain on post-Dec-26 data
Shadow model ready:     YES — v2.4 trained, accuracy 99.1%
Deploy v2.4?            Awaiting approval
```

Drift caught 17 days early. **Zero bad wafers. Zero missed deadlines.**

---

### Use Case 4 — The Audit That Used to Take 3 Weeks

**Scenario:** A semiconductor company's quality team needs to demonstrate to a customer that their computational lithography process is operating within specifications. This audit currently takes 3 weeks of manual data collection and analysis.

**With LTO:** The audit report is auto-generated in 4 minutes. Every tradeoff decision made in the last 90 days, fully documented with confidence intervals, anomaly flags, and engineer override logs.

---

### Use Case 5 — The New Engineer Who Doesn't Know What They Don't Know

**Scenario:** A new lithography engineer joins the team. They have ML knowledge but limited semiconductor domain experience. Without guardrails, they could configure jobs that look fine on paper but violate subtle process constraints.

**With LTO:**
```
[PRE-FLIGHT ADVISORY — Job submitted by: j.chen@company.com]

Parameter review for Job #5521:
  NA: 0.55                    ✓  Within range
  Dose: 22 mJ/cm²             ⚠  High for this resist type
                                  Typical range: 14-18 mJ/cm²
  Grid size: 0.5nm            ✓  Appropriate for feature size
  AI surrogate: enabled       ✓  Accuracy at 99.1% for this job class

Depth of focus prediction:    31nm  ⚠  Below recommended 40nm
                                       for this wafer topology

Recommendation: Reduce dose to 16 mJ/cm² — improves DoF to 47nm
                with negligible impact on pattern fidelity

Accept recommendation? [YES] [NO — run as configured] [EXPLAIN MORE]
```

The new engineer learns the system's tradeoffs **in context, on real jobs**, not in a training room.

---

## 🏗 Architecture

*See [ARCHITECTURE.md](ARCHITECTURE.md) for the complete technical deep-dive.*

### System at a Glance

```
                    ┌─────────────────────────────┐
                    │   DATA SOURCES               │
                    │  Synthetic Sim · PROLITH     │
                    │  cuLitho · GPU Metrics       │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   INGESTION + QUEUE          │
                    │   FastAPI + Redis Streams    │
                    └──────────┬──────────────────┘
                               │
          ┌────────────────────┼───────────────────┐
          │                    │                   │
┌─────────▼──────┐  ┌──────────▼──────┐  ┌────────▼────────┐
│  SIMULATION    │  │  ML TRADEOFF    │  │  TIME SERIES    │
│  DATA STORE    │  │  ENGINE         │  │  METRICS        │
│  DuckDB        │  │  PyTorch +      │  │  Prometheus +   │
│                │  │  scikit-learn   │  │  InfluxDB       │
└────────────────┘  └──────────┬──────┘  └────────┬────────┘
                               │                   │
                    ┌──────────▼───────────────────▼──────┐
                    │   ORCHESTRATION ENGINE               │
                    │   Prefect Workflows + Ray Compute   │
                    └──────────┬──────────────────────────┘
                               │
          ┌────────────────────┼───────────────────┐
          │                    │                   │
┌─────────▼──────┐  ┌──────────▼──────┐  ┌────────▼────────┐
│  ALERT         │  │  DASHBOARD      │  │  DECISION API   │
│  ENGINE        │  │  Streamlit /    │  │  FastAPI        │
│                │  │  Grafana        │  │  REST + WS      │
└────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.11+
python --version

# Docker + Docker Compose
docker --version
docker compose version
```

### Install

```bash
# Clone
git clone https://github.com/your-org/lto.git
cd lto

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Full Stack

```bash
# Start all services (Redis, Prometheus, Grafana)
docker compose up -d

# Start the LTO API
uvicorn lto.api.main:app --reload --port 8000

# Start the orchestration engine
prefect server start &
python -m lto.orchestration.flows

# Launch the dashboard
streamlit run lto/dashboard/app.py
```

### Run Your First Simulation

```python
from lto.simulator import LithographySimulator
from lto.ml import TradeoffModel
from lto.orchestrator import JobOrchestrator

# Initialize components
sim = LithographySimulator()
model = TradeoffModel.load("models/tradeoff_v1.pkl")
orchestrator = JobOrchestrator(simulator=sim, model=model)

# Submit a job
result = orchestrator.submit_job({
    "na": 0.33,
    "wavelength_nm": 13.5,
    "dose_mj_cm2": 15.2,
    "grid_size_nm": 1.0,
    "use_ai_surrogate": True
})

print(result.tradeoff_report())
# OUTPUT:
# ┌─────────────────────────────────────────┐
# │  TRADEOFF REPORT — Job #0001            │
# │  Speed vs Accuracy:     0.84  ✓ GOOD   │
# │  Resolution vs DoF:     0.71  ✓ GOOD   │
# │  Compute Efficiency:    0.76  ✓ GOOD   │
# │  Yield Risk:            0.12  ✓ LOW    │
# │  Overall Health:        0.81  ✓ SAFE   │
# └─────────────────────────────────────────┘
```

---

## 📦 Components

| Component | Description | Status |
|-----------|-------------|--------|
| `lto/simulator` | Synthetic lithography data generator | ✅ V0.1 |
| `lto/ml` | Ensemble ML tradeoff model + uncertainty quantification | ✅ V0.1 |
| `lto/orchestration` | Prefect workflow engine | 🔧 In Progress |
| `lto/api` | FastAPI REST + WebSocket service | 🔧 In Progress |
| `lto/dashboard` | Streamlit engineer + manager views | 📋 Planned |
| `lto/alerts` | Alert engine with Slack/email integration | 📋 Planned |
| `lto/adapters/prolith` | PROLITH data adapter | 📋 Planned |
| `lto/adapters/culitho` | cuLitho metrics adapter | 📋 Planned |

---

## 🗺 Roadmap

```
V0.1  ██████████  Synthetic simulator + basic ML model
V0.2  ████░░░░░░  Orchestration + alerts + API
V1.0  ░░░░░░░░░░  PROLITH integration + production dashboards
V2.0  ░░░░░░░░░░  cuLitho adapter + enterprise features
```

### V0.1 — Proof of Concept ✅
- [x] Synthetic lithography simulator with physics-inspired tradeoff relationships
- [x] XGBoost baseline tradeoff model
- [x] Basic tradeoff scoring and reporting

### V0.2 — MVP 🔧
- [ ] Ensemble ML model with Gaussian Process uncertainty quantification
- [ ] Prefect orchestration with pre-flight tradeoff checks
- [ ] FastAPI service
- [ ] Slack alert integration
- [ ] Docker Compose full stack

### V1.0 — Pilot Ready 📋
- [ ] PROLITH data adapter
- [ ] Engineer + manager Streamlit dashboards
- [ ] Automated model retraining pipeline
- [ ] Shadow deployment for new models
- [ ] On-premise deployment package

### V2.0 — Production 📋
- [ ] cuLitho metrics adapter
- [ ] GPU cluster resource optimization recommendations
- [ ] Multi-tenant support
- [ ] Full audit logging and compliance reporting
- [ ] Pattern complexity analysis module

---

## 🤝 Contributing

LTO is early-stage and actively looking for contributors with backgrounds in:

- **Computational lithography / EDA** — domain expertise to validate our tradeoff models
- **ML engineering** — improving model accuracy and uncertainty quantification
- **Systems engineering** — scaling the orchestration layer
- **Semiconductor industry** — real-world data and use case validation

See [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE)

---

## 🙏 Acknowledgements

- **Chris Mack** (lithoguru.com) — foundational lithography education and simulation methodology
- **KLA PROLITH** team — for making PROLITH available for academic research
- **NVIDIA cuLitho** team — for demonstrating what GPU-accelerated lithography compute can achieve
- The open source scientific Python ecosystem — NumPy, SciPy, PyTorch, scikit-learn

---

<div align="center">

**Built for the engineers keeping Moore's Law alive.**

*If you find LTO useful or want to collaborate, open an issue or reach out.*

</div>

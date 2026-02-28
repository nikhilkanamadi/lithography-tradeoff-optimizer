"""LTO Streamlit Dashboard.

Launch with:
    streamlit run lto/dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    st.set_page_config(
        page_title="LTO — Lithography Tradeoff Optimizer",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for a polished look
    st.markdown("""
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown("## 🔬 LTO Dashboard")
    view = st.sidebar.radio(
        "View",
        ["🛠️ Engineer View", "📊 Manager View", "🧪 Predict"],
        label_visibility="collapsed",
    )

    if view == "🛠️ Engineer View":
        engineer_view()
    elif view == "📊 Manager View":
        manager_view()
    else:
        predict_view()


def engineer_view():
    """Engineer view — tradeoff scores, sensitivity charts, recent jobs."""
    st.markdown('<div class="main-header">🛠️ Engineer View</div>', unsafe_allow_html=True)
    st.caption("Real-time tradeoff analysis and sensitivity exploration")

    # Generate sample data
    from lto.simulator.synthetic import SyntheticSimulator
    sim = SyntheticSimulator(seed=42)
    df = sim.generate_dataframe(200)

    # Tradeoff Health Overview
    st.subheader("📈 Tradeoff Health Distribution")
    col1, col2, col3, col4, col5 = st.columns(5)
    tradeoff_cols = [
        ("speed_vs_accuracy", "Speed vs Accuracy", col1),
        ("resolution_vs_dof", "Resolution vs DoF", col2),
        ("cost_vs_fidelity", "Cost vs Fidelity", col3),
        ("surrogate_reliability", "Surrogate Reliability", col4),
        ("yield_risk", "Yield Risk", col5),
    ]

    for col_name, label, col in tradeoff_cols:
        with col:
            mean_val = df[col_name].mean()
            delta = f"{df[col_name].std():.3f} σ"
            color = "normal" if (col_name != "yield_risk" and mean_val > 0.5) or (col_name == "yield_risk" and mean_val < 0.5) else "inverse"
            st.metric(label, f"{mean_val:.3f}", delta, delta_color=color)

    # Sensitivity Chart
    st.subheader("🎯 Parameter Sensitivity")
    param_x = st.selectbox("X-Axis Parameter", ["na", "dose_mj_cm2", "sigma", "wavelength_nm"])
    target_y = st.selectbox("Target Metric", ["overall_health", "yield_risk", "speed_vs_accuracy", "resolution_vs_dof"])

    chart_data = df[[param_x, target_y]].sort_values(param_x)
    st.scatter_chart(chart_data, x=param_x, y=target_y, height=300)

    # Recent Jobs Table
    st.subheader("📋 Recent Simulation Results")
    display_cols = ["na", "wavelength_nm", "dose_mj_cm2", "resolution_nm",
                    "depth_of_focus_nm", "overall_health", "yield_risk"]
    st.dataframe(
        df[display_cols].tail(20).style.background_gradient(
            subset=["overall_health"], cmap="RdYlGn"
        ).background_gradient(
            subset=["yield_risk"], cmap="RdYlGn_r"
        ),
        use_container_width=True,
    )


def manager_view():
    """Manager view — system health, yield trends, alerts."""
    st.markdown('<div class="main-header">📊 Manager View</div>', unsafe_allow_html=True)
    st.caption("System health overview and yield trending")

    from lto.simulator.synthetic import SyntheticSimulator
    sim = SyntheticSimulator(seed=42)
    df = sim.generate_dataframe(500)

    # System Health Score
    overall = df["overall_health"].mean()
    col1, col2, col3 = st.columns(3)
    with col1:
        score_pct = int(overall * 100)
        color = "🟢" if score_pct > 70 else "🟡" if score_pct > 50 else "🔴"
        st.metric(f"{color} System Health", f"{score_pct}/100")
    with col2:
        avg_yield_risk = df["yield_risk"].mean()
        st.metric("Avg Yield Risk", f"{avg_yield_risk:.3f}")
    with col3:
        high_risk_count = (df["yield_risk"] > 0.7).sum()
        st.metric("⚠️ High-Risk Jobs", f"{high_risk_count}/{len(df)}")

    # Yield Trend (simulated time series)
    st.subheader("📉 Yield Health Trend")
    window = 20
    df["rolling_health"] = df["overall_health"].rolling(window=window, min_periods=1).mean()
    df["rolling_risk"] = df["yield_risk"].rolling(window=window, min_periods=1).mean()
    st.line_chart(df[["rolling_health", "rolling_risk"]], height=250)

    # Health Distribution by Wavelength
    st.subheader("🔬 Health by Technology Node")
    for wl in [13.5, 193.0, 248.0]:
        subset = df[df["wavelength_nm"] == wl]
        if len(subset) > 0:
            wl_label = {13.5: "EUV", 193.0: "ArF DUV", 248.0: "KrF DUV"}[wl]
            health = subset["overall_health"].mean()
            st.progress(health, text=f"{wl_label} ({wl}nm): Health = {health:.2f}")


def predict_view():
    """Interactive prediction — enter parameters, get tradeoff prediction."""
    st.markdown('<div class="main-header">🧪 Tradeoff Predictor</div>', unsafe_allow_html=True)
    st.caption("Enter lithography parameters to predict tradeoff health")

    col1, col2 = st.columns(2)
    with col1:
        na = st.slider("Numerical Aperture (NA)", 0.10, 0.55, 0.33, 0.01)
        wavelength = st.selectbox("Wavelength (nm)", [13.5, 193.0, 248.0])
        dose = st.slider("Dose (mJ/cm²)", 1.0, 50.0, 15.0, 0.5)
        sigma = st.slider("Partial Coherence (σ)", 0.1, 1.0, 0.8, 0.05)
    with col2:
        resist = st.slider("Resist Thickness (nm)", 5.0, 100.0, 30.0, 1.0)
        grid = st.selectbox("Grid Size (nm)", [0.5, 1.0, 2.0, 5.0])
        surrogate = st.checkbox("Use AI Surrogate", value=True)
        complexity = st.selectbox("Pattern Complexity", ["simple", "moderate", "complex", "extreme"])

    if st.button("🔮 Predict Tradeoffs", type="primary", use_container_width=True):
        from lto.schemas import JobParameters
        from lto.simulator.synthetic import SyntheticSimulator

        params = JobParameters(
            na=na, wavelength_nm=wavelength, dose_mj_cm2=dose,
            sigma=sigma, resist_thickness_nm=resist, grid_size_nm=grid,
            use_ai_surrogate=surrogate, pattern_complexity=complexity,
        )

        # Run simulation
        sim = SyntheticSimulator(seed=None)
        result = sim.run_job(params)

        st.divider()
        st.subheader("📊 Results")

        # Output metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Resolution", f"{result.outputs.resolution_nm:.2f} nm")
        c2.metric("Depth of Focus", f"{result.outputs.depth_of_focus_nm:.1f} nm")
        c3.metric("Yield Prediction", f"{result.outputs.yield_prediction:.3f}")

        # Tradeoff signals
        st.subheader("⚖️ Tradeoff Signals")
        signals = {
            "Speed vs Accuracy": result.tradeoff_signals.speed_vs_accuracy,
            "Resolution vs DoF": result.tradeoff_signals.resolution_vs_dof,
            "Cost vs Fidelity": result.tradeoff_signals.cost_vs_fidelity,
            "Surrogate Reliability": result.tradeoff_signals.surrogate_reliability,
            "Overall Health": result.tradeoff_signals.overall_health,
        }
        for name, val in signals.items():
            color = "🟢" if val > 0.7 else "🟡" if val > 0.4 else "🔴"
            st.progress(val, text=f"{color} {name}: {val:.3f}")

        # Yield risk
        risk = result.tradeoff_signals.yield_risk
        risk_color = "🟢" if risk < 0.3 else "🟡" if risk < 0.7 else "🔴"
        st.metric(f"{risk_color} Yield Risk", f"{risk:.3f}",
                  delta="Low" if risk < 0.3 else "Medium" if risk < 0.7 else "HIGH",
                  delta_color="normal" if risk < 0.3 else "off" if risk < 0.7 else "inverse")


if __name__ == "__main__":
    main()

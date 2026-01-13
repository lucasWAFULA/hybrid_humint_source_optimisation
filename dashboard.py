# ======================================================
# GLOBAL CONFIGURATION & STYLING
# ======================================================
MODE = "streamlit"  # options: "streamlit", "api", "batch"

import streamlit as st
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import uuid
import hashlib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from pathlib import Path

# Consistent color scheme and behavior classes
COLORS = {
    "cooperative": "#10b981",
    "uncertain": "#f59e0b",
    "coerced": "#7c3aed",
    "deceptive": "#ef4444",
    "baseline": "#3b82f6",
    "neutral": "#6b7280"
}

BEHAVIOR_CLASSES = ["Cooperative", "Uncertain", "Coerced", "Deceptive"]

TASK_ROSTER = [f"Task {i + 1:02d}" for i in range(20)]

def render_kpi_indicator(title: str, value: float | None, *, reference: float | None = None,
                         suffix: str = "", note: str = "", height: int = 150, key: str | None = None):
    """Plotly-based KPI with hover, zoom, and optional delta comparison."""
    display_value = 0.0 if value is None else float(value)
    indicator_cfg = dict(
        mode="number+delta" if reference is not None else "number",
        value=display_value,
        number={"suffix": suffix, "font": {"size": 30, "color": "#1e3a8a"}},
        title={"text": title, "font": {"size": 12, "color": "#6b7280"}}
    )
    if reference is not None:
        indicator_cfg["delta"] = {
            "reference": reference,
            "valueformat": ".3f",
            "increasing": {"color": "#10b981"},
            "decreasing": {"color": "#ef4444"}
        }
    fig = go.Figure(go.Indicator(**indicator_cfg))
    if value is None:
        fig.add_annotation(text="Awaiting run", x=0.5, y=0.1, showarrow=False,
                           font=dict(size=11, color="#9ca3af"))
    elif note:
        fig.add_annotation(text=note, x=0.5, y=0.1, showarrow=False,
                           font=dict(size=11, color="#4b5563"))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=25, b=0),
                      paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True, key=key or f"kpi_{title}_{note}")

# ...existing _init_streamlit() function...
def _init_streamlit():
    """Initialize Streamlit config with enhanced typography and styling."""
    st.set_page_config(
        page_title="ML–TSSP HUMINT Tasking Dashboard",
        layout="wide",
        page_icon="🛰️"
    )

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Sans+3:wght@400;500;600&display=swap');
    
    * {
        font-family: 'Inter', 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    html, body {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: #1f2937;
        font-size: 15px;
    }
    
    .main {
        background-color: transparent;
    }
    
    h1 {
        font-size: 36px;
        font-weight: 700;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    
    h2 {
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.3px;
        line-height: 1.3;
    }
    
    h3 {
        font-size: 22px;
        font-weight: 600;
        letter-spacing: -0.2px;
    }
    
    h4 {
        font-size: 18px;
        font-weight: 600;
    }
    
    body, p {
        font-size: 15px;
        line-height: 1.6;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 700;
    }
    
    .metric-label {
        font-size: 14px;
        font-weight: 500;
        color: #6b7280;
    }
    
    .dashboard-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(30, 58, 138, 0.2);
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .dashboard-header h1 {
        margin: 0;
        font-size: 40px;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .dashboard-header p {
        margin: 0.8rem 0 0 0;
        font-size: 16px;
        opacity: 0.95;
    }
    
    .control-panel {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        position: sticky;
        top: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    .control-panel-header {
        font-size: 18px;
        font-weight: 600;
        color: #1e3a8a;
        margin-bottom: 1.2rem;
        padding-bottom: 0.8rem;
        border-bottom: 2px solid #3b82f6;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .metric-card:hover {
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.12);
        transform: translateY(-4px);
    }
    
    .metric-card.success {
        border-left-color: #10b981;
    }
    
    .metric-card.warning {
        border-left-color: #f59e0b;
    }
    
    .metric-card.danger {
        border-left-color: #ef4444;
    }
    
    .kpi-value {
        font-size: 32px;
        font-weight: 700;
        color: #1e3a8a;
        margin: 0.5rem 0;
    }
    
    .kpi-label {
        font-size: 14px;
        font-weight: 500;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .section-frame {
        background: white;
        border-radius: 15px;
        padding: 1.8rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid #e5e7eb;
        border-top: 4px solid #3b82f6;
    }
    
    .section-header {
        font-weight: 700;
        color: #1e3a8a;
        font-size: 22px;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.8rem;
        margin-top: 0;
        margin-bottom: 1.2rem;
        text-align: center;
        letter-spacing: -0.3px;
    }
    
    .chart-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    
    .chart-card-title {
        font-size: 16px;
        font-weight: 600;
        color: #1e3a8a;
        margin-bottom: 1rem;
        padding-bottom: 0.8rem;
        border-bottom: 2px solid #f0f0f0;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        padding: 1.2rem;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        border: 1px solid #bfdbfe;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        font-size: 15px;
    }
    
    .success-box {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        padding: 1.2rem;
        border-left: 4px solid #10b981;
        border-radius: 8px;
        border: 1px solid #a7f3d0;
        font-size: 15px;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        padding: 1.2rem;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        border: 1px solid #fde68a;
        font-size: 15px;
    }
    
    .error-box {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        padding: 1.2rem;
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        border: 1px solid #fecaca;
        font-size: 15px;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.9rem 1.8rem !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.25) !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
    }
    
    .stButton button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 20px rgba(30, 58, 138, 0.35) !important;
    }
    
    .stButton button:active {
        transform: translateY(-1px) !important;
    }
    
    [data-baseweb="tab"] {
        background: #e5e7eb !important;
        color: #1f2937 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 10px 10px 0 0 !important;
        padding: 0.7rem 1.3rem !important;
        margin-right: 0.25rem !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all 0.25s ease !important;
    }
    
    [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 6px 14px rgba(30, 58, 138, 0.25) !important;
        border-color: #1e3a8a !important;
    }
    
    [data-baseweb="tab"][aria-selected="false"] {
        background: #e5e7eb !important;
        color: #4b5563 !important;
    }
    
    [data-baseweb="tab"][aria-selected="false"]:hover {
        background: #dbeafe !important;
        color: #1e3a8a !important;
        box-shadow: 0 4px 10px rgba(59, 130, 246, 0.18) !important;
    }
    
    [data-testid="stExpander"] {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 10px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06) !important;
    }
    
    [data-testid="dataframe"] {
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        overflow: hidden;
        font-size: 14px;
    }
    
    hr {
        border: none;
        border-top: 2px solid #e5e7eb;
        margin: 2rem 0;
    }
    
    pre {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border-radius: 10px;
        padding: 1.2rem;
        color: #f3f4f6;
        border: 1px solid #374151;
        overflow-x: auto;
        font-size: 13px;
    }
    
    code {
        color: #f3f4f6;
        font-family: 'Courier New', monospace;
        font-size: 13px;
    }
    
    a {
        color: #3b82f6;
        text-decoration: none;
        font-weight: 600;
        transition: color 0.3s ease;
    }
    
    a:hover {
        color: #1e40af;
        text-decoration: underline;
    }
    
    caption {
        font-size: 13px;
        color: #6b7280;
    }
    </style>
    """, unsafe_allow_html=True)

# ...existing API helper functions...
try:
    from api import run_optimization as local_run_optimization
    from api import explain_source as local_explain_source
    USE_LOCAL_API = True
except ImportError:
    USE_LOCAL_API = False
    BACKEND_URL = "http://backend:8000"

def run_optimization(payload: dict):
    if USE_LOCAL_API:
        return local_run_optimization(payload)
    else:
        r = requests.post(f"{BACKEND_URL}/optimize", json=payload)
        r.raise_for_status()
        return r.json()

def request_shap_explanation(source_payload: dict):
    if USE_LOCAL_API:
        source_id = source_payload.get("source_id", "UNKNOWN")
        features = source_payload.get("features", {})
        
        shap_values = {}
        for behavior in BEHAVIOR_CLASSES:
            behavior_shap = {}
            
            tsr = float(features.get("task_success_rate", 0.5))
            cor = float(features.get("corroboration_score", 0.5))
            time = float(features.get("report_timeliness", 0.5))
            
            if behavior == "Cooperative":
                behavior_shap["task_success_rate"] = tsr * 0.3
                behavior_shap["corroboration_score"] = cor * 0.25
                behavior_shap["report_timeliness"] = time * 0.15
                behavior_shap["reliability_trend"] = (1 - tsr) * -0.05
            elif behavior == "Uncertain":
                behavior_shap["task_success_rate"] = (1 - tsr) * 0.2
                behavior_shap["corroboration_score"] = (1 - cor) * 0.25
                behavior_shap["report_timeliness"] = (1 - time) * 0.15
                behavior_shap["reliability_trend"] = abs(0.5 - tsr) * 0.2
            elif behavior == "Coerced":
                behavior_shap["corroboration_score"] = (1 - cor) * 0.3
                behavior_shap["task_success_rate"] = (1 - tsr) * 0.25
                behavior_shap["report_timeliness"] = (1 - time) * 0.2
                behavior_shap["consistency_volatility"] = abs(0.5 - cor) * 0.15
            elif behavior == "Deceptive":
                behavior_shap["corroboration_score"] = (1 - cor) * 0.35
                behavior_shap["task_success_rate"] = abs(0.7 - tsr) * 0.25
                behavior_shap["reliability_trend"] = (1 - tsr) * 0.2
                behavior_shap["consistency_volatility"] = (1 - cor) * 0.2
            
            shap_values[behavior] = behavior_shap
        
        return {"shap_values": shap_values}
    else:
        r = requests.post(f"{BACKEND_URL}/explain", json=source_payload)
        r.raise_for_status()
        return r.json()

def fetch_gru_drift(source_id: str):
    if USE_LOCAL_API:
        dates = [datetime.now() - timedelta(days=i) for i in range(10, 0, -1)]
        return [
            {"timestamp": d.isoformat(), "reliability": 0.6 + i*0.02, "deception": 0.3 - i*0.01}
            for i, d in enumerate(dates)
        ]
    else:
        r = requests.get(f"{BACKEND_URL}/drift/{source_id}")
        r.raise_for_status()
        return r.json()

def _decompose_risk(policy_data):
    """Risk decomposition by behavior class."""
    totals = {b: 0.0 for b in BEHAVIOR_CLASSES}
    for assignment in policy_data or []:
        probs = assignment.get("behavior_probs")
        costs = assignment.get("behavior_costs")
        if isinstance(probs, dict) and isinstance(costs, dict):
            for b in BEHAVIOR_CLASSES:
                p = float(probs.get(b, 0.0))
                c = float(costs.get(b, 0.0))
                totals[b] += p * c
        else:
            r = float(assignment.get("expected_risk", 0))
            totals["Cooperative"] += r * 0.20
            totals["Uncertain"] += r * 0.30
            totals["Coerced"] += r * 0.25
            totals["Deceptive"] += r * 0.25
    return totals

def compute_emv(policy_data):
    """Compute EMV from policy assignments."""
    emv = 0.0
    for assignment in policy_data or []:
        probs = assignment.get("behavior_probs")
        costs = assignment.get("behavior_costs")
        if isinstance(probs, dict) and isinstance(costs, dict):
            for b in BEHAVIOR_CLASSES:
                emv += float(probs.get(b, 0.0)) * float(costs.get(b, 0.0))
        else:
            emv += float(assignment.get("expected_risk", 0.0))
    return emv

def enforce_assignment_constraints(policy_data):
    """One task per source; randomize task assignment based on probabilities."""
    if not policy_data:
        return []
    seen_sources = set()
    tasks = TASK_ROSTER
    fixed = []
    rng = np.random.default_rng(42)
    
    for a in policy_data:
        sid = a.get("source_id")
        if sid in seen_sources:
            continue
        seen_sources.add(sid)
        new_a = dict(a)
        
        risk = float(a.get("expected_risk", 0.5))
        weights = np.array([1.0 / (1.0 + i * risk) for i in range(len(tasks))])
        weights = weights / weights.sum()
        
        new_a["task"] = rng.choice(tasks, p=weights)
        fixed.append(new_a)
    
    return fixed

# ======================================================
# DECISION INTELLIGENCE HELPER RENDERERS
# (moved above render_streamlit_app to avoid NameErrors)
# ======================================================
def _render_strategic_decision_section(sources, ml_policy, ml_emv, risk_reduction):
    st.markdown("""
    <div class="insight-box">
        <strong>📊 Optimization Complete!</strong> Key outcomes from the latest ML–TSSP run.
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_indicator("Total Sources", len(sources), note="All assigned", key="kpi_total_sources_tab0")
    with col2:
        render_kpi_indicator("Risk (EMV)", ml_emv, key="kpi_risk_tab0")
    with col3:
        render_kpi_indicator("Low Risk", len([a for a in ml_policy if a.get("expected_risk", 0) < 0.3]), key="kpi_low_risk_tab0")
    with col4:
        render_kpi_indicator("Improvement", risk_reduction, suffix="%", note="Vs baseline", key="kpi_improvement_tab0")
    st.divider()
    st.markdown("""
    <div class="success-box">
        <p style="margin:0;"><strong>Recommendation:</strong> Deploy ML–TSSP policy to minimize expected risk while maintaining coverage.</p>
    </div>
    """, unsafe_allow_html=True)

def _render_policy_framework_section(ml_policy, det_policy, uni_policy, ml_emv, det_emv, uni_emv):
    if not ml_policy:
        st.info("No ML–TSSP assignments yet. Run the optimizer to populate policy comparisons.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="chart-card"><div class="chart-card-title">📊 Task Distribution (ML–TSSP)</div>', unsafe_allow_html=True)
        task_counts = pd.Series([a.get("task", "Unassigned") for a in ml_policy]).value_counts()
        if task_counts.empty:
            st.warning("Nothing to display for task distribution.")
        else:
            fig = go.Figure(data=[go.Pie(labels=task_counts.index, values=task_counts.values, hole=.45)])
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True, key="policy_task_split")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="chart-card"><div class="chart-card-title">⚠️ Risk Distribution (ML–TSSP)</div>', unsafe_allow_html=True)
        bins = {"Low (<0.3)": 0, "Medium (0.3-0.6)": 0, "High (>0.6)": 0}
        for r in [a.get("expected_risk", 0) for a in ml_policy]:
            if r < 0.3:
                bins["Low (<0.3)"] += 1
            elif r < 0.6:
                bins["Medium (0.3-0.6)"] += 1
            else:
                bins["High (>0.6)"] += 1
        if not any(bins.values()):
            st.warning("Nothing to display for risk distribution.")
        else:
            fig = go.Figure(data=[go.Pie(labels=list(bins.keys()), values=list(bins.values()), hole=.45)])
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True, key="policy_risk_split")
        st.markdown('</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="chart-card"><div class="chart-card-title">🫧 Risk vs Coverage (All Policies)</div>', unsafe_allow_html=True)
    df = pd.DataFrame([
        {"Policy": "ML–TSSP", "Risk": float(ml_emv), "Coverage": len(set(a.get("task") for a in ml_policy)), "Sources": len(ml_policy)},
        {"Policy": "Deterministic", "Risk": float(det_emv), "Coverage": len(set(a.get("task") for a in det_policy)), "Sources": len(det_policy)},
        {"Policy": "Uniform", "Risk": float(uni_emv), "Coverage": len(set(a.get("task") for a in uni_policy)), "Sources": len(uni_policy)},
    ])
    bubble = px.scatter(df, x="Risk", y="Coverage", size="Sources", color="Policy")
    bubble.update_layout(height=360, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(bubble, use_container_width=True, key="policy_bubble")
    st.markdown('</div>', unsafe_allow_html=True)

def _render_optimal_policy_section(results):
    st.markdown('<div class="insight-box">Recommended ML–TSSP assignment details.</div>', unsafe_allow_html=True)
    policy = results.get("policies", {}).get("ml_tssp", [])
    if policy:
        st.dataframe(pd.DataFrame(policy))
        risk_levels = pd.Series(["Low" if a.get("expected_risk", 0) < 0.3 else "High" if a.get("expected_risk", 0) > 0.6 else "Medium" for a in policy]).value_counts()
        fig = go.Figure(data=[go.Pie(labels=risk_levels.index, values=risk_levels.values, hole=.45)])
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True, key="optimal_policy_risk")

def _render_baseline_policy_section(title, policy_key, results):
    st.markdown(f'<div class="insight-box">{title} breakdown.</div>', unsafe_allow_html=True)
    policy = results.get("policies", {}).get(policy_key, [])
    if policy:
        st.dataframe(pd.DataFrame(policy))
        risk_levels = pd.Series(["Low" if a.get("expected_risk", 0) < 0.3 else "High" if a.get("expected_risk", 0) > 0.6 else "Medium" for a in policy]).value_counts()
        fig = go.Figure(data=[go.Pie(labels=risk_levels.index, values=risk_levels.values, hole=.45)])
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True, key=f"{policy_key}_risk_split")

def _render_shap_section(num_sources):
    shap_columns = {
        "Cooperative": "task_success_rate",
        "Uncertain": "corroboration_score",
        "Coerced": "report_timeliness",
        "Deceptive": "reliability_trend"
    }
    shap_data = []
    shap_values = st.session_state.get("results", {}).get("explanations", {}).get("shap_values", {})
    for behavior, feature in shap_columns.items():
        for i in range(num_sources):
            src_id = f"SRC_{i + 1:03d}"
            shap_data.append({
                "Source ID": src_id,
                "Behavior Class": behavior,
                "Feature": feature,
                "SHAP Value": shap_values.get(behavior, {}).get(feature, 0.0)
            })
    st.dataframe(pd.DataFrame(shap_data))

def _render_evpi_section(ml_policy, uni_policy):
    ml_risk_map = {a.get("source_id"): float(a.get("expected_risk", 0)) for a in ml_policy}
    uni_risk_map = {a.get("source_id"): float(a.get("expected_risk", 0)) for a in uni_policy}
    evpi_rows = []
    for sid, ml_risk in ml_risk_map.items():
        uniform_risk = uni_risk_map.get(sid, ml_risk)
        evpi_val = max(0.0, uniform_risk - ml_risk)
        evpi_rows.append({"Source": sid, "EVPI": evpi_val, "Potential Gain": uniform_risk - ml_risk})
    evpi_df = pd.DataFrame(evpi_rows).sort_values("EVPI", ascending=False)
    k1, k2, k3 = st.columns(3)
    with k1:
        render_kpi_indicator("🔴 Max EVPI", evpi_df["EVPI"].max() if not evpi_df.empty else 0.0, key="kpi_evpi_max_exp")
    with k2:
        render_kpi_indicator("📊 Avg EVPI", evpi_df["EVPI"].mean() if not evpi_df.empty else 0.0, key="kpi_evpi_avg_exp")
    with k3:
        pct = (len(evpi_df[evpi_df["EVPI"] > evpi_df["EVPI"].quantile(0.75)]) / len(evpi_df) * 100) if len(evpi_df) else 0.0
        render_kpi_indicator("🎯 High-Value Sources", pct, suffix="%", key="kpi_evpi_high_value_exp")
    st.dataframe(evpi_df.reset_index(drop=True))

def _render_stress_section(ml_policy, ml_emv, det_emv, uni_emv, risk_reduction):
    st.markdown('<div class="insight-box">Stress testing controls and sensitivity outputs.</div>', unsafe_allow_html=True)
    st.metric("EMV (ML–TSSP)", f"{ml_emv:.3f}")
    st.metric("Risk Reduction vs Uniform", f"{risk_reduction:.1f}%")
    if st.button("▶ Execute Stress Test", key="stress_test_run_exp"):
        rel_vals = np.linspace(0.25, 0.75, 6)
        dec_vals = np.linspace(0.2, 0.8, 6)
        data = []
        for r in rel_vals:
            for d in dec_vals:
                emv = ml_emv + (abs(r - 0.45) * 0.3) + (abs(d - 0.5) * 0.2)
                data.append({"Reliability": r, "Deception": d, "EMV": emv})
        heat_df = pd.DataFrame(data)
        pivot = heat_df.pivot_table(values="EMV", index="Deception", columns="Reliability")
        fig = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="RdYlGn_r"))
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True, key="stress_heatmap_exp")
        st.dataframe(heat_df)

def _render_drift_section():
    st.info("📡 Drift monitoring module coming soon.")

HEADER_IMAGE_PATH = Path(r"D:\FINAL HUMINT DASH\background-logo.png")

def _load_header_background() -> str:
    try:
        with HEADER_IMAGE_PATH.open("rb") as fh:
            encoded = base64.b64encode(fh.read()).decode("utf-8")
            return (
                "linear-gradient(120deg, rgba(15,23,42,0.88), rgba(30,64,175,0.75)), "
                f"url('data:image/png;base64,{encoded}')"
            )
    except FileNotFoundError:
        return "linear-gradient(120deg, rgba(15,23,42,0.88), rgba(30,64,175,0.75))"

def render_streamlit_app():
    """Main Streamlit application with left-side controls."""
    _init_streamlit()
    
    # ======================================================
    # HEADER
    # ======================================================
    # ======================================================
    # HEADER
    # ======================================================
    hero_bg = _load_header_background()
    st.markdown(f"""
    <div style="
        position: relative;
        border-radius: 20px;
        margin-bottom: 2.5rem;
        overflow: hidden;
        box-shadow: 0 15px 40px rgba(15, 23, 42, 0.35);
        border: 2px solid rgba(255, 255, 255, 0.1);
        background-image:
            {hero_bg};
        background-size: cover;
        background-position: center;
    ">
        <div style="padding: 3.5rem 2.5rem; text-align: bottom; color: #f8fafc;">
            <h1 style="
                margin: 0;
                font-size: 35px;
                font-weight: 800;
                letter-spacing: -1px;
                text-shadow: 0 6px 16px rgba(0, 0, 0, 0.45);
            ">
                🛰️ Hybrid HUMINT Sources Optimization Engine
            </h1>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-frame">', unsafe_allow_html=True)
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #f4f7fb 0%, #e5e9f1 100%);
        border-radius: 14px;
        padding: 1.5rem 1.75rem;
        box-shadow: inset 0 1px 4px rgba(15, 23, 42, 0.08);">
        <p style="
            font-size:16px;
            margin:0;
            text-align:center;
            line-height:1.75;
            font-weight:500;
            color:#0F2A44;">
            Supports intelligence operations through a unified framework integrating XGBoost-based behavioral classification,
            GRU-driven forecasting of source reliability and deception, and two-stage stochastic optimization for risk-aware
            resource allocation.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ======================================================
    # OPERATIONAL OVERVIEW
    # ======================================================
    st.markdown("""
    <div style="
        background: linear-gradient(118deg, #0b1736 0%, #15306c 45%, #1d4ad1 100%);
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 8px 24px rgba(8, 15, 35, 0.55);
        padding: 1rem 1.7rem;
        margin: -1rem -1rem 1.5rem -1rem;
        border-radius: 0 0 16px 16px;">
        <p style="margin: 0; font-size: 11.5px; font-weight: 650; color: #e3e9ff; text-transform: uppercase; letter-spacing: 0.6px;">
            📊 Operational Overview
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if "sources_count" not in st.session_state:
        st.session_state.sources_count = 3
    if "results" not in st.session_state:
        st.session_state.results = None
    
    ov_col1, ov_col2, ov_col3, ov_col4 = st.columns(4)
    with ov_col1:
        render_kpi_indicator("🧠 Total Sources", st.session_state.get("sources_count", 0), key="kpi_total_sources_overview")
    with ov_col2:
        ml_policy = st.session_state.get("results", {}).get("policies", {}).get("ml_tssp", []) if st.session_state.get("results") else []
        avg_risk = np.mean([a.get("expected_risk", 0.5) for a in ml_policy]) if ml_policy else None
        render_kpi_indicator("📉 Avg Risk", avg_risk, suffix="", key="kpi_avg_risk_overview")
    with ov_col3:
        high_risk = sum(1 for a in ml_policy if a.get("expected_risk", 0) > 0.6) if ml_policy else None
        render_kpi_indicator("⚠️ High Risk", high_risk, note="Sources > 0.6 risk", key="kpi_high_risk_overview")
    with ov_col4:
        render_kpi_indicator("🎯 Tasks", len(TASK_ROSTER), note="Available slots", key="kpi_tasks_overview")
    
    st.divider()
    
    # ======================================================
    # TWO-COLUMN LAYOUT: LEFT CONTROLS + RIGHT CONTENT
    # ======================================================
    nav_labels = [
        "📋 Source Profiles",
        "📈 Policy Insights",
        "💰 EVPI Focus",
        "🔬 Stress Lab"
    ]
    nav_lookup = {
        "📋 Source Profiles": "profiles",
        "📈 Policy Insights": "policies",
        "💰 EVPI Focus": "evpi",
        "🔬 Stress Lab": "stress"
    }
    nav_choice = st.radio("Navigate dashboard", nav_labels, horizontal=True, key="nav_pills",
                          label_visibility="hidden")
    nav_key = nav_lookup[nav_choice]
    
    with st.container():
        filt1, filt2, filt3 = st.columns([1.2, 1, 1])
        with filt1:
            scenario_preset = st.selectbox(
                "Scenario preset",
                ["Baseline Ops", "High Threat", "Denied Terrain"],
                key="scenario_preset")
        with filt2:
            review_horizon = st.slider("Review horizon (days)", 14, 180, 60, key="review_horizon")
        with filt3:
            priority_tag = st.multiselect("Priority tags", ["SIGINT", "CI", "Liaison"], default=["SIGINT"],
                                          key="priority_tags")
        st.session_state["scenario_filters"] = {
            "preset": scenario_preset,
            "horizon": review_horizon,
            "tags": priority_tag
        }
    
    with st.sidebar:
        st.markdown("""
        <div class="control-panel">
            <div class="control-panel-header">⚙️ Configuration</div>
        """, unsafe_allow_html=True)
        st.markdown("<p class='kpi-label'>🧮 Simulation Scope</p>", unsafe_allow_html=True)
        num_sources = st.slider("Number of sources", 1, 80, st.session_state.sources_count,
                                 key="num_sources_slider")
        st.session_state.sources_count = num_sources
        source_ids = [f"SRC_{k + 1:03d}" for k in range(num_sources)]
        jump_source_id = st.selectbox(
            "Jump to source",
            source_ids,
            index=None,
            key="jump_source",
            placeholder="Type or select a source"
        )
        st.markdown("""<div class='warning-box'>
            <p style='margin:0;font-weight:600;'>📋 Scenario Summary</p>
            <p style='margin:0.2rem 0;'>Sources: <strong>{}</strong></p>
            <p style='margin:0.2rem 0;'>Review load: ~<strong>{}</strong></p>
            <p style='margin:0.2rem 0;'>Est. runtime: <strong>< 2s</strong></p>
        </div>""".format(num_sources, int(num_sources * 0.3)), unsafe_allow_html=True)
        st.markdown("<p class='kpi-label'>⚖️ Decision Thresholds</p>", unsafe_allow_html=True)
        rel_cols = st.columns(2)
        with rel_cols[0]:
            rel_disengage = st.slider("Reliability disengage", 0.0, 1.0, 0.35, 0.05,
                                      key="rel_disengage_slider")
        with rel_cols[1]:
            rel_ci_flag = st.slider("Reliability flag", 0.0, 1.0, 0.5, 0.05,
                                    key="rel_ci_flag_slider")
        dec_cols = st.columns(2)
        with dec_cols[0]:
            dec_disengage = st.slider("Deception reject", 0.0, 1.0, 0.75, 0.05,
                                      key="dec_disengage_slider")
        with dec_cols[1]:
            dec_ci_flag = st.slider("Deception escalate", 0.0, 1.0, 0.6, 0.05,
                                    key="dec_ci_flag_slider")
        st.session_state.recourse_rules = {
            "rel_disengage": float(rel_disengage),
            "rel_ci_flag": float(rel_ci_flag),
            "dec_disengage": float(dec_disengage),
            "dec_ci_flag": float(dec_ci_flag),
        }
        st.markdown("</div>", unsafe_allow_html=True)
    
    sources = []
    with st.expander("📋 Source Profiles & Tasking", expanded=(nav_key == "profiles")):
        st.markdown("""
        <div style="background:#ffffff;border-radius:15px;padding:1.8rem;
                    box-shadow:0 4px 15px rgba(0,0,0,0.08);border:1px solid #e5e7eb;
                    border-top:4px solid #3b82f6;">
            <h3 class="section-header" style="margin-top:0;">📋 Source Profiles</h3>
            <p style="text-align:center;color:#6b7280;font-size:13px;margin:0 0 1.2rem 0;">
                Configure attributes for optimization
            </p>
        """, unsafe_allow_html=True)
        source_selector_col, source_profile_col = st.columns([1.2, 2.8])
        # ========== LEFT PANEL: SOURCE SELECTOR CONSOLE ==========
        with source_selector_col:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 1rem; border-radius: 10px; border: 1px solid #e5e7eb; margin-bottom: 1rem; font-size: 12px;">
                <p style="margin: 0 0 0.8rem 0; font-weight: 700; color: #1e3a8a; text-transform: uppercase; letter-spacing: 0.5px;">📑 Source Selection</p>
            """, unsafe_allow_html=True)
            
            # Source selection as clickable list
            for i in range(min(num_sources, 10)):
                src_id = f"SRC_{i + 1:03d}"
                status_icon = "🟢" if np.random.random() > 0.4 else "🟡"
                risk_level = "Low" if np.random.random() > 0.6 else "Medium" if np.random.random() > 0.3 else "High"
                risk_color = "#10b981" if risk_level == "Low" else "#f59e0b" if risk_level == "Medium" else "#ef4444"
                
                try:
                    if st.session_state.get("results"):
                        ml_policy = st.session_state["results"].get("policies", {}).get("ml_tssp", [])
                        match = next((a for a in ml_policy if a.get("source_id") == src_id), None)
                        if match:
                            task_assign = str(match.get("task", "—"))
                        else:
                            task_assign = "—"
                    else:
                        task_assign = "—"
                except Exception:
                    task_assign = "—"
                
                # Clickable source card
                st.markdown(f"""
                <div style="background: white; border-left: 3px solid {risk_color}; padding: 0.7rem; border-radius: 8px; border: 1px solid #e5e7eb; margin-bottom: 0.6rem; cursor: pointer; transition: all 0.2s ease;" onmouseover="this.style.boxShadow='0 4px 12px rgba(59, 130, 246, 0.15)';" onmouseout="this.style.boxShadow='0 2px 6px rgba(0,0,0,0.06)';">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="flex: 1;">
                            <p style="margin: 0; font-size: 11px; font-weight: 700; color: #1e3a8a;">{src_id}</p>
                            <p style="margin: 0.2rem 0 0 0; font-size: 10px; color: #6b7280;">{status_icon} {risk_level} Risk</p>
                        </div>
                        <div style="text-align: right;">
                            <p style="margin: 0; font-size: 10px; font-weight: 600; color: #3b82f6;">{task_assign}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            if num_sources > 10:
                st.markdown(f"""
                <div style="background: #eff6ff; padding: 0.6rem; border-radius: 6px; text-align: center; font-size: 10px; color: #1e40af; font-weight: 600;">
                    +{num_sources - 10} more sources
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ========== RIGHT PANEL: SOURCE INTELLIGENCE PROFILE =========
        with source_profile_col:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #f1fdf8 100%); padding: 1rem; border-radius: 10px; border: 1px solid #bfdbfe; margin-bottom: 1rem;">
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.divider()
        
        # ========== SOURCE DETAIL PANELS (INTERACTIVE) ==========
        sources = []
        for i in range(num_sources):
            is_jump_target = source_ids[i] == jump_source_id if jump_source_id else (i == 0)
            
            with st.expander(f"🔹 {source_ids[i]} — Detailed Profile", expanded=is_jump_target):
                # ========== PROFILE HEADER WITH QUICK ACTIONS ==========
                profile_header_col1, profile_header_col2, profile_header_col3 = st.columns([2, 1, 1])
                
                with profile_header_col1:
                    st.markdown(f"""
                    <div>
                        <p style="margin: 0; font-size: 13px; font-weight: 700; color: #1e3a8a;">{source_ids[i]}</p>
                        <p style="margin: 0.3rem 0 0 0; font-size: 11px; color: #6b7280;">Source Intelligence Profile</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with profile_header_col2:
                    st.button("📋 Copy", key=f"copy_src_{i}", help="Copy source data", use_container_width=True)
                
                with profile_header_col3:
                    st.button("📊 Export", key=f"export_src_{i}", help="Export source report", use_container_width=True)
                
                st.divider()
                
                # ========== SOURCE ATTRIBUTE CONTROLS ==========
                rng = np.random.default_rng(i + 1)
                tsr_default = float(np.clip(rng.beta(5, 3), 0.0, 1.0))
                cor_default = float(np.clip(rng.beta(4, 4), 0.0, 1.0))
                time_default = float(np.clip(rng.beta(4, 4), 0.0, 1.0))

                gauge_cols = st.columns(3)
                with gauge_cols[0]:
                    st.markdown("**Competence Level**")
                    
                    fig_comp_mini = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=tsr_default * 100,
                        title={'text': "Task Success Rate %", 'font': {'size': 12, 'color': '#1e3a8a'}},
                        number={'suffix': "%", 'font': {'size': 16, 'color': '#1e3a8a'}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1.5, 'tickcolor': '#bfdbfe', 'tickfont': {'size': 9}},
                            'bar': {'color': COLORS['baseline'], 'thickness': 0.15},
                            'bgcolor': '#f0f9ff',
                            'borderwidth': 1.5,
                            'bordercolor': '#bfdbfe',
                            'steps': [
                                {'range': [0, 33], 'color': 'rgba(239, 68, 68, 0.12)'},
                                {'range': [33, 66], 'color': 'rgba(245, 158, 11, 0.12)'},
                                {'range': [66, 100], 'color': 'rgba(16, 185, 129, 0.12)'}
                            ],
                            'threshold': {
                                'line': {'color': '#ef4444', 'width': 2},
                                'thickness': 0.7,
                                'value': 50
                            }
                        }
                    ))
                    fig_comp_mini.update_layout(
                        height=200, 
                        margin=dict(l=5, r=5, t=35, b=5), 
                        paper_bgcolor='white', 
                        font=dict(size=10),
                        hovermode=False,
                        clickmode='event+select'
                    )
                    st.plotly_chart(fig_comp_mini, use_container_width=True, key=f'gauge_comp_interactive_{i}')
                    
                    tsr = st.number_input("Adjust Task Success Rate", 0.0, 1.0, tsr_default, step=0.05, key=f"tsr_input_{i}")
                with gauge_cols[1]:
                    st.markdown("**Reporting Consistency**")
                    
                    fig_cons_mini = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=cor_default * 100,
                        title={'text': "Corroboration Score %", 'font': {'size': 12, 'color': '#1e3a8a'}},
                        number={'suffix': "%", 'font': {'size': 16, 'color': '#1e3a8a'}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1.5, 'tickcolor': '#d1fae5', 'tickfont': {'size': 9}},
                            'bar': {'color': COLORS['cooperative'], 'thickness': 0.15},
                            'bgcolor': '#f0fdf4',
                            'borderwidth': 1.5,
                            'bordercolor': '#d1fae5',
                            'steps': [
                                {'range': [0, 33], 'color': 'rgba(239, 68, 68, 0.12)'},
                                {'range': [33, 66], 'color': 'rgba(245, 158, 11, 0.12)'},
                                {'range': [66, 100], 'color': 'rgba(16, 185, 129, 0.12)'}
                            ],
                            'threshold': {
                                'line': {'color': '#ef4444', 'width': 2},
                                'thickness': 0.7,
                                'value': 50
                            }
                        }
                    ))
                    fig_cons_mini.update_layout(
                        height=200, 
                        margin=dict(l=5, r=5, t=35, b=5), 
                        paper_bgcolor='white', 
                        font=dict(size=10),
                        hovermode=False,
                        clickmode='event+select'
                    )
                    st.plotly_chart(fig_cons_mini, use_container_width=True, key=f'gauge_cons_interactive_{i}')
                    
                    cor = st.number_input("Adjust Corroboration Level", 0.0, 1.0, cor_default, step=0.05, key=f"cor_input_{i}")
                with gauge_cols[2]:
                    st.markdown("**Report Timeliness**")
                    
                    fig_time_mini = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=time_default * 100,
                        title={'text': "Report Speed %", 'font': {'size': 12, 'color': '#1e3a8a'}},
                        number={'suffix': "%", 'font': {'size': 16, 'color': '#1e3a8a'}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1.5, 'tickcolor': '#fde68a', 'tickfont': {'size': 9}},
                            'bar': {'color': COLORS['uncertain'], 'thickness': 0.15},
                            'bgcolor': '#fffbeb',
                            'borderwidth': 1.5,
                            'bordercolor': '#fde68a',
                            'steps': [
                                {'range': [0, 33], 'color': 'rgba(239, 68, 68, 0.12)'},
                                {'range': [33, 66], 'color': 'rgba(245, 158, 11, 0.12)'},
                                {'range': [66, 100], 'color': 'rgba(16, 185, 129, 0.12)'}
                            ],
                            'threshold': {
                                'line': {'color': '#ef4444', 'width': 2},
                                'thickness': 0.7,
                                'value': 50
                            }
                        }
                    ))
                    fig_time_mini.update_layout(
                        height=200, 
                        margin=dict(l=5, r=5, t=35, b=5), 
                        paper_bgcolor='white', 
                        font=dict(size=10),
                        hovermode=False,
                        clickmode='event+select'
                    )
                    st.plotly_chart(fig_time_mini, use_container_width=True, key=f'gauge_time_interactive_{i}')
                    
                    time = st.number_input("Adjust Report Speed", 0.0, 1.0, time_default, step=0.05, key=f"time_input_{i}")
                
                st.markdown("**60-Day Reliability Forecast**")
                st.caption("Expanded horizon to observe medium-term reliability trajectory (60 periods).")
                
                periods = 60
                rng_forecast = np.random.default_rng(10_000 + i)
                base_rel = np.clip(0.35 + 0.25 * tsr + 0.20 * cor + 0.15 * time, 0.2, 0.9)
                drift = 0.012 + 0.006 * rng_forecast.normal()
                reliability_ts = [np.clip(base_rel + drift * j + rng_forecast.normal(0, 0.02), 0.25, 0.98) for j in range(periods)]
                
                window = 7
                rel_ma = []
                for j in range(len(reliability_ts)):
                    start_idx = max(0, j - window + 1)
                    window_vals = reliability_ts[start_idx:j + 1]
                    rel_ma.append(np.mean(window_vals))
                
                rel_df = pd.DataFrame({
                    'period': range(periods),
                    'reliability': reliability_ts,
                    'ma': rel_ma,
                    'upper': [min(r + 0.1, 1.0) for r in reliability_ts],
                    'lower': [max(r - 0.1, 0.0) for r in reliability_ts]
                })
                
                fig_rel = go.Figure()
                fig_rel.add_trace(go.Scatter(x=rel_df['period'], y=rel_df['reliability'], mode='lines+markers', name='Predicted', line=dict(color=COLORS['baseline'], width=2.5), marker=dict(size=7), hovertemplate='<b>Period %{x}</b><br>Reliability: %{y:.2f}<extra></extra>'))
                fig_rel.add_trace(go.Scatter(x=rel_df['period'], y=rel_df['ma'], mode='lines', name='Moving Avg (3)', line=dict(color=COLORS['cooperative'], width=2.5, dash='dash'), hovertemplate='<b>Period %{x}</b><br>MA: %{y:.2f}<extra></extra>'))
                fig_rel.add_trace(go.Scatter(x=rel_df['period'], y=rel_df['upper'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                fig_rel.add_trace(go.Scatter(x=rel_df['period'], y=rel_df['lower'], mode='lines', line=dict(width=0), fillcolor='rgba(59, 130, 246, 0.2)', fill='tonexty', showlegend=False, hoverinfo='skip', name='Confidence'))
                fig_rel.add_hline(y=0.5, line_dash='dash', line_color=COLORS['deceptive'], opacity=0.6, annotation_text="Risk Threshold")
                fig_rel.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='#f0f9ff', plot_bgcolor='#f8fafc', xaxis_title='Period', yaxis_title='Reliability', showlegend=True, font=dict(size=10), hovermode='x unified', dragmode='zoom')
                st.plotly_chart(fig_rel, use_container_width=True, key=f'rel_chart_{i}')
                
                st.divider()
                
                avg_rel = np.mean(reliability_ts)
                deception_risk = 1.0 - cor
                
                # ========== SUMMARY METRICS (STREAMLINED) ==========
                st.markdown('<h4 style="color: #1e3a8a; margin: 0.5rem 0 1rem 0;">📊 Assessment Summary</h4>', unsafe_allow_html=True)
                
                # Three-column layout with key metrics only
                met_col1, met_col2, met_col3 = st.columns(3)
                
                with met_col1:
                    st.metric(
                        "🎯 Reliability",
                        f"{avg_rel:.2f}",
                        delta=f"{(avg_rel - 0.5) * 100:+.0f}%" if avg_rel >= 0.5 else f"{(avg_rel - 0.5) * 100:.0f}%",
                        delta_color="normal"
                    )
                
                with met_col2:
                    st.metric(
                        "⚠️ Risk Level",
                        "High" if deception_risk > 0.6 else "Med" if deception_risk > 0.3 else "Low",
                        delta=f"{deception_risk:.2f}",
                        delta_color="inverse"
                    )
                
                with met_col3:
                    assigned_task_display = "—"
                    try:
                        if st.session_state.get("results"):
                            ml_assignments = st.session_state["results"].get("policies", {}).get("ml_tssp", [])
                            my_id = f"SRC_{i + 1:03d}"
                            match = next((a for a in ml_assignments if a.get("source_id") == my_id), None)
                            if match:
                                assigned_task_display = str(match.get("task") or "—")
                    except Exception:
                        pass
                    st.metric(
                        "📋 Assignment",
                        assigned_task_display,
                        help="ML-TSSP optimized task"
                    )
                
                st.divider()
                
                # ========== INLINE INSIGHT PANEL ==========
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {'#f0fdf4' if avg_rel > 0.6 else '#fffbeb' if avg_rel > 0.4 else '#fef2f2'} 0%, {'#d1fae5' if avg_rel > 0.6 else '#fef3c7' if avg_rel > 0.4 else '#fee2e2'} 100%); 
                            padding: 0.8rem; 
                            border-radius: 8px; 
                            border-left: 3px solid {'#10b981' if avg_rel > 0.6 else '#f59e0b' if avg_rel > 0.4 else '#ef4444'};">
                    <p style="margin: 0; font-size: 11px; font-weight: 600; color: {'#15803d' if avg_rel > 0.6 else '#92400e' if avg_rel > 0.4 else '#991b1b'}; text-transform: uppercase;">
                        {'✅ Recommended' if avg_rel > 0.6 else '⚠️ Review Recommended' if avg_rel > 0.4 else '❌ High Risk'}
                    </p>
                    <p style="margin: 0.4rem 0 0 0; font-size: 10px; color: #1f2937;">
                        {'Reliable source with strong performance indicators' if avg_rel > 0.6 else 'Moderate reliability—consider enhanced monitoring' if avg_rel > 0.4 else 'Low reliability—prioritize for validation'}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                features = {
                    "task_success_rate": float(tsr),
                    "corroboration_score": float(cor),
                    "report_timeliness": float(time)
                }

                sources.append({
                    "source_id": f"SRC_{i + 1:03d}",
                    "features": features,
                    "reliability_series": reliability_ts,
                    "recourse_rules": {}
                })

        # ========== RUN OPTIMIZATION COMMAND PANEL ==========
        st.markdown('<div class="section-frame">', unsafe_allow_html=True)
        
        st.markdown("""
        <h3 style="font-weight: 700; color: #1e3a8a; font-size: 20px; margin: 0 0 0.4rem 0;">
            🧠 Decision Optimization Engine
        </h3>
        <p style="margin: 0; font-size: 12px; color: #6b7280;">Configure parameters and execute the ML–TSSP engine</p>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # ========== OPTIMIZATION CONTROL PANEL ==========
        st.markdown('<h4 style="color: #1e3a8a; margin-bottom: 1rem;">🧪 Optimization Control Panel</h4>', unsafe_allow_html=True)
        
        col_run, col_reset = st.columns([2, 1])
        with col_run:
            run_button_right = st.button("▶ Execute Optimization", type="primary", use_container_width=True, key="run_opt_btn_right", help="Execute ML–TSSP with current configuration")
        with col_reset:
            reset_button_right = st.button("↺ Reset Configuration", use_container_width=True, key="reset_btn_right", help="Clear configuration and results")
        
        if reset_button_right:
            st.session_state.results = None
            st.rerun()
        
        st.divider()
        
        # ========== EXECUTION FEEDBACK & STATUS CONSOLE ==========
        if st.session_state.results is None:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #f1fdf8 100%); padding: 1.5rem; border-radius: 12px; border: 2px dashed #bfdbfe; text-align: center;">
                <p style="margin: 0; font-size: 14px; color: #1e3a8a; font-weight: 600;">⏳ Ready for Optimization</p>
                <p style="margin: 0.5rem 0 0 0; font-size: 12px; color: #6b7280;">Click <strong>Execute Optimization</strong> to run the ML–TSSP algorithm</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); padding: 1.5rem; border-radius: 12px; border: 2px solid #a7f3d0;">
                <p style="margin: 0; font-size: 14px; color: #15803d; font-weight: 600;">✅ Optimization Complete</p>
                <p style="margin: 0.5rem 0 0 0; font-size: 12px; color: #1f2937;">Results ready for analysis. Review decision summary below.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # ========== EXECUTIVE DECISION SUMMARY ==========
        st.markdown('<h4 style="color: #1e3a8a; margin-bottom: 1rem;">📊 Executive Decision Summary</h4>', unsafe_allow_html=True)
        
        if st.session_state.results is None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sources Configured", len(sources))
            with col2:
                st.metric("Expected Risk", "—")
            with col3:
                st.metric("Improvement vs Uniform", "—")
        else:
            results = st.session_state.results
            ml_emv = results.get("emv", {}).get("ml_tssp", 0)
            uni_emv = results.get("emv", {}).get("uniform", 0)
            risk_reduction = ((uni_emv - ml_emv) / uni_emv * 100) if uni_emv > 0 else 0.0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                render_kpi_indicator("Total Sources", len(sources), note="All assigned", key="kpi_total_sources_exec")
            with col2:
                render_kpi_indicator("Risk (EMV)", ml_emv, reference=uni_emv, note="vs Uniform", key="kpi_risk_exec")
            with col3:
                low_risk = sum(1 for a in results.get("policies", {}).get("ml_tssp", []) if a.get("expected_risk", 0) < 0.3)
                render_kpi_indicator("Low Risk Sources", low_risk, suffix=f" / {len(sources)}", key="kpi_low_risk_exec")
            with col4:
                render_kpi_indicator("Improvement", risk_reduction, suffix="%", note="Vs baseline", key="kpi_improvement_exec")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ========== RUN OPTIMIZATION EXECUTION ==========
        if run_button_right:
            payload = {
                "sources": sources,
                "seed": 42
            }

            try:
                with st.spinner("🔄 Running optimization…"):
                    result = run_optimization(payload)
                    if isinstance(result, dict) and isinstance(result.get("policies"), dict):
                        for pkey in ["ml_tssp", "deterministic", "uniform"]:
                            plist = result["policies"].get(pkey) or []
                            fixed = enforce_assignment_constraints(plist)
                            result["policies"][pkey] = fixed
                            result.setdefault("emv", {})[pkey] = compute_emv(fixed)
                    st.session_state.results = result
                    st.session_state.sources = sources
                st.success("✅ Optimization complete! Review decision summary above.")
                st.session_state.show_results_popup = True
                st.rerun()
            except Exception as e:
                st.error(f"❌ Optimization failed: {e}")

    # ======================================================
    # DECISION INTELLIGENCE SUITE (MAIN ANALYTICS)
    # ======================================================
    results = st.session_state.results

    if results is not None:
        st.markdown('<div class="section-frame">', unsafe_allow_html=True)
        st.markdown("""<h3 class="section-header">🏆 Decision Intelligence Suite</h3>
        <p style="text-align:center;color:#6b7280;font-size:13px;margin:0 0 1rem 0;">
            Comprehensive analysis of ML–TSSP optimization results with policy comparisons and sensitivity assessments.
        </p>""", unsafe_allow_html=True)
        ml_policy = results.get("policies", {}).get("ml_tssp", [])
        det_policy = results.get("policies", {}).get("deterministic", [])
        uni_policy = results.get("policies", {}).get("uniform", [])
        ml_emv = results.get("emv", {}).get("ml_tssp", 0)
        det_emv = results.get("emv", {}).get("deterministic", 0)
        uni_emv = results.get("emv", {}).get("uniform", 0)
        risk_reduction = ((uni_emv - ml_emv) / uni_emv * 100) if uni_emv > 0 else 0.0

        with st.expander("⚙️ Strategic Decision Optimization", expanded=(nav_key == "profiles")):
            _render_strategic_decision_section(sources, ml_policy, ml_emv, risk_reduction)
        with st.expander("📈 Policy Framework Comparison", expanded=(nav_key == "policies")):
            _render_policy_framework_section(ml_policy, det_policy, uni_policy, ml_emv, det_emv, uni_emv)
        with st.expander("🏆 ML–TSSP Optimal Policy (Recommended)"):
            _render_optimal_policy_section(results)
        with st.expander("📐 Deterministic Policy (Baseline)"):
            _render_baseline_policy_section("Deterministic Policy (Baseline)", "deterministic", results)
        with st.expander("📊 Uniform Allocation (Baseline)"):
            _render_baseline_policy_section("Uniform Allocation Policy", "uniform", results)
        with st.expander("🧠 SHAP Explanations"):
            _render_shap_section(num_sources)
        with st.expander("💰 Expected Value of Perfect Information (EVPI)", expanded=(nav_key == "evpi")):
            _render_evpi_section(ml_policy, uni_policy)
        with st.expander("🔬 Behavioral Uncertainty & Stress Analysis (What-If)", expanded=(nav_key == "stress")):
            _render_stress_section(ml_policy, ml_emv, det_emv, uni_emv, risk_reduction)
        with st.expander("📡 Source Drift Monitoring (Reliability & Deception)"):
            _render_drift_section()
        st.markdown('</div>', unsafe_allow_html=True)

if MODE == "streamlit":
	render_streamlit_app()
elif __name__ == "__main__":
	render_streamlit_app()
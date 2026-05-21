"""
Entry point for the AI Recommendations module — Premium Dashboard Edition.

Merged:
- Script 1's overall structure, CSS, and PowerBI-style AI output
- Script 2's clickable model card selection logic
- Fixed pipeline plan review so step selection actually works
- Fixed execution to use the reviewed (filtered) plan
"""

import streamlit as st
import pandas as pd
import io
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, Any, Optional
import re
import warnings
import requests
import threading
import time
from streamlit.runtime.scriptrunner import add_script_run_ctx
warnings.filterwarnings("ignore")

from .config import PLANNER_MODEL, WRITER_MODEL
from .state_managers import load_dataset, get_current_dataset
from .data_profiling import infer_task_type
from .report_generator import generate_enhanced_recommendations
from .visuals import parse_and_enhance_recommendations
from .pdf_generator import markdown_to_luxury_pdf as markdown_to_pdf

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline_executor


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — Premium SaaS look & PowerBI Dashboard Styles
# ─────────────────────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

.stApp { font-family: 'Inter', sans-serif; }
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 20px rgba(102,126,234,.35); }
    50%       { box-shadow: 0 0 45px rgba(118,75,162,.55); }
}
@keyframes slide-up {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes step-in {
    from { opacity: 0; transform: translateX(-16px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes pulse-border {
    0%, 100% { border-color: rgba(99,102,241,0.4); }
    50%      { border-color: rgba(99,102,241,1); }
}

.dss-hero {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 24px; padding: 48px 40px 40px; margin-bottom: 32px;
    position: relative; overflow: hidden; animation: pulse-glow 4s ease-in-out infinite;
}
.dss-hero-title { font-size: 2.6rem; font-weight: 800; color: #fff; margin: 0 0 8px; z-index: 1; position: relative;}
.dss-hero-subtitle { font-size: 1.05rem; color: rgba(255,255,255,.72); margin: 0 0 32px; z-index: 1; position: relative;}
.dss-model-cards { display: flex; gap: 16px; flex-wrap: wrap; z-index: 1; position: relative;}
.dss-model-card-hero {
    background: rgba(255,255,255,.08); backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,.15); border-radius: 14px;
    padding: 16px 20px; min-width: 160px; flex: 1;
}
.dss-model-card-icon { font-size: 1.6rem; margin-bottom: 6px; }
.dss-model-card-label { font-size: .72rem; font-weight: 600; color: rgba(255,255,255,.55); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.dss-model-card-name { font-size: .95rem; font-weight: 600; color: #e0e7ff; word-break: break-all; }
.dss-model-card-desc { font-size: .78rem; color: rgba(255,255,255,.52); margin-top: 4px; }

.stat-card {
    background: linear-gradient(145deg, #1e1b4b, #312e81);
    border: 1px solid rgba(129,140,248,.25); border-radius: 16px;
    padding: 20px; text-align: center; animation: slide-up .4s ease both;
}
.stat-card-icon { font-size: 1.8rem; margin-bottom: 6px; }
.stat-card-value { font-size: 1.7rem; font-weight: 700; color: #a5b4fc; }
.stat-card-label { font-size: .78rem; color: rgba(255,255,255,.55); text-transform: uppercase; letter-spacing: .8px; margin-top: 2px; }

.section-heading {
    font-size: 1.25rem; font-weight: 700; color: #e2e8f0;
    display: flex; align-items: center; gap: 10px;
    margin: 32px 0 16px; padding-bottom: 10px; border-bottom: 1px solid rgba(99,102,241,.3);
}
.section-heading::before {
    content: ''; display: inline-block; width: 4px; height: 22px;
    background: linear-gradient(180deg, #818cf8, #a78bfa); border-radius: 2px;
}

div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    border: none !important; border-radius: 14px !important;
    font-weight: 700 !important; font-size: 1.05rem !important; letter-spacing: .3px !important;
    box-shadow: 0 8px 28px rgba(99,102,241,.45) !important; transition: all .25s ease !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-2px) !important; box-shadow: 0 14px 36px rgba(99,102,241,.6) !important;
}

/* ── ✨ PowerBI Style AI Output Formatting ── */
.rec-container {
    background: transparent; border: none; padding: 0; box-shadow: none; animation: slide-up .5s ease;
}
.rec-container p {
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8));
    border: 1px solid rgba(255,255,255,0.07); padding: 22px; border-radius: 14px;
    font-size: 0.95rem; color: #e2e8f0; line-height: 1.7;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15); margin-bottom: 24px;
}
.rec-container h1, .rec-container h2 {
    font-family: 'Inter', sans-serif; font-weight: 800; color: #ffffff;
    margin-top: 36px; margin-bottom: 16px; display: inline-block;
    background: linear-gradient(90deg, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.rec-container h3 {
    font-family: 'Inter', sans-serif; font-weight: 700; color: #34d399;
    margin-top: 28px; margin-bottom: 14px; display: flex; align-items: center; gap: 8px;
}
.rec-container h3::before { content: '⚡'; font-size: 1.2rem; }
.rec-container ul {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px;
    padding: 0; list-style-type: none; margin-bottom: 24px;
}
.rec-container li {
    background: rgba(99, 102, 241, 0.08); border-left: 4px solid #818cf8;
    padding: 16px 20px; border-radius: 0 10px 10px 0; font-size: 0.92rem; color: #cbd5e1;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
.rec-container strong {
    color: #fde68a; font-weight: 700; background: rgba(251,191,36,0.12);
    padding: 3px 8px; border-radius: 6px; border: 1px solid rgba(251,191,36,0.25);
}

/* ── Stepper ── */
.gen-stepper {
    background: linear-gradient(135deg, #0f172a, #1e1b4b);
    border: 1px solid rgba(99,102,241,.3); border-radius: 20px; padding: 28px; margin: 16px 0;
}
.gen-step { display: flex; align-items: center; gap: 16px; padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,.06); animation: step-in .4s ease both; }
.gen-step:last-child { border-bottom: none; }
.gen-step-dot {
    width: 34px; height: 34px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center; font-size: .9rem; flex-shrink: 0;
}
.gen-step-dot.done   { background: rgba(16,185,129,.2); border: 2px solid #10b981; }
.gen-step-dot.active { background: rgba(99,102,241,.25); border: 2px solid #818cf8; animation: spin 1.2s linear infinite; }
.gen-step-dot.pending { background: rgba(255,255,255,.05); border: 2px solid rgba(255,255,255,.15); }
.gen-step-text { font-size: .9rem; }
.gen-step-text.done   { color: #6ee7b7; }
.gen-step-text.active { color: #a5b4fc; font-weight: 600; }
.gen-step-text.pending { color: rgba(255,255,255,.35); }

/* ── Clickable Model Cards ── */
.model-sel-card {
    margin-bottom: 16px; 
    background: linear-gradient(145deg, #1e293b, #0f172a);
    border: 1.5px solid rgba(255,255,255,.1);
    border-radius: 16px;
    padding: 20px;
    position: relative;
    transition: all .15s ease;
    min-height: 90px;
    cursor: pointer;
}
.model-sel-card.selected {
    border-color: #10b981 !important;
    box-shadow: 0 0 22px rgba(16,185,129,.28);
    background: linear-gradient(145deg, rgba(16,185,129,0.12), #0f172a);
    transform: scale(1.018);
}
.model-sel-card:hover {
    border-color: rgba(129,140,248,.55);
    box-shadow: 0 8px 24px rgba(99,102,241,.2);
    transform: translateY(-2px);
}
.model-name {
    font-weight: 700;
    font-size: 1.02rem;
    color: #ffffff;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.model-sel-card-btn {
    margin-top: 10px;
    width: 100%;
}

.rating-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: .75rem; font-weight: 600; margin-top: 6px; }
.badge-champion  { background: rgba(16,185,129,.15); color: #34d399; border: 1px solid rgba(16,185,129,.3); }
.badge-strong    { background: rgba(59,130,246,.15);  color: #60a5fa; border: 1px solid rgba(59,130,246,.3);  }
.badge-baseline  { background: rgba(251,191,36,.12);  color: #fbbf24; border: 1px solid rgba(251,191,36,.25); }
.badge-experimental { background: rgba(156,163,175,.1); color: #9ca3af; border: 1px solid rgba(156,163,175,.2); }

/* ── Pipeline Plan ── */
.v-step { display: flex; align-items: flex-start; gap: 16px; }
.v-step-num {
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: .82rem; font-weight: 700; flex-shrink: 0; box-shadow: 0 4px 14px rgba(0,0,0,.3);
}
.v-step-icon { font-size: 1.4rem; flex-shrink: 0; margin-top: 2px; }
.v-step-title { font-weight: 700; font-size: 1.05rem; }
.v-step-detail { font-size: .88rem; margin-top: 4px; }

.gemini-card { background: linear-gradient(145deg, #1c1917, #292524); border: 1px solid rgba(251,191,36,.25); border-radius: 16px; padding: 20px 24px; margin: 16px 0; display: flex; align-items: flex-start; gap: 14px; }
.gemini-card-icon { font-size: 1.8rem; flex-shrink: 0; }
.gemini-card-title { font-size: 1rem; font-weight: 600; color: #fde68a; margin-bottom: 2px; }
.gemini-card-desc { font-size: .82rem; color: rgba(255,255,255,.5); }
.chart-insight { background: rgba(99,102,241,.08); border-left: 3px solid #818cf8; border-radius: 0 10px 10px 0; padding: 10px 14px; margin-top: 4px; font-size: .84rem; color: #cbd5e1; }
.pipeline-hero { background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%); padding: 36px 40px; border-radius: 20px; margin: 16px 0; border: 1px solid rgba(129,140,248,.3); position: relative; overflow: hidden; }
.pipeline-hero h2 { font-size: 1.9rem; font-weight: 800; color: #e0e7ff; margin: 0 0 6px; }
.pipeline-hero p { color: rgba(255,255,255,.6); font-size: .95rem; margin: 0; }

.exec-terminal { background: #020617; border: 1px solid rgba(99,102,241,.3); border-radius: 16px; overflow: hidden; margin-top:12px; }
.terminal-header { background: rgba(99,102,241,.12); padding: 10px 18px; display: flex; align-items: center; gap: 8px; border-bottom: 1px solid rgba(99,102,241,.2); }
.terminal-dot { width: 10px; height: 10px; border-radius: 50%; }
.terminal-title { font-size: .82rem; color: rgba(255,255,255,.5); margin-left: 6px; }
.terminal-body { padding: 18px; max-height: 260px; overflow-y: auto; font-family: 'Menlo', 'Monaco', monospace; font-size: .82rem; }
.log-ok   { color: #34d399; padding: 3px 0; border-bottom: 1px solid rgba(255,255,255,.04); }
.log-err  { color: #f87171; padding: 3px 0; }
.log-info { color: #a5b4fc; padding: 3px 0; border-bottom: 1px solid rgba(255,255,255,.04); }
.log-run  { color: #fde68a; padding: 3px 0; border-bottom: 1px solid rgba(255,255,255,.04); }

.exec-progress-wrap { background: linear-gradient(135deg, #0f172a, #1e1b4b); border: 1px solid rgba(99,102,241,.25); border-radius: 20px; padding: 28px 32px; margin: 16px 0; }
.exec-progress-label { font-size: .9rem; color: #a5b4fc; margin-bottom: 12px; display: flex; align-items: center; gap: 10px; }
.spinner { width: 16px; height: 16px; border: 2px solid rgba(129,140,248,.3); border-top-color: #818cf8; border-radius: 50%; animation: spin .8s linear infinite; display: inline-block; }
.result-nav-card { background: linear-gradient(145deg, #1e293b, #0f172a); border: 1.5px solid rgba(99,102,241,.25); border-radius: 18px; padding: 26px 20px; text-align: center; transition: all .25s ease; }
.result-nav-card:hover { border-color: #818cf8; box-shadow: 0 12px 36px rgba(99,102,241,.25); transform: translateY(-4px); }
.result-nav-icon { font-size: 2.2rem; margin-bottom: 10px; }
.result-nav-title { font-weight: 700; color: #e0e7ff; font-size: 1rem; margin-bottom: 6px; }
.result-nav-desc { font-size: .8rem; color: rgba(255,255,255,.45); }
.ready-prompt { background: linear-gradient(145deg, rgba(99,102,241,.08), rgba(139,92,246,.06)); border: 1.5px dashed rgba(129,140,248,.35); border-radius: 20px; padding: 40px; text-align: center; margin: 20px 0; animation: slide-up .5s ease; }
.ready-prompt-icon { font-size: 3rem; margin-bottom: 12px; }
.ready-prompt-title { font-size: 1.3rem; font-weight: 700; color: #a5b4fc; margin-bottom: 8px; }
.ready-prompt-text { color: rgba(255,255,255,.5); font-size: .92rem; }
.gradient-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(99,102,241,.5), transparent); margin: 36px 0; border: none; }
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _log_class(line: str) -> str:
    if line.startswith("✅") or "complet" in line.lower() or "success" in line.lower(): return "log-ok"
    if line.startswith("❌") or "error" in line.lower() or "fail" in line.lower(): return "log-err"
    if line.startswith("⚠️") or "skip" in line.lower(): return "log-run"
    return "log-info"


def _badge_class(rating: str) -> str:
    r = rating.lower()
    if "champion" in r: return "badge-champion"
    if "strong" in r: return "badge-strong"
    if "baseline" in r: return "badge-baseline"
    return "badge-experimental"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION RENDERERS
# ─────────────────────────────────────────────────────────────────────────────

def _render_hero():
    st.markdown(f"""<div class="dss-hero">
<div class="dss-hero-title">🚀 DSS Recommendations Hub</div>
<div class="dss-hero-subtitle">AI-powered data science advisor — Upload your dataset, generate a strategy, then let the pipeline execute it end-to-end.</div>
<div class="dss-model-cards">
<div class="dss-model-card-hero"><div class="dss-model-card-icon">🧠</div><div class="dss-model-card-label">Planner Model</div><div class="dss-model-card-name">{PLANNER_MODEL}</div><div class="dss-model-card-desc">Reasons through ML strategy & dataset facts</div></div>
<div class="dss-model-card-hero"><div class="dss-model-card-icon">✍️</div><div class="dss-model-card-label">Writer Model</div><div class="dss-model-card-name">{WRITER_MODEL}</div><div class="dss-model-card-desc">Converts decisions into a polished report</div></div>
<div class="dss-model-card-hero"><div class="dss-model-card-icon">⚡</div><div class="dss-model-card-label">Pipeline Engine</div><div class="dss-model-card-name">Auto-Executor</div><div class="dss-model-card-desc">Transforms data & trains models automatically</div></div>
</div>
</div>""", unsafe_allow_html=True)


def _render_dataset_section(current_df: pd.DataFrame):
    st.markdown('<div class="section-heading">📂 Dataset Overview</div>', unsafe_allow_html=True)
    missing_pct = round(current_df.isnull().sum().sum() / max(current_df.size, 1) * 100, 1)
    num_cols  = len(current_df.select_dtypes(include="number").columns)
    cat_cols  = len(current_df.select_dtypes(exclude="number").columns)
    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "🗂️", f"{current_df.shape[0]:,}", "Rows"),
        (c2, "📋", f"{current_df.shape[1]}", "Columns"),
        (c3, "🔢", f"{num_cols}", "Numeric"),
        (c4, "🏷️", f"{cat_cols}", "Categorical"),
        (c5, "❓", f"{missing_pct}%", "Missing")
    ]
    for col, icon, val, label in cards:
        with col:
            st.markdown(
                f'<div class="stat-card"><div class="stat-card-icon">{icon}</div><div class="stat-card-value">{val}</div><div class="stat-card-label">{label}</div></div>',
                unsafe_allow_html=True
            )
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        with st.expander("📋 Dataset Preview", expanded=False):
            st.dataframe(current_df.head(8), use_container_width=True)
    with col_r:
        with st.expander("📊 Column Summary", expanded=False):
            buf = io.StringIO()
            current_df.info(buf=buf)
            st.text(buf.getvalue())


def _render_generation_section(current_df: pd.DataFrame):
    st.markdown('<div class="section-heading">✨ AI Engine Settings</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gemini-card"><div class="gemini-card-icon">✨</div><div><div class="gemini-card-title">Gemini AI Enhancement (Optional)</div><div class="gemini-card-desc">Enable Google Gemini 2.0 for deeper Feature Engineering & Deployment advice — requires a free API key.</div></div></div>',
        unsafe_allow_html=True
    )
    use_gemini = st.checkbox("Enable Gemini AI for enhanced recommendations", value=False, help="Get your key at https://ai.google.dev/")
    user_api_key = st.text_input("Gemini API Key", type="password", placeholder="AIza...", help="Leave blank to use default if configured.") if use_gemini else None
    return use_gemini, user_api_key


def _render_generation_stepper():
    logs = st.session_state.get("gen_logs", [])
    if not logs: return
    STEPS = [
        ("🔍", "Profiling Data"),
        ("🧠", "Planning Strategy"),
        ("📊", "Building Feature Engineering"),
        ("🤖", "Generating Model Selection"),
        ("📈", "Composing Evaluation Framework"),
        ("🚀", "Writing Deployment Strategy"),
        ("✅", "Report Complete")
    ]

    def _detect_step(logs_list):
        if not st.session_state.get("is_generating", False) and st.session_state.get("recommendations"):
            return len(STEPS)
        joined = " ".join(logs_list).lower()
        if "complete" in joined: return 6
        if "deploy" in joined: return 5
        if "eval" in joined: return 4
        if "model sel" in joined or "targeted" in joined: return 3
        if "feature" in joined: return 2
        if "planner" in joined or "planning" in joined: return 1
        return 0

    current = _detect_step(logs)
    is_running = st.session_state.get("is_generating", False)
    steps_html = ""
    for i, (icon, label) in enumerate(STEPS):
        if i < current: cls = "done"; dot_icon = "✓"
        elif i == current and is_running: cls = "active"; dot_icon = "◌"
        else: cls = "pending"; dot_icon = str(i + 1)
        steps_html += f'<div class="gen-step"><div class="gen-step-dot {cls}">{dot_icon}</div><div class="gen-step-text {cls}">{icon} {label}</div></div>'

    st.markdown(
        f'<div class="gen-stepper"><div style="font-size:.8rem; color:rgba(255,255,255,.4); margin-bottom:16px; text-transform:uppercase; letter-spacing:1px;">AI Generation Progress</div>{steps_html}</div>',
        unsafe_allow_html=True
    )
    if is_running:
        st.markdown(
            '<p style="font-size:.82rem; color:rgba(255,255,255,.4); text-align:center; margin-top:8px;">💡 You can browse other tabs — results appear automatically.</p>',
            unsafe_allow_html=True
        )


def _render_recommendations_output(enhanced: dict):
    st.markdown('<div class="section-heading">🧭 AI Strategy Report</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="rec-container">\n\n{enhanced["full_text"]}\n\n</div>', 
        unsafe_allow_html=True
    )


def _render_visualizations(enhanced: dict):
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">📊 Visual Intelligence Dashboard</div>', unsafe_allow_html=True)
    is_unsup = enhanced.get("is_unsupervised", False)
    visuals  = enhanced.get("visuals", {})

    if is_unsup:
        st.markdown('<div class="chart-insight">ℹ️ <strong>Unsupervised Mode</strong> — no target column. Showing exploratory charts.</div><div style="height:16px"></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(visuals["data_quality"], use_container_width=True)
            st.markdown('<div class="chart-insight">🔍 <strong>Data Quality</strong> — Missing values.</div>', unsafe_allow_html=True)
        with c2:
            st.plotly_chart(visuals["feature_variance"], use_container_width=True)
            st.markdown('<div class="chart-insight">📐 <strong>Feature Spread</strong> — Variance.</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        st.plotly_chart(visuals["pairplot"], use_container_width=True)
        st.markdown('<div class="chart-insight">🔭 <strong>Pair Plot</strong> — Groupings.</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(visuals["cat_frequency"], use_container_width=True)
            st.markdown('<div class="chart-insight">📊 <strong>Categorical</strong> — Distributions.</div>', unsafe_allow_html=True)
        with c4:
            st.plotly_chart(visuals["correlation"], use_container_width=True)
            st.markdown('<div class="chart-insight">🔗 <strong>Correlations</strong>.</div>', unsafe_allow_html=True)
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(visuals["data_quality"], use_container_width=True)
            st.markdown('<div class="chart-insight">🔍 <strong>Data Quality</strong> — Missing value coverage.</div>', unsafe_allow_html=True)
        with c2:
            if visuals.get("target_dist"):
                st.plotly_chart(visuals["target_dist"], use_container_width=True)
                st.markdown('<div class="chart-insight">🎯 <strong>Target Distribution</strong>.</div>', unsafe_allow_html=True)
            else:
                st.info("Select a target column to see distribution metrics.")
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        st.plotly_chart(visuals["correlation"], use_container_width=True)
        st.markdown('<div class="chart-insight">🔗 <strong>Feature Correlations</strong> — Strongest predictive relationships.</div>', unsafe_allow_html=True)


def _render_download_bar(enhanced: dict):
    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "💾 Download Markdown Report",
            data=enhanced["full_text"],
            file_name="DSS_Recommendations.md",
            mime="text/markdown",
            use_container_width=True
        )
    with c2:
        try:
            pdf_bytes = markdown_to_pdf(
                enhanced["full_text"],
                cover_image_path="https://github.com/lWAHBAl/img/raw/20c1f928721c741288562640559fc6e039d2b034/image.png"
            )
            st.download_button(
                "📄 Download PDF Report",
                data=pdf_bytes,
                file_name="DSS_Recommendations.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"PDF generation error: {e}")


def _render_pipeline_hero():
    st.markdown(
        '<div class="pipeline-hero"><h2>⚡ Automated ML Pipeline</h2><p>Select models, review the execution plan, then let the AI build your full pipeline.</p></div>',
        unsafe_allow_html=True
    )


def _render_model_selection(p_json: dict):
    """
    Clickable card selection — the entire card is the button.
    No visible toggle button beneath. Selection state shown via card border/color.
    """
    st.markdown('<div class="section-heading">🤖 Select Models to Train</div>', unsafe_allow_html=True)

    recommended_models = p_json.get("models", [])
    if not recommended_models:
        st.info("No model recommendations found.")
        return []

    # Initialize state
    for m in recommended_models:
        key = f"model_sel_{m['name']}"
        if key not in st.session_state:
            st.session_state[key] = m.get("rating", "") in ["Champion", "Strong Alternative"]

    cols = st.columns(min(len(recommended_models), 4))
    selected_model_names = []

    # Wrap in a div with class 'model-sel-grid' so the CSS overlay only targets this area
    st.markdown('<div class="model-sel-grid">', unsafe_allow_html=True)
    for i, m in enumerate(recommended_models):
        m_name    = m["name"]
        rating    = m.get("rating", "")
        badge_cls = _badge_class(rating)
        state_key = f"model_sel_{m_name}"
        is_selected = st.session_state[state_key]

        if is_selected:
            selected_model_names.append(m_name)

        sel_class  = "selected" if is_selected else ""
        check_icon = "✅" if is_selected else ""

        with cols[i % len(cols)]:
            # Visual card (no invisible overlay — clean and fast)
            st.markdown(f"""
            <div class="model-sel-card {sel_class}">
                <div class="model-name">
                    <span>{m_name}</span>
                    <span>{check_icon}</span>
                </div>
                <span class="rating-badge {badge_cls}">{rating}</span>
            </div>
            """, unsafe_allow_html=True)

            # Visible Select / ✅ Selected toggle button
            btn_label = "✅ Selected" if is_selected else "Select"
            btn_type  = "primary" if is_selected else "secondary"
            if st.button(btn_label, key=f"btn_{m_name}", use_container_width=True, type=btn_type):
                st.session_state[state_key] = not is_selected
                st.rerun()

    st.markdown(
        f'<p style="color:#a5b4fc; font-weight:700; font-size:0.92rem; '
        f'text-align:right; margin-top:10px;">'
        f'✅ Selected: {len(selected_model_names)} model(s)</p>',
        unsafe_allow_html=True
    )
    return selected_model_names

def _render_plan_review(plan: list) -> list:
    """
    Fixed plan review:
    - Each step has its own stable checkbox key based on index
    - Returns only the steps the user kept checked
    - Does NOT mutate session state automation_plan directly
    """
    st.markdown(
        '<p style="color:rgba(255,255,255,.45); font-size:.86rem; margin-bottom:18px;">'
        'Review and <strong>uncheck any steps</strong> you want to skip before execution.</p>',
        unsafe_allow_html=True
    )

    # Initialize keep-state for each step once (stable keys by index)
    for i in range(len(plan)):
        key = f"keep_step_{i}"
        if key not in st.session_state:
            st.session_state[key] = True


    edited_plan = []
    for i, step in enumerate(plan):
        key = f"keep_step_{i}"
        col1, col2 = st.columns([0.08, 0.92])

        with col1:
            st.markdown("<div style='margin-top: 15px;'>", unsafe_allow_html=True)
            keep_step = st.checkbox(
                "Keep",
                value=st.session_state[key],
                key=key,
                label_visibility="collapsed"
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            title_color  = "#ffffff" if keep_step else "#64748b"
            text_decor   = "none" if keep_step else "line-through"
            border_style = "1px solid rgba(255,255,255,.08)" if i < len(plan) - 1 else "none"
            num_bg       = "linear-gradient(135deg, #6366f1, #8b5cf6)" if keep_step else "#334155"
            opacity      = 1.0 if keep_step else 0.45

            st.markdown(
                f'<div class="v-step" style="border-bottom:{border_style}; padding:10px 0; opacity:{opacity};">'
                f'<div class="v-step-num" style="background:{num_bg}; color:#fff;">{i+1}</div>'
                f'<div class="v-step-icon">{step["icon"]}</div>'
                f'<div>'
                f'<div class="v-step-title" style="color:{title_color}; text-decoration:{text_decor};">{step["name"]}</div>'
                f'<div class="v-step-detail" style="color:#cbd5e1;">{step["details"]}</div>'
                f'</div></div>',
                unsafe_allow_html=True
            )

        if keep_step:
            edited_plan.append(step)

    return edited_plan


def _render_execution_progress(prog: dict, is_running: bool):
    progress_value = min(prog["current_step"] / max(prog["total_steps"], 1), 1.0)
    pct = int(progress_value * 100)
    if is_running:
        status_html = (
            f'<div class="exec-progress-label"><span class="spinner"></span>'
            f'<span>Executing: <strong style="color:#e0e7ff;">{prog["message"]}</strong></span>'
            f'<span style="margin-left:auto; color:#818cf8; font-weight:700;">{pct}%</span></div>'
        )
    else:
        status_html = '<div class="exec-progress-label" style="color:#34d399;">✅ &nbsp;<strong>Pipeline execution complete!</strong></div>'

    st.markdown(
        f'<div class="exec-progress-wrap">'
        f'<div style="font-size:.82rem; color:rgba(255,255,255,.4); text-transform:uppercase; letter-spacing:1px; margin-bottom:14px;">⚡ Execution Status</div>'
        f'{status_html}</div>',
        unsafe_allow_html=True
    )
    st.progress(progress_value)

    log_lines_html = ""
    for line in prog.get("log", []):
        cls  = _log_class(line)
        safe = line.replace("<", "&lt;").replace(">", "&gt;")
        log_lines_html += f'<div class="{cls}">→ {safe}</div>'
    if is_running and prog.get("message"):
        safe_msg = prog["message"].replace("<", "&lt;").replace(">", "&gt;")
        log_lines_html += f'<div class="log-run">⏳ {safe_msg}...</div>'

    if log_lines_html:
        with st.expander("💻 View Terminal Execution Logs", expanded=False):
            st.markdown(
                f'<div class="exec-terminal">'
                f'<div class="terminal-header">'
                f'<div class="terminal-dot" style="background:#ef4444;"></div>'
                f'<div class="terminal-dot" style="background:#f59e0b;"></div>'
                f'<div class="terminal-dot" style="background:#10b981;"></div>'
                f'<div class="terminal-title">Pipeline Output</div></div>'
                f'<div class="terminal-body">{log_lines_html}</div></div>',
                unsafe_allow_html=True
            )


def _render_completion_cards():
    st.markdown(
        '<div class="gradient-divider"></div>'
        '<h3 style="color:#e0e7ff; text-align:center; margin-bottom:6px;">🎉 Your Pipeline is Ready!</h3>'
        '<p style="color:rgba(255,255,255,.45); text-align:center; margin-bottom:28px; font-size:.9rem;">Explore your results across the platform</p>',
        unsafe_allow_html=True
    )
    c1, c2, c3 = st.columns(3)
    nav = [
        (c1, "🔧", "Transform Tab",       "Cleaning & scaling steps applied"),
        (c2, "🤖", "Machine Learning Tab", "Interact with trained models"),
        (c3, "📄", "Reporting Tab",        "Deep-dive performance analytics")
    ]
    for col, icon, title, desc in nav:
        with col:
            st.markdown(
                f'<div class="result-nav-card"><div class="result-nav-icon">{icon}</div>'
                f'<div class="result-nav-title">{title}</div>'
                f'<div class="result-nav-desc">{desc}</div></div>',
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUN
# ─────────────────────────────────────────────────────────────────────────────
def run():
    st.markdown(_CSS, unsafe_allow_html=True)

    defaults = {
        "current_dataset":    None,
        "dataset_loaded":     False,
        "dataset_name":       None,
        "recommendations":    None,
        "current_task_type":  "Classification",
        "target_column":      None,
        "is_generating":      False,
        "gen_logs":           [],
        "gen_thread":         None,
        "pipeline_json":      None,
        "automation_plan":    None,
        "automation_running": False,
        "automation_done":    False,
        "automation_progress": {"current_step": 0, "total_steps": 0, "message": "", "log": []},
        "selected_trans":     [],
        "automation_thread":  None,
        "reviewed_plan":      None,   # ← stores the user-filtered plan separately
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    _render_hero()

    st.markdown('<div class="section-heading">📂 Upload Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your dataset here (CSV, XLSX, XLS)",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed"
    )
    current_df = None

    if uploaded_file is not None:
        current_df = load_dataset(uploaded_file)
    elif st.session_state.dataset_loaded and st.session_state.current_dataset is not None:
        current_df = st.session_state.current_dataset
        st.info(f"📁 Using previously loaded dataset: **{st.session_state.dataset_name}**")

    if current_df is None or current_df.empty:
        st.markdown(
            '<div class="ready-prompt"><div class="ready-prompt-icon">📂</div>'
            '<div class="ready-prompt-title">Upload a Dataset to Begin</div>'
            '<div class="ready-prompt-text">Drag & drop or browse a CSV / Excel file to start your AI-powered analysis.</div></div>',
            unsafe_allow_html=True
        )
        return

    _render_dataset_section(current_df)
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    target_options = ["None"] + current_df.columns.tolist()
    default_idx    = target_options.index(st.session_state.target_column) if st.session_state.target_column in target_options else 0
    selected_target = st.selectbox(
        "🎯 Target column (leave None for unsupervised analysis)",
        target_options,
        index=default_idx
    )
    st.session_state.target_column = None if selected_target == "None" else selected_target

    use_gemini, user_api_key = _render_generation_section(current_df)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    if st.session_state.get("gen_logs"):
        _render_generation_stepper()

    if not st.session_state.is_generating:
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        if st.button("🚀  Generate AI Strategy Report", type="primary", use_container_width=True):
            st.session_state.is_generating  = True
            st.session_state.gen_logs       = ["🔍 Inferring task type and profiling data..."]
            detected_task = infer_task_type(current_df, st.session_state.target_column)
            st.session_state.current_task_type = detected_task

            def background_worker(df_in, task_type_in, target_col_in, use_gemini_in, api_key_in):
                def log_callback(msg):
                    st.session_state.gen_logs.append(msg)
                try:
                    res = generate_enhanced_recommendations(
                        "", df_in, task_type_in, target_col_in, use_gemini_in, api_key_in, log_callback
                    )
                    st.session_state.raw_rec_text = res
                except Exception as exc:
                    st.session_state.gen_logs.append(f"❌ Error: {exc}")
                    st.session_state.raw_rec_text = None

            thread = threading.Thread(
                target=background_worker,
                args=(current_df, detected_task, st.session_state.target_column, use_gemini, user_api_key)
            )
            add_script_run_ctx(thread)
            st.session_state.gen_thread = thread
            thread.start()
            st.rerun()
    else:
        st.markdown(
            '<div style="text-align:center; padding:12px 0;">'
            '<span class="spinner" style="width:22px;height:22px;border-width:3px;"></span>'
            '<span style="color:#a5b4fc; font-size:.9rem; margin-left:10px;">AI is generating your strategy report…</span></div>',
            unsafe_allow_html=True
        )

    # ── Recommendations + Pipeline ──────────────────────────────────────────
    if st.session_state.recommendations is not None and isinstance(st.session_state.recommendations, dict):
        enhanced = st.session_state.recommendations
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        _render_recommendations_output(enhanced)
        _render_visualizations(enhanced)
        _render_download_bar(enhanced)

        # ── Clear Recommendations button — placed safely after the report ──
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        if st.button("🗑️ Clear Recommendations", type="secondary", use_container_width=False):
            st.session_state.recommendations = None
            st.rerun()

        if st.session_state.get("pipeline_json"):
            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
            _render_pipeline_hero()
            p_json = st.session_state.pipeline_json

            if not st.session_state.automation_running and not st.session_state.automation_done:
                selected_model_names = _render_model_selection(p_json)
                st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="gemini-card" style="border-color:rgba(99,102,241,.3);">'
                    '<div class="gemini-card-icon">⚙️</div>'
                    '<div><div class="gemini-card-title" style="color:#a5b4fc;">Hyperparameter Grid Search</div>'
                    '<div class="gemini-card-desc">Slower but finds optimal parameters via cross-validation.</div>'
                    '</div></div>',
                    unsafe_allow_html=True
                )
                grid_search = st.toggle("Enable Grid Search (slower, more thorough)", value=False)
                st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

                # ── Build plan button ────────────────────────────────────────
                if st.button("⚡ Build Pipeline Plan", type="primary", use_container_width=True):
                    if not selected_model_names:
                        st.warning("⚠️ Please select at least one model to continue.")
                    else:
                        plan = []
                        for c in p_json.get("cleaning", []):
                            if c["action"] == "drop_column":
                                plan.append({"icon": "🧹", "name": "Drop Column",    "details": f"{c['column']} — {c['reason']}", "_meta_cat": "cleaning", "_meta_data": c})
                            else:
                                plan.append({"icon": "🔧", "name": "Handle Missing", "details": f"{c['column']} → {c['strategy']} imputation", "_meta_cat": "cleaning", "_meta_data": c})
                        for t in p_json.get("transformations", []):
                            plan.append({"icon": "📐", "name": t["type"].capitalize(), "details": f"{t['category']} on {', '.join(t['columns'])}", "_meta_cat": "transformations", "_meta_data": t})
                        for m_name in selected_model_names:
                            plan.append({"icon": "🤖", "name": "Train Model",     "details": m_name, "_meta_cat": "model", "_meta_data": m_name})
                            plan.append({"icon": "📊", "name": "Generate Report", "details": f"Metrics & plots for {m_name}", "_meta_cat": "report", "_meta_data": m_name})

                        # Reset checkbox state so new plan gets fresh defaults
                        for i in range(len(plan)):
                            st.session_state[f"keep_step_{i}"] = True

                        st.session_state.automation_plan      = plan
                        st.session_state.reviewed_plan        = None   # reset reviewed plan
                        st.session_state.selected_models_auto = selected_model_names

                # ── Plan review + Execute ────────────────────────────────────
                if st.session_state.automation_plan:
                    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

                    # Render plan steps directly (no expander — it disappears on Execute anyway)
                    reviewed_plan = _render_plan_review(st.session_state.automation_plan)
                    if reviewed_plan:
                        st.session_state.reviewed_plan = reviewed_plan

                    # ── Action buttons rendered below the plan ──────────────
                    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
                    col_exec, col_can = st.columns([3, 1])
                    with col_exec:
                        if st.button("✅ Execute Pipeline", type="primary", use_container_width=True):
                            current_reviewed = st.session_state.get("reviewed_plan") or []
                            if not current_reviewed:
                                st.warning("⚠️ No steps selected. Please keep at least one step.")
                            else:
                                # Clear the plan immediately so it vanishes from the UI
                                st.session_state.automation_plan    = None
                                st.session_state.automation_running = True
                                st.session_state.automation_progress = {
                                    "current_step": 0,
                                    "total_steps":  len(current_reviewed),
                                    "message":      "Initializing...",
                                    "log":          []
                                }

                                filtered_p_json = {
                                    "target_column": p_json.get("target_column"),
                                    "task_type": p_json.get("task_type"),
                                    "models": p_json.get("models"),
                                    "cleaning": [s["_meta_data"] for s in current_reviewed if s.get("_meta_cat") == "cleaning"],
                                    "transformations": [s["_meta_data"] for s in current_reviewed if s.get("_meta_cat") == "transformations"]
                                }
                                filtered_models = []
                                for s in current_reviewed:
                                    if s.get("_meta_cat") == "model" and s["_meta_data"] not in filtered_models:
                                        filtered_models.append(s["_meta_data"])

                                def auto_worker(plan_in, df_in, p_json_in, models_in):
                                    original_dataframe = st.dataframe
                                    original_table     = st.table
                                    st.dataframe = lambda *args, **kwargs: None
                                    st.table     = lambda *args, **kwargs: None
                                    try:
                                        for k, default in [("pipeline", {}), ("selected_trans", []), ("load", False)]:
                                            if k not in st.session_state:
                                                st.session_state[k] = default
                                        for sub in ["transformations", "ML", "report_items"]:
                                            st.session_state.pipeline.setdefault(sub, [])
                                        if "df_original" not in st.session_state:
                                            st.session_state.df_original = df_in.copy() if df_in is not None else None

                                        def prog_cb(step, total, msg):
                                            st.session_state.automation_progress["current_step"] = step
                                            st.session_state.automation_progress["total_steps"]  = total
                                            st.session_state.automation_progress["message"]      = msg

                                        pipeline_executor.execute_automation_pipeline(
                                            plan_in, df_in, p_json_in, models_in, prog_cb
                                        )
                                    finally:
                                        st.dataframe = original_dataframe
                                        st.table     = original_table

                                thread = threading.Thread(
                                    target=auto_worker,
                                    args=(current_reviewed, current_df, filtered_p_json, filtered_models)
                                )
                                add_script_run_ctx(thread)
                                st.session_state.automation_thread = thread
                                thread.start()
                                st.rerun()

                    with col_can:
                        if st.button("✏️ Modify Plan", use_container_width=True):
                            st.session_state.automation_plan = None
                            st.session_state.reviewed_plan   = None
                            st.rerun()

            # ── Live execution display ───────────────────────────────────────
            if st.session_state.automation_running or st.session_state.automation_done:
                prog = st.session_state.automation_progress
                _render_execution_progress(prog, st.session_state.automation_running)

                if st.session_state.automation_done:
                    _render_completion_cards()
                    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)
                    if st.columns([1, 2, 1])[1].button("🔄 Start New Automation", type="secondary", use_container_width=True):
                        st.session_state.automation_done = False
                        st.session_state.automation_plan = None
                        st.session_state.reviewed_plan   = None
                        st.rerun()

    # ── Live Update Engine ───────────────────────────────────────────────────
    if st.session_state.get("is_generating"):
        thread = st.session_state.gen_thread
        if thread and not thread.is_alive():
            rec_text = st.session_state.get("raw_rec_text")
            if rec_text:
                st.session_state.recommendations = parse_and_enhance_recommendations(
                    rec_text,
                    st.session_state.current_task_type,
                    st.session_state.current_dataset,
                    st.session_state.target_column
                )
            st.session_state.is_generating = False
            st.rerun()
        else:
            time.sleep(0.3)
            st.rerun()

    if st.session_state.get("automation_running"):
        thread = st.session_state.automation_thread
        if thread and not thread.is_alive():
            st.session_state.automation_running = False
            st.session_state.automation_done    = True
            st.rerun()
        else:
            time.sleep(0.3)
            st.rerun()


if __name__ == "__main__":
    run()
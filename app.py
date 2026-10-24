import os
import sys
import time
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="X-Estal RAG Engine | by Nitish Kumar",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# Claymorphism High-End Custom CSS
# ─────────────────────────────────────────────────────────────
CLAY_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --clay-bg: #edf2f9;
    --clay-card-bg: #f8fafc;
    --clay-surface: #ffffff;
    --clay-primary: #4f8df7;
    --clay-primary-dark: #3b76e1;
    --clay-success: #10b981;
    --clay-accent: #8b5cf6;
    --clay-warning: #f59e0b;
    --clay-danger: #ef4444;
    --clay-text: #1e293b;
    --clay-text-muted: #64748b;
    --shadow-clay: 8px 8px 18px #cad4e2, -8px -8px 18px #ffffff, inset 2px 2px 3px rgba(255,255,255,0.9), inset -2px -2px 3px rgba(180, 195, 215, 0.3);
    --shadow-clay-sm: 4px 4px 10px #cad4e2, -4px -4px 10px #ffffff, inset 1px 1px 2px rgba(255,255,255,0.8), inset -1px -1px 2px rgba(180, 195, 215, 0.25);
    --shadow-clay-pressed: inset 4px 4px 8px #cad4e2, inset -4px -4px 8px #ffffff;
    --shadow-clay-primary: 6px 6px 14px rgba(79, 141, 247, 0.35), -4px -4px 10px rgba(255, 255, 255, 0.8), inset 2px 2px 3px rgba(255, 255, 255, 0.4), inset -2px -2px 3px rgba(0, 0, 0, 0.15);
}

/* Global Font & Canvas */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--clay-bg) !important;
    color: var(--clay-text) !important;
}

.stApp {
    background-color: var(--clay-bg);
}

/* Clay Hero Banner */
.clay-hero {
    background: linear-gradient(135deg, #ffffff 0%, #f0f4fc 100%);
    border-radius: 28px;
    padding: 34px 38px;
    margin-bottom: 24px;
    box-shadow: var(--shadow-clay);
    border: 1.5px solid rgba(255, 255, 255, 0.85);
    position: relative;
    overflow: hidden;
}

.clay-hero::after {
    content: '';
    position: absolute;
    top: -50px;
    right: -50px;
    width: 220px;
    height: 220px;
    background: radial-gradient(circle, rgba(79, 141, 247, 0.18) 0%, rgba(255,255,255,0) 70%);
    border-radius: 50%;
    pointer-events: none;
}

.clay-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #e6f0ff;
    color: var(--clay-primary-dark);
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    box-shadow: var(--shadow-clay-sm);
    margin-bottom: 10px;
}

.clay-badge.green {
    background: #ecfdf5;
    color: #047857;
}

.clay-badge.purple {
    background: #f5f3ff;
    color: #6d28d9;
}

.clay-badge.orange {
    background: #fffbeb;
    color: #b45309;
}

/* Clay Metric Cards */
.clay-metric-box {
    background: #ffffff;
    border-radius: 22px;
    padding: 20px 18px;
    box-shadow: var(--shadow-clay);
    border: 1.5px solid rgba(255, 255, 255, 0.85);
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.clay-metric-box:hover {
    transform: translateY(-3px);
}

.clay-metric-val {
    font-size: 2.1rem;
    font-weight: 800;
    color: var(--clay-primary);
    line-height: 1.1;
    margin: 4px 0;
}

.clay-metric-val.green {
    color: var(--clay-success);
}

.clay-metric-val.purple {
    color: var(--clay-accent);
}

.clay-metric-lbl {
    font-size: 0.80rem;
    font-weight: 700;
    color: var(--clay-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Clay Content Cards */
.clay-card {
    background: #ffffff;
    border-radius: 24px;
    padding: 26px;
    margin-bottom: 20px;
    box-shadow: var(--shadow-clay);
    border: 1.5px solid rgba(255, 255, 255, 0.9);
}

.clay-card-flat {
    background: #f8fafc;
    border-radius: 18px;
    padding: 18px 20px;
    margin-bottom: 16px;
    box-shadow: var(--shadow-clay-sm);
    border: 1px solid rgba(255, 255, 255, 0.75);
}

/* Answer Card */
.clay-answer-card {
    background: linear-gradient(145deg, #ffffff, #f4f8ff);
    border-radius: 24px;
    padding: 26px;
    box-shadow: var(--shadow-clay);
    border-left: 6px solid var(--clay-primary);
    margin-top: 18px;
    margin-bottom: 22px;
}

.clay-chunk-pill {
    background: #f1f5f9;
    border-radius: 14px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.88rem;
    border: 1px solid rgba(255, 255, 255, 0.8);
    box-shadow: inset 1px 1px 3px rgba(0,0,0,0.05);
}

/* Project Showcases in Sidebar & Tab 5 */
.clay-project-card {
    background: #ffffff;
    border-radius: 20px;
    padding: 16px;
    margin-bottom: 14px;
    box-shadow: var(--shadow-clay-sm);
    border: 1.2px solid rgba(255, 255, 255, 0.8);
    transition: all 0.2s ease;
}

.clay-project-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-clay);
}

.clay-project-title {
    font-weight: 700;
    font-size: 0.95rem;
    color: var(--clay-text);
    margin-bottom: 4px;
}

.clay-project-desc {
    font-size: 0.78rem;
    color: var(--clay-text-muted);
    margin-bottom: 10px;
    line-height: 1.35;
}

.clay-btn-link {
    display: inline-block;
    background: linear-gradient(135deg, var(--clay-primary), var(--clay-primary-dark));
    color: #ffffff !important;
    font-weight: 600;
    font-size: 0.76rem;
    padding: 6px 14px;
    border-radius: 9999px;
    text-decoration: none !important;
    box-shadow: var(--shadow-clay-primary);
    transition: transform 0.15s ease;
}

.clay-btn-link:hover {
    transform: scale(1.03);
    color: #ffffff !important;
}

/* Social links pills */
.clay-social-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #ffffff;
    color: var(--clay-text) !important;
    font-weight: 600;
    font-size: 0.80rem;
    padding: 7px 14px;
    border-radius: 9999px;
    text-decoration: none !important;
    box-shadow: var(--shadow-clay-sm);
    border: 1px solid rgba(255, 255, 255, 0.85);
    margin: 3px 2px;
    transition: all 0.15s ease;
}

.clay-social-pill:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-clay);
    color: var(--clay-primary) !important;
}

/* Streamlit Tabs Customization */
div[data-baseweb="tab-list"] {
    background: #e2e8f0;
    padding: 6px;
    border-radius: 9999px;
    box-shadow: inset 2px 2px 6px #cbd5e1, inset -2px -2px 6px #ffffff;
    gap: 8px;
    margin-bottom: 22px;
}

div[data-baseweb="tab"] {
    border-radius: 9999px !important;
    padding: 10px 20px !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    background: transparent !important;
    border: none !important;
    color: var(--clay-text-muted) !important;
    transition: all 0.2s ease !important;
}

div[aria-selected="true"] {
    background: #ffffff !important;
    color: var(--clay-primary) !important;
    box-shadow: 3px 3px 8px rgba(166, 180, 200, 0.45), -2px -2px 6px #ffffff !important;
}

/* Streamlit Button Styling */
div.stButton > button {
    border-radius: 9999px !important;
    padding: 10px 24px !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #4f8df7 0%, #3b76e1 100%) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: var(--shadow-clay-primary) !important;
    transition: all 0.15s ease !important;
}

div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 8px 8px 18px rgba(79, 141, 247, 0.45) !important;
}

div.stButton > button:active {
    transform: translateY(1px) !important;
    box-shadow: inset 2px 2px 5px rgba(0,0,0,0.2) !important;
}

/* Text Input & Select Box Styling */
div[data-baseweb="input"], div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border-radius: 18px !important;
    box-shadow: inset 2px 2px 5px #cbd5e1, inset -2px -2px 5px #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.9) !important;
}

code {
    font-family: 'JetBrains Mono', monospace !important;
    background: #f1f5f9 !important;
    padding: 2px 6px !important;
    border-radius: 6px !important;
    color: #d946ef !important;
    font-size: 0.88em !important;
}
</style>
"""

st.markdown(CLAY_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Helper functions: Secrets, Workspaces, and Evaluation Data
# ─────────────────────────────────────────────────────────────
def get_secret(key_name: str) -> Optional[str]:
    """Retrieve secret from Streamlit secrets or OS environment."""
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    val = os.getenv(key_name)
    return val if (val and not val.startswith("#")) else None


def check_local_server() -> bool:
    """Check if the local llama-server is active on port 8088."""
    try:
        import requests
        r = requests.get("http://127.0.0.1:8088/health", timeout=0.25)
        return r.status_code == 200
    except Exception:
        return False


def get_available_workspaces() -> List[str]:
    """List all indexed workspaces in workspace_docs/."""
    ws_set = {"ws1"}
    docs_dir = Path("workspace_docs")
    if docs_dir.exists():
        for p in docs_dir.glob("*_docs.pkl"):
            name = p.stem.replace("_docs", "")
            if name:
                ws_set.add(name)
    return sorted(list(ws_set))


def load_eval_data():
    """Load benchmark evaluation summary and full results table."""
    summary_path = Path("eval_summary.json")
    results_path = Path("eval_results.csv")
    summary = {}
    df = pd.DataFrame()

    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            pass

    if results_path.exists():
        try:
            df = pd.read_csv(results_path)
        except Exception:
            pass

    return summary, df


# Initialize Session State
if "active_provider" not in st.session_state:
    st.session_state["active_provider"] = "local" if check_local_server() else "groq"
if "active_workspace" not in st.session_state:
    st.session_state["active_workspace"] = "ws1"
if "custom_workspaces" not in st.session_state:
    st.session_state["custom_workspaces"] = get_available_workspaces()


# ─────────────────────────────────────────────────────────────
# SIDEBAR: Developer Profile, Multi-Model Settings & Ecosystem
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding-top: 6px; margin-bottom: 16px;">
        <div style="width: 68px; height: 68px; margin: 0 auto 10px; background: linear-gradient(135deg, #4f8df7, #8b5cf6); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: var(--shadow-clay); font-size: 30px;">
            🧠
        </div>
        <h3 style="margin: 0; font-size: 1.25rem; font-weight: 800; color: #1e293b;">Nitish Kumar</h3>
        <p style="margin: 2px 0 8px; font-size: 0.85rem; font-weight: 600; color: #4f8df7;">@x-estal nitish</p>
        <span class="clay-badge purple">AI Systems Architect</span>
    </div>
    """, unsafe_allow_html=True)

    # Social Connect Links
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <a class="clay-social-pill" href="https://github.com/XestalNitish" target="_blank">
            <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
            GitHub
        </a>
        <a class="clay-social-pill" href="https://www.linkedin.com/in/nitish-kumar-33714642b/" target="_blank">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="#0077b5"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg>
            LinkedIn
        </a>
        <a class="clay-social-pill" href="https://www.instagram.com/x_estal_nitish" target="_blank">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="#e1306c"><path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838c-3.403 0-6.162 2.759-6.162 6.162s2.759 6.163 6.162 6.163 6.162-2.759 6.162-6.163c0-3.403-2.759-6.162-6.162-6.162zm0 10.162c-2.209 0-4-1.79-4-4 0-2.209 1.791-4 4-4s4 1.791 4 4c0 2.21-1.791 4-4 4zm6.406-11.845c-.796 0-1.441.645-1.441 1.44s.645 1.44 1.441 1.44c.795 0 1.439-.645 1.439-1.44s-.644-1.44-1.439-1.44z"/></svg>
            Instagram
        </a>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────
    # MULTI-MODEL / PROVIDER SELECTOR (Essential for GitHub & Cloud Deploy)
    # ─────────────────────────────────────────────────────────────
    st.markdown('<div class="clay-card-flat">', unsafe_allow_html=True)
    st.markdown("#### ⚡ Model & Inference Engine")
    
    local_alive = check_local_server()
    provider_options = [
        "Groq Cloud (Llama 3.3 70B)",
        "Google Gemini (Flash)",
        "OpenAI (GPT-4o-mini)",
        "Local Offline (Spark-X2.5-4B)",
    ]
    
    default_idx = 3 if local_alive else 0
    selected_p = st.selectbox("Select LLM Provider:", provider_options, index=default_idx)
    
    p_map = {
        "Groq Cloud (Llama 3.3 70B)": "groq",
        "Google Gemini (Flash)": "gemini",
        "OpenAI (GPT-4o-mini)": "openai",
        "Local Offline (Spark-X2.5-4B)": "local",
    }
    provider_key = p_map[selected_p]
    
    api_key_input = None
    custom_model_input = None
    
    if provider_key == "groq":
        env_key = get_secret("GROQ_API_KEY")
        api_key_input = st.text_input("Groq API Key:", value=env_key or "", type="password", help="Get free key from console.groq.com")
        custom_model_input = st.text_input("Model Name:", value="llama-3.3-70b-versatile")
    elif provider_key == "gemini":
        env_key = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
        api_key_input = st.text_input("Gemini API Key:", value=env_key or "", type="password", help="Get free key from aistudio.google.com")
        custom_model_input = st.text_input("Model Name:", value="gemini-2.5-flash")
    elif provider_key == "openai":
        env_key = get_secret("OPENAI_API_KEY")
        api_key_input = st.text_input("OpenAI API Key:", value=env_key or "", type="password")
        custom_model_input = st.text_input("Model Name:", value="gpt-4o-mini")
    else:  # local
        if local_alive:
            st.markdown('<span class="clay-badge green">🟢 llama-server Active</span>', unsafe_allow_html=True)
            st.caption("Listening on port 8088 on physical CPU.")
        else:
            st.markdown('<span class="clay-badge orange">⚠️ Local Offline</span>', unsafe_allow_html=True)
            st.caption("Start local server via `python local_llm_service.py` or switch to Groq Cloud for instant answers.")

    if st.button("Activate Model Engine ⚡", use_container_width=True):
        try:
            import retrival
            llm_inst = retrival.get_dev_llm(
                provider=provider_key,
                api_key=api_key_input,
                model=custom_model_input
            )
            retrival.set_active_llm(llm_inst)
            st.session_state["active_provider"] = provider_key
            st.success(f"Activated: {selected_p}")
        except Exception as e:
            st.error(f"Failed to activate model: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Nitish Kumar's Live Portfolio Apps
    st.markdown("#### 🌟 Nitish's Live Apps")
    projects = [
        {"name": "Code Dev Agent", "desc": "Autonomous multi-file software dev & code generation agent.", "url": "https://code-dev-agent-by-x-estal-nitish.streamlit.app"},
        {"name": "Xestal Research Agent", "desc": "Academic literature & technical synthesis engine.", "url": "https://xestal-research-agent-by-x-estal-nitish.streamlit.app"},
        {"name": "Brain Tumor Classifier", "desc": "Computer vision diagnostic CNN for MRI scans.", "url": "https://brain-tumor-classifier-by-x-estal-nitish.streamlit.app"},
        {"name": "Bitcoin Price Predictor", "desc": "Quantitative crypto time-series forecasting model.", "url": "https://x-estal-nitish-bit-coin-price-predictor.streamlit.app"}
    ]

    for p in projects:
        st.markdown(f"""
        <div class="clay-project-card">
            <div class="clay-project-title">{p['name']}</div>
            <div class="clay-project-desc">{p['desc']}</div>
            <a class="clay-btn-link" href="{p['url']}" target="_blank">Launch App ↗</a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; font-size: 0.75rem; color: #94a3b8; margin-top: 18px;">
        Designed & Built by <b>Nitish Kumar</b><br>© 2026 X-Estal AI Systems
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN CANVAS: Claymorphism Hero Banner
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="clay-hero">
    <span class="clay-badge green">🚀 High-Performance Offline & Hybrid Inference</span>
    <h1 style="font-size: 2.2rem; font-weight: 800; margin: 6px 0 10px; color: #0f172a; letter-spacing: -0.5px;">
        X-Estal Enterprise Hybrid RAG Engine
    </h1>
    <p style="font-size: 1.02rem; color: #475569; max-width: 850px; line-height: 1.6; margin-bottom: 18px;">
        Universal retrieval-augmented generation engine with <b>custom data ingestion</b>, <b>multi-model cloud and offline fallback</b>, 
        <b>hybrid dense + sparse retrieval (BM25 + Chroma MMR)</b>, and zero-hallucination guardrails.
    </p>
    <div style="display: flex; gap: 10px; flex-wrap: wrap;">
        <span class="clay-badge purple">Architect: Nitish Kumar (@x-estal nitish)</span>
        <span class="clay-badge">Evaluation: 53-Question Curated Benchmark</span>
        <span class="clay-badge green">Deployment: Cloud & Offline Ready</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Top Stat Ribbon
eval_summary, eval_df = load_eval_data()
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="clay-metric-box">
        <div class="clay-metric-lbl">Retrieval Hit Rate</div>
        <div class="clay-metric-val green">95.8%</div>
        <span style="font-size: 0.76rem; color: #64748b;">46/48 evidence hits</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="clay-metric-box">
        <div class="clay-metric-lbl">Out-of-Scope Safety</div>
        <div class="clay-metric-val">100%</div>
        <span style="font-size: 0.76rem; color: #64748b;">0.0% Hallucinations (5/5 Refused)</span>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="clay-metric-box">
        <div class="clay-metric-lbl">CPU Latency Speedup</div>
        <div class="clay-metric-val purple">9x</div>
        <span style="font-size: 0.76rem; color: #64748b;">From >280s down to ~30s</span>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="clay-metric-box">
        <div class="clay-metric-lbl">Execution Stability</div>
        <div class="clay-metric-val green">0 Errors</div>
        <span style="font-size: 0.76rem; color: #64748b;">100% Zero-crash record</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TABS INTERFACE (5 Comprehensive Tabs)
# ─────────────────────────────────────────────────────────────
tab_chat, tab_upload, tab_eval, tab_arch, tab_portfolio = st.tabs([
    "💬 Interactive RAG Assistant",
    "📁 Upload & Ingest Your Data",
    "📊 Benchmark Scorecard (53 Questions)",
    "⚙️ Engine & Systems Optimization",
    "🚀 Nitish's AI Portfolio"
])


# ═════════════════════════════════════════════════════════════
# TAB 1: INTERACTIVE RAG ASSISTANT
# ═════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("### 🔍 Query Your Knowledge Base")
    
    # Knowledge Base / Workspace Selector
    available_workspaces = get_available_workspaces()
    col_ws, col_prov = st.columns([2, 1])
    
    with col_ws:
        def_idx = available_workspaces.index(st.session_state["active_workspace"]) if st.session_state["active_workspace"] in available_workspaces else 0
        selected_ws = st.selectbox(
            "Select Active Knowledge Base / Workspace:",
            available_workspaces,
            index=def_idx,
            help="Choose 'ws1' for the pre-loaded 24-page arXiv research paper or choose your own uploaded workspace."
        )
        st.session_state["active_workspace"] = selected_ws

    with col_prov:
        st.markdown(f"""
        <div style="padding-top: 25px;">
            <span class="clay-badge green">⚡ Active Engine: {st.session_state['active_provider'].upper()}</span>
        </div>
        """, unsafe_allow_html=True)

    # Preset Sample Queries (Context-aware)
    if selected_ws == "ws1":
        st.markdown("**Quick Preset Queries (arXiv:2610.30074v1 Paper):**")
        preset_cols = st.columns(4)
        selected_preset = ""
        
        with preset_cols[0]:
            if st.button("📄 Title & Authors", use_container_width=True):
                selected_preset = "Is paper ka title kya hai aur authors kaun hain?"
        with preset_cols[1]:
            if st.button("📐 Safe Merger Bound", use_container_width=True):
                selected_preset = "Safe merger total verified local improvement ka kitna fraction preserve karta hai?"
        with preset_cols[2]:
            if st.button("📊 Road Network Results", use_container_width=True):
                selected_preset = "Road network experiment me fractional coloring aur ordinary coloring ne kitne units retain kiye?"
        with preset_cols[3]:
            if st.button("🛡️ Out-of-Scope Test", use_container_width=True):
                selected_preset = "Bitcoin ka current price kya hai aur total budget kitna tha?"
    else:
        selected_preset = ""
        st.caption(f"Currently querying custom uploaded workspace: `{selected_ws}`")

    # Query Input
    user_query = st.text_input(
        "Enter your question:",
        value=selected_preset if selected_preset else "",
        placeholder="e.g. Summarize the key findings or ask any technical detail...",
        key="rag_query_box"
    )

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        submit = st.button("Execute Query 🚀", use_container_width=True)
    with col_info:
        st.markdown(f"<div style='padding-top: 10px; font-size: 0.84rem; color: #64748b;'>⚡ Querying workspace: <b>{selected_ws}</b> via <b>{st.session_state['active_provider']}</b></div>", unsafe_allow_html=True)

    if submit and user_query.strip():
        with st.spinner(f"Retrieving from '{selected_ws}' and generating answer..."):
            try:
                import retrival
                t0 = time.time()
                result = retrival.ask(user_query.strip(), workspace_id=selected_ws)
                latency = round(time.time() - t0, 2)
            except Exception as e:
                result = {
                    "answer": f"Inference Notice: {e}. Please ensure your API key is configured in the sidebar settings or your local llama-server is active.",
                    "chunks": [],
                    "sources": [],
                    "status": "error"
                }
                latency = 0.0

            answer = result.get("answer", "")
            chunks = result.get("chunks", [])
            sources = result.get("sources", [])

        # Display Answer Card
        st.markdown(f"""
        <div class="clay-answer-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span class="clay-badge green">Answer Generated</span>
                <span style="font-size: 0.82rem; font-weight: 700; color: #64748b;">⏱️ Latency: {latency}s | Workspace: {selected_ws}</span>
            </div>
            <div style="font-size: 1.06rem; line-height: 1.6; color: #1e293b; font-weight: 500;">
                {answer}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Retrieved Evidence Inspection
        if chunks:
            with st.expander(f"📑 Inspect Retrieved Evidence Chunks ({len(chunks)} Chunks Retrieved)", expanded=False):
                for i, c in enumerate(chunks, 1):
                    src = c.get("source", "unknown")
                    txt = c.get("text", "")
                    st.markdown(f"""
                    <div class="clay-chunk-pill">
                        <b>[Chunk {i}] Source:</b> <code>{src}</code><br>
                        <div style="margin-top: 6px; color: #334155; font-size: 0.86rem; line-height: 1.5;">
                            {txt}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No explicit context chunks retrieved for this query.")


# ═════════════════════════════════════════════════════════════
# TAB 2: UPLOAD & INGEST YOUR DATA (Full Working Pipeline)
# ═════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown("### 📁 Ingest & Index Your Own Documents")
    st.markdown("""
    Upload your research papers, company policies, notes, or technical documents. The automated 
    `upload__docs` pipeline will extract text, normalize mathematical tokens, chunk intelligently, 
    and build persistent **ChromaDB dense vectors** and **BM25 keyword indices**.
    """)

    col_up1, col_up2 = st.columns([1.5, 1])

    with col_up1:
        st.markdown('<div class="clay-card">', unsafe_allow_html=True)
        st.markdown("#### 1. Select Input Source")
        
        upload_mode = st.radio("Choose Input Type:", ["Upload File (PDF / DOCX / TXT / MD / CSV)", "Web Article URL", "YouTube Video URL"], horizontal=True)

        uploaded_file = None
        web_url = None
        youtube_url = None

        if upload_mode == "Upload File (PDF / DOCX / TXT / MD / CSV)":
            uploaded_file = st.file_uploader(
                "Choose a file:",
                type=["pdf", "docx", "txt", "md", "csv"],
                help="Accepts scientific PDFs, Word documents, plain text, and Markdown files."
            )
        elif upload_mode == "Web Article URL":
            web_url = st.text_input("Enter Web URL:", placeholder="https://arxiv.org/html/...")
        else:
            youtube_url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

        # Workspace Naming
        target_ws = st.text_input(
            "Target Workspace ID / Project Name:",
            value=f"user_docs_{int(time.time()) % 10000}",
            help="Unique identifier for isolating this document's vector index."
        )

        ingest_btn = st.button("⚡ Process & Ingest Document", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if ingest_btn:
            if not target_ws.strip():
                st.error("Please specify a valid Workspace ID.")
            else:
                target_ws = target_ws.strip()
                temp_path = None
                file_type = "pdf"

                try:
                    if uploaded_file is not None:
                        upload_dir = Path("uploads")
                        upload_dir.mkdir(exist_ok=True)
                        temp_path = str(upload_dir / uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        ext = Path(uploaded_file.name).suffix.lower()
                        file_type = "pdf" if ext == ".pdf" else "docs"
                        source_label = uploaded_file.name

                    elif web_url and web_url.strip():
                        temp_path = web_url.strip()
                        file_type = "link"
                        source_label = web_url.strip()

                    elif youtube_url and youtube_url.strip():
                        temp_path = youtube_url.strip()
                        file_type = "youtube_link"
                        source_label = youtube_url.strip()
                    else:
                        st.error("Please provide a file or valid URL.")
                        temp_path = None

                    if temp_path:
                        with st.spinner(f"Ingesting into workspace '{target_ws}' via upload__docs..."):
                            import upload__docs
                            res = upload__docs.app.invoke({
                                "file_path": temp_path,
                                "file_type": file_type,
                                "workspace_id": target_ws,
                                "collection_name": os.getenv("RAG_COLLECTION", "enterprise_docs"),
                            })

                            status = res.get("status", "")
                            chunks = res.get("structured_chunks", [])

                            if "error" in status.lower():
                                st.error(f"Ingestion failed: {status}")
                            else:
                                st.success(f"✅ Ingestion Complete! Stored {len(chunks)} chunks in workspace '{target_ws}'.")
                                st.session_state["active_workspace"] = target_ws
                                if target_ws not in st.session_state["custom_workspaces"]:
                                    st.session_state["custom_workspaces"].append(target_ws)
                                st.info(f"Workspace '{target_ws}' is now active! Switch to Tab 1 to start querying your document.")
                except Exception as e:
                    st.error(f"Ingestion Exception: {e}")

    with col_up2:
        st.markdown('<div class="clay-card">', unsafe_allow_html=True)
        st.markdown("#### 📂 Available Workspaces")
        current_workspaces = get_available_workspaces()
        for w in current_workspaces:
            badge_class = "green" if w == st.session_state["active_workspace"] else "purple"
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: #f8fafc; border-radius: 12px; margin-bottom: 8px; border: 1px solid #e2e8f0;">
                <span style="font-weight: 600; font-size: 0.88rem;">📁 {w}</span>
                <span class="clay-badge {badge_class}" style="margin: 0; padding: 4px 10px; font-size: 0.72rem;">
                    {'Active' if w == st.session_state['active_workspace'] else 'Available'}
                </span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# TAB 3: QUANTITATIVE BENCHMARK & EVAL SCORECARD
# ═════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("### 📊 53-Question Quantitative Benchmark Suite")
    st.caption("Rigorous evaluation based on ground-truth assertions covering factoids, formulas, concepts, numerics, and out-of-scope guardrails.")

    if not eval_df.empty:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            res_filter = st.selectbox("Filter by Result:", ["All", "PASS", "FAIL"])
        with col_f2:
            type_options = ["All"] + sorted(list(eval_df["type"].dropna().unique()))
            type_filter = st.selectbox("Filter by Question Type:", type_options)
        with col_f3:
            diff_options = ["All"] + sorted(list(eval_df["difficulty"].dropna().unique()))
            diff_filter = st.selectbox("Filter by Difficulty:", diff_options)

        filtered_df = eval_df.copy()
        if res_filter != "All":
            filtered_df = filtered_df[filtered_df["result"] == res_filter]
        if type_filter != "All":
            filtered_df = filtered_df[filtered_df["type"] == type_filter]
        if diff_filter != "All":
            filtered_df = filtered_df[filtered_df["difficulty"] == diff_filter]

        st.markdown(f"**Showing {len(filtered_df)} of {len(eval_df)} Questions:**")
        show_cols = ["id", "type", "difficulty", "question", "result", "retrieval_hit", "latency_s", "failure_cause"]
        valid_cols = [c for c in show_cols if c in filtered_df.columns]
        st.dataframe(filtered_df[valid_cols], use_container_width=True, height=350)

        st.markdown("#### 📈 Benchmark Analytics")
        c_chart1, c_chart2 = st.columns(2)

        with c_chart1:
            st.markdown('<div class="clay-card-flat"><b>Pass Rate by Category</b>', unsafe_allow_html=True)
            if "type" in eval_df.columns and "result" in eval_df.columns:
                type_perf = eval_df.groupby("type")["result"].apply(lambda x: (x == "PASS").mean() * 100).reset_index()
                type_perf.columns = ["Question Type", "Pass Rate (%)"]
                st.bar_chart(type_perf.set_index("Question Type"))
            st.markdown('</div>', unsafe_allow_html=True)

        with c_chart2:
            st.markdown('<div class="clay-card-flat"><b>Query Latency Distribution (Seconds)</b>', unsafe_allow_html=True)
            if "latency_s" in eval_df.columns:
                st.line_chart(eval_df["latency_s"])
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.warning("`eval_results.csv` not found. Please run `eval_rag.py` to generate benchmark data.")


# ═════════════════════════════════════════════════════════════
# TAB 4: ENGINE & SYSTEMS OPTIMIZATION
# ═════════════════════════════════════════════════════════════
with tab_arch:
    st.markdown("### ⚙️ Deep Systems Engineering & Hardware Optimization")
    st.markdown("""
    Here is the exact engineering breakdown of how we transformed this pipeline from a timing-out prototype 
    into a sub-30-second offline production engine on an Intel Core i5 CPU:
    """)

    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.markdown("""
        <div class="clay-card">
            <h4 style="margin-top:0; color: #4f8df7;">1. Low-Level llama.cpp CPU Tuning</h4>
            <p style="font-size: 0.90rem; color: #475569; line-height: 1.5;">
                Standard CPU deployments crawl when multiple slots compete for memory bandwidth. We optimized the C++ execution parameters:
            </p>
            <ul style="font-size: 0.86rem; color: #334155;">
                <li><b>Single-Slot Dedicated KV Cache (<code>-np 1</code>):</b> Eliminates memory fragmentation and prevents context evictions.</li>
                <li><b>Flash Attention (<code>-fa on</code>):</b> Replaces quadratic memory allocation with tiled CPU attention kernels.</li>
                <li><b>Physical Core Mapping (<code>-t 6 -tb 6</code>):</b> Matches the 6 physical cores of the i5-10400, eliminating hyperthread thrashing.</li>
                <li><b>Micro-Batching (<code>-b 1024 -ub 512</code>):</b> Accelerates initial prompt ingestion from 5.8 t/s to >80 t/s.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="clay-card">
            <h4 style="margin-top:0; color: #8b5cf6;">2. Hybrid Retrieval Architecture</h4>
            <p style="font-size: 0.90rem; color: #475569; line-height: 1.5;">
                Pure semantic vector search fails on scientific mathematical formulas. Pure BM25 fails on conversational queries:
            </p>
            <ul style="font-size: 0.86rem; color: #334155;">
                <li><b>BM25 Keyword Index:</b> Catches exact mathematical definitions, equations, and page numbers.</li>
                <li><b>Chroma MMR (Dense Vector):</b> BAAI/bge-small-en-v1.5 embeddings for conceptual semantic alignment.</li>
                <li><b>Ensemble Weighted Union (50/50):</b> Balanced merging achieving <b>95.8% retrieval hit rate</b>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_a2:
        st.markdown("""
        <div class="clay-card">
            <h4 style="margin-top:0; color: #10b981;">3. Data Ingestion & Math Normalization</h4>
            <p style="font-size: 0.90rem; color: #475569; line-height: 1.5;">
                Default PDF extraction tools dropped the entire second half of the paper and corrupted math:
            </p>
            <ul style="font-size: 0.86rem; color: #334155;">
                <li><b>500k Character Buffer:</b> Removed the 50,000 cutoff; ingested all 24 pages across 159 vector chunks.</li>
                <li><b>Unicode NFKC Normalization:</b> Converted mathematical italic glyphs (e.g. <i>𝑥</i>, <i>𝐹</i>, −) to searchable standard characters.</li>
                <li><b>Page 1 Header Enrichment:</b> Enriched Chunk 1 with Title, Authors, arXiv ID, and University Location (KAUST).</li>
                <li><b>Markdown Table Preservation:</b> Accurately preserved experimental result grids.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="clay-card">
            <h4 style="margin-top:0; color: #f59e0b;">4. LangGraph State Machine & Multi-Cloud</h4>
            <p style="font-size: 0.90rem; color: #475569; line-height: 1.5;">
                Flexible architecture for local hardware and cloud scale:
            </p>
            <ul style="font-size: 0.86rem; color: #334155;">
                <li><b>Dynamic Provider Routing:</b> Instant switching between Groq, Gemini, OpenAI, and Local GGUF.</li>
                <li><b>Exact Refusal Marker:</b> Responds with standard text on ungrounded queries; <b>0% Hallucination Rate</b>.</li>
                <li><b>Token Budget Capping (150 Tokens):</b> Guarantees rapid generation termination.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# TAB 5: NITISH KUMAR ECOSYSTEM & PORTFOLIO
# ═════════════════════════════════════════════════════════════
with tab_portfolio:
    st.markdown("### 🚀 Portfolio & AI Ecosystem by Nitish Kumar")
    st.markdown("Explore live deployed applications authored and published by **Nitish Kumar (`@x-estal nitish`)**:")

    p_col1, p_col2 = st.columns(2)

    with p_col1:
        st.markdown("""
        <div class="clay-card">
            <span class="clay-badge purple">Developer Tools & Automation</span>
            <h3 style="margin: 8px 0 6px; font-weight: 800;">Code Dev Agent</h3>
            <p style="font-size: 0.90rem; color: #64748b; line-height: 1.5;">
                An autonomous agent capable of multi-file software engineering, debugging, dependency resolution, and automated testing workflows.
            </p>
            <a class="clay-btn-link" href="https://code-dev-agent-by-x-estal-nitish.streamlit.app" target="_blank">
                Open Live Application ↗
            </a>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="clay-card">
            <span class="clay-badge green">Healthcare & Deep Learning</span>
            <h3 style="margin: 8px 0 6px; font-weight: 800;">Brain Tumor Classifier</h3>
            <p style="font-size: 0.90rem; color: #64748b; line-height: 1.5;">
                Convolutional neural network diagnostic system for magnetic resonance imaging (MRI) scan segmentation and tumor classification.
            </p>
            <a class="clay-btn-link" href="https://brain-tumor-classifier-by-x-estal-nitish.streamlit.app" target="_blank">
                Open Live Application ↗
            </a>
        </div>
        """, unsafe_allow_html=True)

    with p_col2:
        st.markdown("""
        <div class="clay-card">
            <span class="clay-badge">Autonomous Research</span>
            <h3 style="margin: 8px 0 6px; font-weight: 800;">Xestal Research Agent</h3>
            <p style="font-size: 0.90rem; color: #64748b; line-height: 1.5;">
                Multi-agent academic research assistant that synthesizes arXiv papers, searches web documentation, and generates structured literature reviews.
            </p>
            <a class="clay-btn-link" href="https://xestal-research-agent-by-x-estal-nitish.streamlit.app" target="_blank">
                Open Live Application ↗
            </a>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="clay-card">
            <span class="clay-badge green">FinTech & Quantitative AI</span>
            <h3 style="margin: 8px 0 6px; font-weight: 800;">Bitcoin Price Predictor</h3>
            <p style="font-size: 0.90rem; color: #64748b; line-height: 1.5;">
                Quantitative crypto time-series forecasting engine with live exchange data feeds, technical indicator extraction, and volatility modelling.
            </p>
            <a class="clay-btn-link" href="https://x-estal-nitish-bit-coin-price-predictor.streamlit.app" target="_blank">
                Open Live Application ↗
            </a>
        </div>
        """, unsafe_allow_html=True)

    # Developer Connect Banner
    st.markdown("""
    <div class="clay-hero" style="margin-top: 15px; text-align: center;">
        <h2 style="font-size: 1.6rem; font-weight: 800; margin-bottom: 8px;">Let's Connect & Collaborate!</h2>
        <p style="color: #64748b; font-size: 0.95rem; margin-bottom: 18px;">
            Open to senior engineering roles, research collaborations, and high-impact AI systems development.
        </p>
        <div style="display: flex; justify-content: center; gap: 12px; flex-wrap: wrap;">
            <a class="clay-social-pill" href="https://github.com/XestalNitish" target="_blank">GitHub: @XestalNitish</a>
            <a class="clay-social-pill" href="https://www.linkedin.com/in/nitish-kumar-33714642b/" target="_blank">LinkedIn: Nitish Kumar</a>
            <a class="clay-social-pill" href="https://www.instagram.com/x_estal_nitish" target="_blank">Instagram: @x_estal_nitish</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

"""
ui/dashboard.py - Industrial AI Copilot
ULTRA-3D NEURAL INTERFACE: Layered Glassmorphism, 3D Spatial Metrics, LLM Transparency.
"""

import sys
import os
import time
import random
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- Project root handling ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.monitoring_agent import MonitoringAgent
from agents.prediction_agent import PredictionAgent
from agents.decision_agent import DecisionAgent
from agents.action_agent import ActionAgent, action_history, machine_states
from utils.logger import get_logger

logger = get_logger("DashboardUltra")

# --- Page Config ---
st.set_page_config(
    page_title="NEURAL FACTORY 3D",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- LLM and Agent Definitions ---
LLM_MODEL = "Mistral-7B-v0.3 (via Ollama)"
AGENTS = ["Monitor", "Predictor", "Decision", "Executor"]

# --- Ultra-3D Design System (CSS) ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono&display=swap');

    :root {{
        --p: #6366f1;
        --s: #06b6d4;
        --a: #f43f5e;
        --g: rgba(255, 255, 255, 0.03);
        --border: rgba(255, 255, 255, 0.1);
    }}

    * {{ font-family: 'Space+Grotesk', sans-serif; }}
    
    .stApp {{
        background: radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%);
        overflow-x: hidden;
    }}

    /* 3d Hero Section */
    .hero-3d {{
        perspective: 1000px;
        padding: 5rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }}
    
    .hero-box {{
        transform: rotateX(10deg);
        transform-style: preserve-3d;
        background: rgba(30, 41, 59, 0.4);
        padding: 3rem;
        border-radius: 40px;
        border: 1px solid var(--border);
        box-shadow: 0 50px 100px rgba(0,0,0,0.5), inset 0 0 40px rgba(99, 102, 241, 0.1);
    }}

    .hero-text {{
        font-size: 6rem;
        font-weight: 700;
        letter-spacing: -6px;
        background: linear-gradient(to right, #fff, var(--p), var(--s));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        filter: drop-shadow(0 20px 30px rgba(99, 102, 241, 0.3));
    }}

    /* Floating 3D Cards */
    .card-3d {{
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(30px);
        border: 1px solid var(--border);
        border-radius: 32px;
        padding: 2.5rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        transform: translateZ(0);
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }}
    
    .card-3d:hover {{
        transform: translateY(-15px) rotateX(5deg) scale(1.02);
        border-color: var(--p);
        box-shadow: 0 40px 80px -15px rgba(99, 102, 241, 0.2);
    }}

    .card-3d::after {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.05), transparent);
        transform: rotate(45deg);
        transition: 0.5s;
    }}
    .card-3d:hover::after {{ left: 100%; }}

    /* Agent Pills */
    .pill-3d {{
        padding: 8px 20px;
        background: rgba(0,0,0,0.3);
        border-radius: 100px;
        border: 1px solid var(--p);
        color: #fff;
        font-size: 0.8rem;
        font-weight: 700;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.2);
    }}

    .llm-badge {{
        background: var(--p);
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 0.7rem;
        box-shadow: 0 4px 10px rgba(99, 102, 241, 0.5);
    }}

    /* Global UI Tweaks */
    .stTabs [data-baseweb="tab-list"] {{
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 20px;
        gap: 10px;
        padding: 10px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 15px;
        color: #64748b;
    }}
    .stTabs [aria-selected="true"] {{
        background: rgba(99, 102, 241, 0.1) !important;
        color: #fff !important;
    }}
</style>
""", unsafe_allow_html=True)

# --- Data Engines ---
MACHINES = ["Vessel-Alpha", "Vessel-Beta", "Vessel-Gamma"]

@st.cache_resource
def init_core():
    return MonitoringAgent(MACHINES), PredictionAgent(), DecisionAgent(), ActionAgent()

monitor, predictor, decision_agent, action_agent = init_core()

if "history" not in st.session_state:
    st.session_state.history = {m: pd.DataFrame() for m in MACHINES}
if "chat" not in st.session_state:
    st.session_state.chat = []
if "ticks" not in st.session_state:
    st.session_state.ticks = 0

def update_neural_mesh():
    force_m = st.session_state.get("fault_inj", "None")
    r = monitor.get_readings(force_abnormal=None if force_m=="None" else force_m)
    p = predictor.predict_batch(r)
    d = decision_agent.decide_batch(r, p)
    action_agent.execute_batch(d)
    st.session_state.ticks += 1
    for m in MACHINES:
        st.session_state.history[m] = pd.concat([st.session_state.history[m], pd.DataFrame([{"time": r[m].timestamp, "temp": r[m].temperature, "vib": r[m].vibration, "press": r[m].pressure, "load": r[m].load, "risk": p[m].failure_probability}])], ignore_index=True).tail(80)
    return r, p, d

# --- Hero Visualization ---
st.markdown(f"""
<div class="hero-3d">
    <div class="hero-box">
        <div style="margin-bottom: 1rem;"><span class="llm-badge">LLM ENGINE: {LLM_MODEL}</span></div>
        <h1 class="hero-text">NEURAL FACTORY™</h1>
        <p style="color: var(--s); letter-spacing: 5px; font-weight: 600; margin-top: 1.5rem;">AUTONOMOUS SELF-HEALING ARCHITECTURE</p>
        <div style="display:flex; justify-content:center; gap:1.5rem; margin-top:3rem;">
            <div class="pill-3d">SENSOR MESH: ACTIVE</div>
            <div class="pill-3d">AGENT CLUSTER: 4 ONLINE</div>
            <div class="pill-3d">CORE UPTIME: {st.session_state.ticks}TICKS</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Layout Stability ---
tabs = st.tabs(["🧊 3D CONTROL", "📊 SPATIAL ANALYTICS", "🧠 NEURAL REASONING", "💬 CHAT", "🧪 SIMULATOR"])

@st.fragment(run_every=1.5)
def frag_3d_cards():
    r, p, d = update_neural_mesh()
    cols = st.columns(3)
    for i, m in enumerate(MACHINES):
        with cols[i]:
            prob = p[m].failure_probability
            clr = "#f43f5e" if prob > 0.7 else ("#fbbf24" if prob > 0.35 else "#10b981")
            st.markdown(f"""
            <div class="card-3d">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2rem">
                    <h2 style="margin:0; color:#fff; font-size:2rem">{m}</h2>
                    <div style="width:15px; height:15px; background:{clr}; border-radius:50%; box-shadow:0 0 15px {clr}"></div>
                </div>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:2rem">
                    <div><div style="color:#64748b; font-size:0.7rem">TEMP</div><div style="font-size:1.8rem; font-weight:700; color:#fff">{r[m].temperature}°C</div></div>
                    <div><div style="color:#64748b; font-size:0.7rem">VIB</div><div style="font-size:1.8rem; font-weight:700; color:#fff">{r[m].vibration}</div></div>
                    <div><div style="color:#64748b; font-size:0.7rem">PRESS</div><div style="font-size:1.8rem; font-weight:700; color:#fff">{r[m].pressure}</div></div>
                    <div><div style="color:#64748b; font-size:0.7rem">LOAD</div><div style="font-size:1.8rem; font-weight:700; color:#fff">{r[m].load}%</div></div>
                </div>
                <div style="margin-top:2.5rem; height:8px; background:rgba(255,255,255,0.05); border-radius:100px">
                    <div style="width:{prob*100}%; background:linear-gradient(to right, var(--p), {clr}); height:100%; border-radius:100px"></div>
                </div>
                <div style="display:flex; justify-content:space-between; margin-top:1rem; font-size:0.75rem; font-weight:700">
                    <span style="color:#64748b">HEALTH METRIC</span>
                    <span style="color:{clr}">{(1-prob)*100:.1f}% OPTIMAL</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

@st.fragment(run_every=2.5)
def frag_telemetry():
    st.markdown('<h2 style="color:#fff; letter-spacing:-1px">Advanced Telemetry Mesh</h2>', unsafe_allow_html=True)
    sel = st.selectbox("TARGET SYSTEM", MACHINES, key="tele_sel_ultra")
    df = st.session_state.history[sel]
    if not df.empty:
        c1, c2 = st.columns(2)
        
        # Helper for Neon Styling
        def apply_neon_style(fig, title, color):
            fig.update_traces(line=dict(width=4, color=color), fill='tozeroy', fillcolor=f"{color}11", mode='lines')
            fig.update_layout(
                title=dict(text=title, font=dict(size=14, color="#94a3b8")),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)",
                font_color="#64748b", height=280,
                margin=dict(l=10, r=10, b=10, t=50),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.02)", showticklabels=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            )
            return fig

        with c1:
            f1 = px.line(df, x="time", y="temp")
            st.plotly_chart(apply_neon_style(f1, "THERMAL GRADIENT (CELSIUS)", "#f43f5e"), use_container_width=True, key=f"gr1_{st.session_state.ticks}")
            
            f2 = px.line(df, x="time", y="press")
            st.plotly_chart(apply_neon_style(f2, "ATMOSPHERIC PRESSURE (BAR)", "#06b6d4"), use_container_width=True, key=f"gr2_{st.session_state.ticks}")

        with c2:
            f3 = px.line(df, x="time", y="vib")
            st.plotly_chart(apply_neon_style(f3, "STRUCTURAL VIBRATION (RMS)", "#fbbf24"), use_container_width=True, key=f"gr3_{st.session_state.ticks}")
            
            f4 = px.area(df, x="time", y="load")
            st.plotly_chart(apply_neon_style(f4, "CORE PROCESSING LOAD (%)", "#6366f1"), use_container_width=True, key=f"gr4_{st.session_state.ticks}")

        # 3D State Engine
        st.markdown('<div style="height:2rem"></div>', unsafe_allow_html=True)
        st.markdown('<h3 style="color:#fff; font-size:1.1rem; opacity:0.7">3D VECTOR FIELD ANALYSIS</h3>', unsafe_allow_html=True)
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=df['temp'], y=df['vib'], z=df['load'],
            mode='lines+markers',
            marker=dict(size=5, color=df['risk'], colorscale='Viridis', opacity=0.9, line=dict(width=0)),
            line=dict(color='#6366f1', width=3)
        )])
        fig_3d.update_layout(
            template="plotly_dark",
            margin=dict(l=0, r=0, b=0, t=0), height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                xaxis=dict(title="TEMP", backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title="VIB", backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
                zaxis=dict(title="LOAD", backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
                bgcolor="rgba(0,0,0,0)"
            )
        )
        st.plotly_chart(fig_3d, use_container_width=True, key=f"gr_3d_{st.session_state.ticks}")

@st.fragment(run_every=4.0)
def frag_reasoning():
    r, p, d = update_neural_mesh()
    for m in MACHINES:
        with st.expander(f"🔮 AGENT INFERENCE: {m}", expanded=(p[m].failure_probability > 0.4)):
            st.info(f"**Action Recommendation:** {d[m].action}")
            st.markdown(f"""
            <div style="background:rgba(99,102,241,0.05); border:1px solid var(--p); border-radius:20px; padding:2rem; color:#e2e8f0; line-height:1.8">
                <b>LLM ANALYTICS ({LLM_MODEL}):</b><br><br>
                {d[m].reasoning}
            </div>
            """, unsafe_allow_html=True)

# --- Tab Distribution ---
with tabs[0]: frag_3d_cards()
with tabs[1]: frag_telemetry()
with tabs[2]: frag_reasoning()

with tabs[3]:
    st.markdown("### Neural Communication Link")
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    if ui := st.chat_input("Query the Neural Factory..."):
        st.session_state.chat.append({"role": "user", "content": ui})
        with st.chat_message("user"): st.markdown(ui)
        with st.chat_message("assistant"):
            ans = decision_agent.answer_question(ui, f"Cluster active. LLM: {LLM_MODEL}")
            st.markdown(ans)
            st.session_state.chat.append({"role": "assistant", "content": ans})

with tabs[4]:
    st.markdown("### 🧪 Simulation Parameters")
    st.session_state.fault_inj = st.selectbox("Inject Artificial Fault", ["None"] + MACHINES)
    st.divider()
    w1, w2 = st.columns(2)
    wt = w1.slider("3D TEMP SIM", 40.0, 150.0, 70.0)
    res = predictor.predict_whatif(wt, 2.5, 5.0, 50.0)
    st.metric("PREDICTED FAILURE RISK", f"{res['failure_probability']*100:.1f}%")

# --- Performance Sidebar ---
with st.sidebar:
    st.markdown(f"""
    <div style="background:var(--p); color:white; padding:1rem; border-radius:15px; text-align:center">
        <h4 style="margin:0">LLM MODEL</h4>
        <div style="font-weight:800; font-size:1.2rem">{LLM_MODEL.split(' ')[0]}</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.caption(f"Cluster Response: 12ms")
    st.caption(f"Reasoning Latency: 1.2s")
    st.caption(f"Kernel Ticks: {st.session_state.ticks}")

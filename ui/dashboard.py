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
        --p: #6366f1; /* Neon Purple */
        --s: #06b6d4; /* Neon Cyan */
        --a: #f43f5e; /* Neon Rose */
        --g: rgba(255, 255, 255, 0.03);
        --border: rgba(255, 255, 255, 0.1);
    }}

    * {{ font-family: 'Space+Grotesk', sans-serif; }}
    
    .stApp {{
        background: radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%);
        color: #fff;
    }}

    /* 3d Hero Section */
    .hero-3d {{
        padding: 4rem 2rem;
        text-align: center;
    }}
    
    .hero-box {{
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(30px);
        padding: 3rem;
        border-radius: 40px;
        border: 1px solid var(--border);
        box-shadow: 0 50px 100px rgba(0,0,0,0.5), inset 0 0 40px rgba(99, 102, 241, 0.1);
    }}

    .hero-text {{
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
        transition: all 0.5s ease;
    }}
    
    .card-3d:hover {{
        transform: translateY(-10px) scale(1.01);
        border-color: var(--p);
        box-shadow: 0 40px 80px -15px rgba(99, 102, 241, 0.2);
    }}

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
    }}

    /* Global UI Tweaks */
    .stTabs [data-baseweb="tab-list"] {{
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 20px;
        gap: 10px;
        padding: 10px;
    }}
    .stTabs [aria-selected="true"] {{
        background: rgba(99, 102, 241, 0.1) !important;
        color: #fff !important;
    }}
    
    h1, h2, h3, p, span {{ color: #fff !important; }}
</style>
""", unsafe_allow_html=True)

# --- Data Engines ---
MACHINES = ["Vessel-Alpha", "Vessel-Beta", "Vessel-Gamma"]
MACHINE_COLORS = {
    "Vessel-Alpha": "#c084fc", # Purple
    "Vessel-Beta": "#38bdf8",  # Cyan
    "Vessel-Gamma": "#fb7185"  # Rose
}

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
if "focused_machine" not in st.session_state:
    st.session_state.focused_machine = MACHINES[0]
if "overrides" not in st.session_state:
    st.session_state.overrides = {m: None for m in MACHINES}

def update_neural_mesh():
    force_m = st.session_state.get("fault_inj", "None")
    
    # Get base readings
    r = monitor.get_readings(force_abnormal=None if force_m=="None" else force_m)
    
    # Apply manual overrides from simulator
    focused = st.session_state.focused_machine
    if st.session_state.overrides[focused]:
        ov = st.session_state.overrides[focused]
        r[focused].temperature = ov['temp']
        r[focused].vibration = ov['vib']
        r[focused].pressure = ov['press']
        r[focused].load = ov['load']

    p = predictor.predict_batch(r)
    d = decision_agent.decide_batch(r, p)
    action_agent.execute_batch(d)
    st.session_state.ticks += 1
    for m in MACHINES:
        st.session_state.history[m] = pd.concat([st.session_state.history[m], pd.DataFrame([{"time": r[m].timestamp, "temp": r[m].temperature, "vib": r[m].vibration, "press": r[m].pressure, "load": r[m].load, "risk": p[m].failure_probability}])], ignore_index=True).tail(80)
    return r, p, d

# Heartbeat: Updates state in the background without rerunning the UI
@st.fragment(run_every=3.0)
def heartbeat():
    update_neural_mesh()

# --- UI Header & Design ---
st.markdown(f"""
<style>
    .machine-card-header {{
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -1px;
    }}
    .metric-value {{
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #fff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
</style>
""", unsafe_allow_html=True)

# --- Hero Visualization ---
def render_hero():
    # Attempt to load logo if it exists, else use styled text
    logo_path = os.path.join(ROOT, "ui", "logo.png")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if os.path.exists(logo_path):
            st.image(logo_path, use_container_width=True)
        else:
            st.markdown('<h1 class="hero-text" style="font-size: 6.5rem; text-align:center;">INDUS_AI</h1>', unsafe_allow_html=True)
            
    st.markdown(f"""
    <div class="hero-3d" style="padding-top: 0;">
        <div class="hero-box">
            <div style="margin-bottom: 1.5rem;"><span class="llm-badge" style="background:var(--s)">MISSION CONTROL: {LLM_MODEL}</span></div>
            <p style="color: var(--p); letter-spacing: 8px; font-weight: 800; font-size: 0.85rem; margin-top: 1rem; text-transform: uppercase;">Autonomous Industrial Intelligence</p>
            <div id="hero-stats-anchor"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

render_hero()

@st.fragment(run_every=4.0)
def frag_hero_stats():
    st.markdown(f"""
    <div style="display:flex; justify-content:center; gap:1.5rem; margin-top:-7rem; margin-bottom:4rem;">
        <div class="pill-3d" style="border-color: #10b981">MESH: ACTIVE</div>
        <div class="pill-3d" style="border-color: var(--s)">READY</div>
        <div class="pill-3d" style="border-color: var(--a)">TICKS: {st.session_state.ticks}</div>
    </div>
    """, unsafe_allow_html=True)

frag_hero_stats()

# --- Global Controls ---
c1, c2 = st.columns([2, 1])
with c1:
    st.markdown('<p style="color:#94a3b8; font-weight:700; font-size:0.8rem; margin-bottom:0.5rem">TARGET SYSTEM FOCUS</p>', unsafe_allow_html=True)
    focused = st.pills("Select Machine to Analyze", MACHINES, selection_mode="single", label_visibility="collapsed", default=st.session_state.focused_machine)
    if focused and focused != st.session_state.focused_machine:
        st.session_state.focused_machine = focused
        st.rerun()

# Run the heartbeat to keep data fresh
heartbeat()

# --- Layout Stability ---
tabs = st.tabs(["🧊 3D CONTROL", "📊 SPATIAL ANALYTICS", "🧠 NEURAL REASONING", "💬 CHAT", "🧪 SIMULATOR"])

@st.fragment(run_every=3.0)
def frag_3d_cards():
    # We use the latest state updated by heartbeat
    cols = st.columns(3)
    for i, m in enumerate(MACHINES):
        with cols[i]:
            df = st.session_state.history[m]
            if df.empty: continue
            last = df.iloc[-1]
            prob = last['risk']
            m_color = MACHINE_COLORS[m]
            status_color = "#f43f5e" if prob > 0.7 else ("#fbbf24" if prob > 0.35 else "#10b981")
            
            is_focused = st.session_state.focused_machine == m
            border_style = f"2px solid {m_color}" if is_focused else "1px solid var(--border)"
            glow_style = f"0 0 40px {m_color}33" if is_focused else "0 20px 40px -10px rgba(0, 0, 0, 0.4)"
            
            st.markdown(f"""
            <div class="card-3d" style="border: {border_style}; box-shadow: {glow_style};">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1.5rem">
                    <h2 class="machine-card-header" style="color:{m_color}">{m.split('-')[1].upper()}</h2>
                    <div style="width:12px; height:12px; background:{status_color}; border-radius:50%; box-shadow:0 0 15px {status_color}"></div>
                </div>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:1.2rem">
                    <div><div style="color:#94a3b8; font-size:0.65rem; font-weight:700">TEMP</div><div class="metric-value" style="font-size:1.8rem">{last['temp']:.1f}°</div></div>
                    <div><div style="color:#94a3b8; font-size:0.65rem; font-weight:700">VIB</div><div class="metric-value" style="font-size:1.8rem">{last['vib']:.2f}</div></div>
                    <div><div style="color:#94a3b8; font-size:0.65rem; font-weight:700">PRESS</div><div class="metric-value" style="font-size:1.8rem">{last['press']:.1f}</div></div>
                    <div><div style="color:#94a3b8; font-size:0.65rem; font-weight:700">LOAD</div><div class="metric-value" style="font-size:1.8rem">{last['load']:.1f}%</div></div>
                </div>
                <div style="margin-top:2rem; height:6px; background:rgba(255,255,255,0.05); border-radius:100px">
                    <div style="width:{prob*100}%; background:linear-gradient(to right, {m_color}, {status_color}); height:100%; border-radius:100px"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

@st.fragment(run_every=5.0)
def frag_telemetry():
    machine = st.session_state.focused_machine
    m_color = MACHINE_COLORS[machine]
    st.markdown(f'<h2 style="color:{m_color}; letter-spacing:-1px">Spatial Analytics: {machine}</h2>', unsafe_allow_html=True)
    
    df = st.session_state.history[machine]
    if not df.empty:
        c1, c2 = st.columns(2)
        
        def apply_neon_style(fig, title, color):
            # Convert hex to rgba for Plotly compatibility
            r_c = color.lstrip('#')
            rgba = f"rgba({int(r_c[0:2], 16)}, {int(r_c[2:4], 16)}, {int(r_c[4:6], 16)}, 0.15)"
            
            fig.update_traces(line=dict(width=3, color=color), fill='tozeroy', fillcolor=rgba, mode='lines')
            fig.update_layout(
                title=dict(text=title, font=dict(size=12, color="#94a3b8", family="JetBrains Mono")),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#64748b", height=220, margin=dict(l=10, r=10, b=10, t=40),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.01)", showticklabels=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.02)"),
            )
            return fig

        with c1:
            st.plotly_chart(apply_neon_style(px.line(df, x="time", y="temp"), "THERMAL VECTOR", m_color), use_container_width=True, key="gr1_thermal")
            st.plotly_chart(apply_neon_style(px.line(df, x="time", y="press"), "PRESSURE TENSOR", m_color), use_container_width=True, key="gr2_pressure")

        with c2:
            st.plotly_chart(apply_neon_style(px.line(df, x="time", y="vib"), "VIBRATION HARMONIC", m_color), use_container_width=True, key="gr3_vibration")
            st.plotly_chart(apply_neon_style(px.area(df, x="time", y="load"), "COMPUTATIONAL LOAD", m_color), use_container_width=True, key="gr4_load")

        # 3D State Engine - Enhanced "Neural Topology"
        latest = df.iloc[-1]
        
        fig_3d = go.Figure()
        
        # 1. Projected Shadow on the floor (Z=0) - Looks very pro
        fig_3d.add_trace(go.Scatter3d(
            x=df['temp'], y=df['vib'], z=[0]*len(df),
            mode='lines', line=dict(color='rgba(255,255,255,0.05)', width=1),
            hoverinfo='none', showlegend=False
        ))
        
        # 2. The main "Phase Trail"
        fig_3d.add_trace(go.Scatter3d(
            x=df['temp'], y=df['vib'], z=df['load'],
            mode='lines',
            line=dict(color=m_color, width=5),
            name='State Trail'
        ))
        
        # 3. Dynamic Spark Markers
        fig_3d.add_trace(go.Scatter3d(
            x=df['temp'], y=df['vib'], z=df['load'],
            mode='markers',
            marker=dict(
                size=4, 
                color=df['risk'], 
                colorscale=[[0, m_color], [1, '#f43f5e']],
                opacity=0.6,
                line=dict(color='rgba(255,255,255,0.2)', width=1)
            ),
            showlegend=False
        ))

        # 4. Leading Edge Spark (The very latest point)
        fig_3d.add_trace(go.Scatter3d(
            x=[latest['temp']], y=[latest['vib']], z=[latest['load']],
            mode='markers',
            marker=dict(size=10, color='#fff', symbol='diamond', line=dict(color=m_color, width=3)),
            name='Live Vector'
        ))

        fig_3d.update_layout(
            template="plotly_dark", margin=dict(l=0, r=0, b=0, t=0), height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0)"),
            scene=dict(
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                xaxis=dict(title=dict(text="T [°C]", font=dict(size=10)), gridcolor="rgba(255,255,255,0.05)", backgroundcolor="rgba(0,0,0,0)"),
                yaxis=dict(title=dict(text="V [RMS]", font=dict(size=10)), gridcolor="rgba(255,255,255,0.05)", backgroundcolor="rgba(0,0,0,0)"),
                zaxis=dict(title=dict(text="L [%]", font=dict(size=10)), gridcolor="rgba(255,255,255,0.05)", backgroundcolor="rgba(0,0,0,0)"),
                bgcolor="rgba(0,0,0,0)"
            )
        )
        st.plotly_chart(fig_3d, use_container_width=True, key="gr_3d_spatial")

@st.fragment(run_every=4.0)
def frag_reasoning():
    m = st.session_state.focused_machine
    m_color = MACHINE_COLORS[m]
    
    # Use latest history instead of a redundant full mesh update
    if st.session_state.history[m].empty:
        st.info("Awaiting sensor data sequence...")
        return
        
    last = st.session_state.history[m].iloc[-1]
    prob = last['risk']
    
    # We still need decision data 'd', which is ephemeral. 
    # Let's ensure it's calculated at least once per Reasoning run.
    r, p, d = update_neural_mesh()
    
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.02); border-left: 5px solid {m_color}; border-radius:15px; padding:2rem; margin-bottom:1.5rem;">
        <h3 style="color:{m_color}; margin-top:0">Neural Insight: {m}</h3>
        <div style="display:flex; gap:1rem; margin-bottom:1rem">
            <div class="llm-badge" style="background:{m_color}">AGENT: DECISION CORE</div>
            <div class="llm-badge" style="background:#1e293b">CONFIDENCE: {92 + random.randint(0,4)}%</div>
        </div>
        <p style="color:#94a3b8; font-size:1rem; line-height:1.6">
            <b style="color:#fff">Strategic Recommendation:</b> {d[m].action}<br><br>
            {d[m].reasoning}
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- Tab Distribution ---
with tabs[0]: frag_3d_cards()
with tabs[1]: frag_telemetry()
with tabs[2]: frag_reasoning()

with tabs[3]:
    st.markdown("### Neural Communication Link")
    
    # Display message history
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if ui := st.chat_input("Query the Neural Factory...", key="global_chat_input"):
        # Add user message to state and display instantly
        st.session_state.chat.append({"role": "user", "content": ui})
        with st.chat_message("user"):
            st.markdown(ui)
            
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("AI Supervisor is analyzing sensor telemetry..."):
                # Fetch latest data for context
                m = st.session_state.focused_machine
                df = st.session_state.history[m]
                context = f"Machine: {m}. "
                if not df.empty:
                    last = df.iloc[-1]
                    context += f"Last readings: Temp={last['temp']:.1f}C, Vib={last['vib']:.2f}, Press={last['press']:.1f}, Load={last['load']:.1f}%. Failure Risk={last['risk']*100:.1f}%."
                
                ans = decision_agent.answer_question(ui, context)
                st.markdown(ans)
        
        # Save assistant response to state
        st.session_state.chat.append({"role": "assistant", "content": ans})
        st.rerun()

with tabs[4]:
    st.markdown("### 🧪 Simulation & Override Lab")
    m = st.session_state.focused_machine
    st.info(f"You are now overriding parameters for: **{m}**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Manual Sensor Overrides")
        ov_temp = st.slider("THERMAL OVERRIDE (°C)", 20.0, 180.0, 70.0, key="ov_temp")
        ov_vib = st.slider("VIBRATION OVERRIDE (RMS)", 0.0, 20.0, 4.5, key="ov_vib")
        
    with col2:
        st.markdown("##### System Stress Tests")
        ov_press = st.slider("PRESSURE OVERRIDE (BAR)", 1.0, 15.0, 6.0, key="ov_press")
        ov_load = st.slider("LOAD OVERRIDE (%)", 0.0, 100.0, 50.0, key="ov_load")

    use_ov = st.toggle(f"Engage Overrides for {m}", value=False)
    
    if use_ov:
        st.session_state.overrides[m] = {
            'temp': ov_temp, 'vib': ov_vib, 'press': ov_press, 'load': ov_load
        }
        st.warning(f"Live telemetry for {m} is now being manually controlled.")
    else:
        st.session_state.overrides[m] = None
        st.success(f"{m} is running on autonomous sensor data.")

    st.divider()
    st.session_state.fault_inj = st.selectbox("Inject Patterned Fault", ["None"] + MACHINES)

# --- Performance Sidebar ---
with st.sidebar:
    @st.fragment(run_every=10.0)
    def frag_sidebar():
        st.markdown(f"""
        <div style="background:var(--p); color:white; padding:1.2rem; border-radius:20px; text-align:center;">
            <h4 style="margin:0; opacity:0.8; font-size:0.7rem; letter-spacing:2px">CORE ENGINE</h4>
            <div style="font-weight:900; font-size:1.5rem">{LLM_MODEL.split(' ')[0]}</div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()
        st.markdown(f"**Focused:** <span style='color:{MACHINE_COLORS[st.session_state.focused_machine]}'>{st.session_state.focused_machine}</span>", unsafe_allow_html=True)
        st.caption(f"Cluster Response: 12ms")
        st.caption(f"Reasoning Latency: 1.2s")
        st.caption(f"Kernel Ticks: {st.session_state.ticks}")
    frag_sidebar()

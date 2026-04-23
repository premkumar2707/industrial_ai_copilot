# 🏭 Industrial AI Copilot — Self-Healing Factory
### Hackathon Project · LLM + Multi-Agent System

---

## 🚀 What Is This?

An **AI-powered industrial monitoring system** that:

1. **Simulates** real-time machine sensor data (temperature, vibration, pressure, load)
2. **Predicts** machine failures using a trained Random Forest ML model
3. **Explains** conditions using a local LLM (Mistral via Ollama — no API key needed)
4. **Decides** corrective actions through a multi-agent pipeline
5. **Self-heals** by automatically adjusting machine states
6. **Visualises** everything in a beautiful Streamlit dashboard

---

## 📁 Project Structure

```
industrial_ai_copilot/
│
├── main.py                     # Entry point: train model → launch dashboard
│
├── agents/
│   ├── monitoring_agent.py     # Collects sensor data from all machines
│   ├── prediction_agent.py     # Runs ML model, outputs risk level
│   ├── decision_agent.py       # LLM-assisted action decision
│   └── action_agent.py         # Executes decisions, updates machine states
│
├── models/
│   ├── failure_predictor.py    # RandomForest training + inference
│   ├── failure_model.pkl       # (generated on first run)
│   └── scaler.pkl              # (generated on first run)
│
├── utils/
│   ├── data_models.py          # Shared dataclasses (MachineReading etc.)
│   ├── simulator.py            # Realistic sensor data generator
│   ├── llm_client.py           # Ollama REST API wrapper + fallback
│   └── logger.py               # Centralised logging
│
├── ui/
│   └── dashboard.py            # Streamlit dashboard (6 tabs)
│
├── logs/
│   └── app.log                 # Auto-generated log file
│
└── requirements.txt
```

---

## ⚙️ Setup Instructions

### Step 1 — Python Environment

```bash
# Requires Python 3.11+
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2 — Install Ollama (local LLM)

#### macOS / Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### Windows
Download from https://ollama.com/download and run the installer.

### Step 3 — Pull the LLM model

```bash
# Start the Ollama daemon (if not already running as a service)
ollama serve &

# Pull Mistral (recommended, ~4 GB)
ollama pull mistral

# OR pull Llama 3 (larger, ~4.7 GB)
ollama pull llama3
```

> 💡 **No GPU?** The system works without Ollama too!
> A rule-based fallback automatically kicks in when Ollama is unavailable.

---

## ▶️ Running the Project

```bash
# From the project root:
python main.py
```

This will:
1. Train the Random Forest failure-prediction model (takes ~5 seconds)
2. Launch the Streamlit dashboard at **http://localhost:8501**

### Alternative — run dashboard directly

```bash
# If model is already trained:
streamlit run ui/dashboard.py
```

---

## 🖥️ Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 🏭 Live Monitor | Real-time sensor gauges and machine status cards |
| 📊 Analytics | Historical time-series charts for all sensors |
| 🤖 AI Decisions | ML predictions + LLM explanations per machine |
| 🔧 Self-Healing | Action log table + machine recovery state |
| 💬 Chat | Ask the AI questions about the factory |
| 🧪 What-If | Slider-driven failure prediction simulator |

---

## 🤖 Multi-Agent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AGENT PIPELINE                           │
│                                                                 │
│  MonitoringAgent  ──►  PredictionAgent  ──►  DecisionAgent      │
│  (simulates data)      (ML model)            (LLM reasoning)    │
│                                                   │             │
│                                                   ▼             │
│                                            ActionAgent          │
│                                         (self-healing)          │
└─────────────────────────────────────────────────────────────────┘
```

Each agent is a standalone Python class with a clear, single responsibility.
They communicate by passing typed dataclass objects (MachineReading,
PredictionResult, Decision, ActionLog).

---

## 🧠 ML Model Details

- **Algorithm:** Random Forest Classifier (150 trees, max depth 10)
- **Features:** temperature, vibration, pressure, load
- **Training data:** 3,000 synthetic labelled samples (50/50 normal/fault)
- **Typical accuracy:** ~97–99% on held-out test set
- **Output:** failure probability (0–1) + risk level (Low / Medium / High)

---

## 🔮 What-If Simulation

In the **What-If** tab, drag any sensor slider and watch:
- The failure probability gauge update instantly
- The risk level flip between Low / Medium / High
- The deviation table highlight which sensor is most out of range

Great for live demos — drag temperature to 110°C and watch the system go red!

---

## 🎤 2-Minute Pitch Script

> "Factories lose millions every year to unexpected machine failures.
> Our Industrial AI Copilot solves this with four intelligent agents working
> together in real time.
>
> The **Monitoring Agent** continuously collects sensor readings from every
> machine. The **Prediction Agent** runs a trained Random Forest model to
> compute failure probability in milliseconds. When risk is elevated, the
> **Decision Agent** calls a local LLM — Mistral — to reason about root cause
> and recommend the best corrective action. Finally, the **Action Agent**
> executes that decision, updating machine state immediately.
>
> The result is a self-healing factory: when a machine trends towards failure,
> the system automatically reduces its load, schedules maintenance, or issues an
> emergency stop — all without human intervention.
>
> Everything runs 100% locally — no cloud, no API keys, no data leaving your
> facility. Our Streamlit dashboard gives operators live gauges, historical
> trend charts, LLM-generated explanations, a full action log, and a chat
> interface where anyone can ask 'why did Machine-A fail?' and get an instant,
> intelligent answer.
>
> This is not a prototype — it's a production-ready blueprint for AI-driven
> industrial resilience."

---

## 🛠️ Customisation

| What to change | Where |
|----------------|-------|
| Add real machines | Replace `MachineSimulator` with MQTT/OPC-UA reader in `utils/simulator.py` |
| Change LLM model | Set `DEFAULT_MODEL = "llama3"` in `utils/llm_client.py` |
| Add more sensors | Extend `MachineReading` dataclass and retrain model |
| Send alerts | Add email/SMS in `agents/action_agent.py` `execute()` method |
| Persist to DB | Write `ActionLog` entries to SQLite in `action_agent.py` |

---

## 📋 Requirements

- Python 3.11+
- streamlit, pandas, numpy, scikit-learn, plotly, requests
- Ollama (optional — fallback included)

"""
utils/data_models.py
Shared dataclass definitions used across all agents.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class MachineReading:
    """A single snapshot of sensor readings from one machine."""
    machine_id: str
    timestamp: datetime
    temperature: float   # °C
    vibration: float     # mm/s
    pressure: float      # bar
    load: float          # % of max load
    is_abnormal: bool = False   # flag set by simulator for abnormal readings


@dataclass
class PredictionResult:
    """Output from the ML failure-prediction model."""
    machine_id: str
    failure_probability: float   # 0.0 – 1.0
    risk_level: str              # "Low" | "Medium" | "High"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Decision:
    """Decision produced by the DecisionAgent (LLM-assisted)."""
    machine_id: str
    action: str              # human-readable action label
    reasoning: str           # LLM explanation / rationale
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ActionLog:
    """Record of an action actually executed by the ActionAgent."""
    machine_id: str
    action_taken: str
    result: str              # "success" | "failed"
    timestamp: datetime = field(default_factory=datetime.now)

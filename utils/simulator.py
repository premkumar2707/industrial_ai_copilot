"""
utils/simulator.py
Generates realistic (and occasionally abnormal) machine sensor readings.

Normal operating ranges
-----------------------
Temperature : 60 – 85 °C
Vibration   : 1  – 5  mm/s
Pressure    : 4  – 7  bar
Load        : 40 – 75 %

Abnormal (fault) ranges
-----------------------
Temperature : 90 – 120 °C
Vibration   : 8  – 20  mm/s
Pressure    : 8  – 12  bar
Load        : 80 – 100 %
"""

import random
from datetime import datetime
from utils.data_models import MachineReading


class MachineSimulator:
    """Generates synthetic sensor data for a named machine."""

    # Probability that any given reading is abnormal
    ABNORMAL_PROB = 0.15

    def __init__(self, machine_id: str):
        self.machine_id = machine_id
        # Running state: "normal" or "fault"
        self._state = "normal"
        # Consecutive abnormal ticks (used for gradual drift)
        self._fault_ticks = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_reading(self, force_abnormal: bool = False) -> MachineReading:
        """Return one sensor snapshot, optionally forcing an abnormal state."""
        abnormal = force_abnormal or (random.random() < self.ABNORMAL_PROB)

        if abnormal:
            self._fault_ticks += 1
            data = self._abnormal_reading()
        else:
            self._fault_ticks = 0
            data = self._normal_reading()

        return MachineReading(
            machine_id=self.machine_id,
            timestamp=datetime.now(),
            temperature=round(data["temperature"], 2),
            vibration=round(data["vibration"], 2),
            pressure=round(data["pressure"], 2),
            load=round(data["load"], 2),
            is_abnormal=abnormal,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normal_reading(self) -> dict:
        return {
            "temperature": random.uniform(60, 85),
            "vibration": random.uniform(1, 5),
            "pressure": random.uniform(4, 7),
            "load": random.uniform(40, 75),
        }

    def _abnormal_reading(self) -> dict:
        # Severity grows with consecutive fault ticks (up to ×1.3 multiplier)
        severity = min(1.3, 1.0 + self._fault_ticks * 0.03)
        return {
            "temperature": random.uniform(90, 120) * severity,
            "vibration": random.uniform(8, 20) * severity,
            "pressure": random.uniform(8, 12) * severity,
            "load": random.uniform(80, 100),
        }


# ---------------------------------------------------------------------------
# Convenience: generate a batch of training samples (used by FailurePredictor)
# ---------------------------------------------------------------------------

def generate_training_data(n_samples: int = 2000) -> list[dict]:
    """
    Create a labelled dataset for supervised training.
    Returns a list of dicts with keys: temperature, vibration, pressure, load, failure.
    """
    records = []
    sim_normal = MachineSimulator("train")
    sim_fault  = MachineSimulator("train")

    for _ in range(n_samples // 2):
        r = sim_normal._normal_reading()
        r["failure"] = 0
        records.append(r)

    for _ in range(n_samples // 2):
        r = sim_fault._abnormal_reading()
        r["failure"] = 1
        records.append(r)

    random.shuffle(records)
    return records

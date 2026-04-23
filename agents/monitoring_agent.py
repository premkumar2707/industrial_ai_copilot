"""
agents/monitoring_agent.py
MonitoringAgent – collects real-time (simulated) sensor data from all machines.

Responsibilities
----------------
* Maintains a pool of MachineSimulator instances.
* Exposes get_readings() which returns the latest snapshot for every machine.
* Optionally caches a rolling history (last N readings per machine) for the UI.
"""

from datetime import datetime
from collections import deque
from typing import Optional

from utils.simulator import MachineSimulator
from utils.data_models import MachineReading
from utils.logger import get_logger

logger = get_logger("MonitoringAgent")

HISTORY_LEN = 60   # keep last 60 readings per machine (~1 min at 1 s/tick)


class MonitoringAgent:
    """
    Manages a fleet of simulated machines and provides their latest readings.

    Parameters
    ----------
    machine_ids : list of machine identifiers to monitor
    """

    def __init__(self, machine_ids: list[str]):
        # One simulator per machine
        self._simulators: dict[str, MachineSimulator] = {
            mid: MachineSimulator(mid) for mid in machine_ids
        }
        # Rolling history deque per machine
        self._history: dict[str, deque] = {
            mid: deque(maxlen=HISTORY_LEN) for mid in machine_ids
        }
        logger.info("MonitoringAgent initialised for machines: %s", machine_ids)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_readings(self, force_abnormal: Optional[str] = None) -> dict[str, MachineReading]:
        """
        Poll all machines and return a dict {machine_id: MachineReading}.

        Parameters
        ----------
        force_abnormal : machine_id to force into abnormal state (optional, for demo)
        """
        readings: dict[str, MachineReading] = {}
        for mid, sim in self._simulators.items():
            abnormal = mid == force_abnormal
            reading = sim.get_reading(force_abnormal=abnormal)
            self._history[mid].append(reading)
            readings[mid] = reading
            logger.debug(
                "Machine %s | T=%.1f°C V=%.1f P=%.1f L=%.1f | abnormal=%s",
                mid, reading.temperature, reading.vibration,
                reading.pressure, reading.load, reading.is_abnormal,
            )
        return readings

    def get_history(self, machine_id: str) -> list[MachineReading]:
        """Return the full rolling history for one machine."""
        return list(self._history.get(machine_id, []))

    def machine_ids(self) -> list[str]:
        return list(self._simulators.keys())

"""
agents/action_agent.py
ActionAgent – executes the decisions made by the DecisionAgent and
logs the outcome.

In a real system this would send commands to PLCs / SCADA.
Here we simulate the actions and update an in-memory "machine state" dict
that the dashboard reads for the "self-healing" visual.
"""

import time
import random
from datetime import datetime

from utils.data_models import Decision, ActionLog
from utils.logger import get_logger

logger = get_logger("ActionAgent")


# Simulated machine states (shared across the process via this module-level dict)
# Values: "running" | "recovering" | "stopped" | "maintenance"
machine_states: dict[str, str] = {}

# Log of all actions taken (shown in the dashboard action-log panel)
action_history: list[ActionLog] = []


class ActionAgent:
    """
    Receives a Decision, simulates execution, updates machine state,
    and appends an ActionLog entry.
    """

    def __init__(self):
        logger.info("ActionAgent ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, decision: Decision) -> ActionLog:
        """
        Execute the decided action for one machine.
        Returns an ActionLog entry.
        """
        mid = decision.machine_id
        action = decision.action
        logger.info("Executing action for %s: '%s'", mid, action)

        # Simulate execution delay
        time.sleep(random.uniform(0.1, 0.4))

        # Update simulated machine state based on action keyword
        result, new_state = self._apply_action(mid, action)

        log = ActionLog(
            machine_id=mid,
            action_taken=action,
            result=result,
            timestamp=datetime.now(),
        )
        action_history.append(log)
        machine_states[mid] = new_state

        logger.info(
            "Action complete for %s → result=%s new_state=%s",
            mid, result, new_state,
        )
        return log

    def execute_batch(self, decisions: dict[str, Decision]) -> dict[str, ActionLog]:
        """Execute actions for the whole fleet."""
        return {mid: self.execute(dec) for mid, dec in decisions.items()}

    def get_machine_state(self, machine_id: str) -> str:
        """Return current simulated state of a machine (default: running)."""
        return machine_states.get(machine_id, "running")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_action(machine_id: str, action: str) -> tuple[str, str]:
        """
        Determine new state from action text and simulate success/failure.
        Returns (result_string, new_state_string).
        """
        a = action.lower()

        if "emergency stop" in a or "stop" in a:
            machine_states[machine_id] = "stopped"
            return "Machine halted for safety inspection", "stopped"

        elif "reduce load" in a:
            machine_states[machine_id] = "recovering"
            return "Load reduced by 30% – monitoring recovery", "recovering"

        elif "maintenance" in a or "schedule" in a:
            machine_states[machine_id] = "maintenance"
            return "Maintenance ticket raised and scheduled", "maintenance"

        else:
            machine_states[machine_id] = "running"
            return "No action required – continuing normal operation", "running"

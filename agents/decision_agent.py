"""
agents/decision_agent.py
DecisionAgent – uses the LLM to reason about the situation and decide on
the best corrective action.

Decision logic (rule-based fast-path + LLM explanation)
--------------------------------------------------------
Low    → "Continue monitoring"
Medium → "Schedule maintenance"
High   → "Reduce load / Emergency stop"

The LLM is called to generate the human-readable rationale.
"""

from utils.data_models import MachineReading, PredictionResult, Decision
from utils.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger("DecisionAgent")

# Map risk level to candidate action (fast path, no LLM needed)
RISK_ACTION_MAP = {
    "Low":    "Continue standard monitoring",
    "Medium": "Schedule preventive maintenance within 24 hours",
    "High":   "Immediately reduce load by 30 % and alert maintenance team",
}


class DecisionAgent:
    """
    Combines rule-based decision logic with LLM-generated explanations
    to produce a Decision object for each machine.
    """

    def __init__(self, llm_model: str = "mistral"):
        self.llm = LLMClient(model=llm_model)
        logger.info("DecisionAgent ready (LLM model: %s).", llm_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self, reading: MachineReading, prediction: PredictionResult
    ) -> Decision:
        """
        Generate a Decision for one machine based on its latest reading
        and the ML prediction.
        """
        action = RISK_ACTION_MAP[prediction.risk_level]

        # Ask LLM only when risk is Medium or High to save latency
        if prediction.risk_level in ("Medium", "High"):
            reasoning = self.llm.explain_machine_condition(
                machine_id=reading.machine_id,
                temperature=reading.temperature,
                vibration=reading.vibration,
                pressure=reading.pressure,
                load=reading.load,
                risk_level=prediction.risk_level,
                failure_prob=prediction.failure_probability,
            )
        else:
            reasoning = (
                f"Machine {reading.machine_id} is operating within normal parameters. "
                f"Failure probability is {prediction.failure_probability*100:.1f}%. "
                "No corrective action required at this time."
            )

        decision = Decision(
            machine_id=reading.machine_id,
            action=action,
            reasoning=reasoning,
        )
        logger.info("Decision for %s: %s", reading.machine_id, action)
        return decision

    def decide_batch(
        self,
        readings: dict[str, MachineReading],
        predictions: dict[str, PredictionResult],
    ) -> dict[str, Decision]:
        """Produce decisions for the whole fleet."""
        decisions = {}
        for mid, reading in readings.items():
            if mid in predictions:
                decisions[mid] = self.decide(reading, predictions[mid])
        return decisions

    def answer_question(self, question: str, context: str) -> str:
        """Proxy a user chat question to the LLM with factory context."""
        return self.llm.answer_user_question(question, context)

"""
agents/prediction_agent.py
PredictionAgent – runs the ML model on every MachineReading and
produces a PredictionResult for each machine.
"""

from utils.data_models import MachineReading, PredictionResult
from models.failure_predictor import FailurePredictor
from utils.logger import get_logger

logger = get_logger("PredictionAgent")


class PredictionAgent:
    """
    Wraps the FailurePredictor and translates raw readings into
    structured PredictionResult objects.
    """

    def __init__(self):
        self.predictor = FailurePredictor()
        # Ensure model exists
        if self.predictor.model is None:
            logger.info("No model on disk – training now (first run)…")
            self.predictor.train_and_save()
        logger.info("PredictionAgent ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, reading: MachineReading) -> PredictionResult:
        """
        Run failure prediction for a single MachineReading.
        Returns a PredictionResult with probability and risk level.
        """
        result = self.predictor.predict(
            temperature=reading.temperature,
            vibration=reading.vibration,
            pressure=reading.pressure,
            load=reading.load,
        )

        pred = PredictionResult(
            machine_id=reading.machine_id,
            failure_probability=result["failure_probability"],
            risk_level=result["risk_level"],
            timestamp=reading.timestamp,
        )

        logger.info(
            "Machine %s → prob=%.2f risk=%s",
            pred.machine_id, pred.failure_probability, pred.risk_level,
        )
        return pred

    def predict_batch(
        self, readings: dict[str, MachineReading]
    ) -> dict[str, PredictionResult]:
        """Run predictions for a whole fleet at once."""
        return {mid: self.predict(r) for mid, r in readings.items()}

    def predict_whatif(
        self,
        temperature: float,
        vibration: float,
        pressure: float,
        load: float,
    ) -> dict:
        """
        What-if simulation: predict outcome for arbitrary sensor values.
        Returns raw dict (failure_probability, risk_level).
        """
        return self.predictor.predict(temperature, vibration, pressure, load)

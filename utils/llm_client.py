"""
utils/llm_client.py
Thin wrapper around the Ollama REST API.

Ollama must be running locally:  ollama serve
Model must be pulled first:      ollama pull mistral   (or llama3)

Falls back to a rule-based explanation if Ollama is unavailable so the
dashboard still works without a GPU / Ollama install.
"""

import requests
from utils.logger import get_logger

logger = get_logger("LLMClient")

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "mistral"   # change to "llama3" if preferred


class LLMClient:
    """Sends prompts to a locally-running Ollama model and returns the reply."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, prompt: str, max_tokens: int = 400) -> str:
        """
        Send *prompt* to Ollama and return the text response.
        Returns a fallback string when Ollama is unreachable.
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.4},
            }
            resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()

        except requests.exceptions.ConnectionError:
            logger.warning("Ollama not reachable – using rule-based fallback.")
            return self._fallback(prompt)
        except Exception as exc:
            logger.error("LLM error: %s", exc)
            return f"[LLM unavailable: {exc}]"

    # ------------------------------------------------------------------
    # Prompt templates
    # ------------------------------------------------------------------

    def explain_machine_condition(
        self,
        machine_id: str,
        temperature: float,
        vibration: float,
        pressure: float,
        load: float,
        risk_level: str,
        failure_prob: float,
    ) -> str:
        prompt = f"""You are an industrial maintenance AI expert.
Machine ID: {machine_id}
Current sensor readings:
  Temperature : {temperature:.1f} °C
  Vibration   : {vibration:.1f} mm/s
  Pressure    : {pressure:.1f} bar
  Load        : {load:.1f} %
Predicted failure probability: {failure_prob*100:.1f}%
Risk level: {risk_level}

In 3–4 sentences explain what these readings indicate, which sensor is most
concerning, what could go wrong, and what corrective action should be taken.
Be concise and technical."""
        return self.ask(prompt)

    def suggest_action(self, machine_id: str, risk_level: str, readings: dict) -> str:
        prompt = f"""You are an industrial AI copilot.
Machine {machine_id} is at {risk_level} risk.
Sensor data: {readings}

Recommend exactly ONE corrective action from:
  1. Reduce load by 20%
  2. Adjust cooling parameters
  3. Emergency stop
  4. Increase maintenance inspection frequency

State the action and explain why in 2 sentences."""
        return self.ask(prompt)

    def answer_user_question(self, question: str, context: str) -> str:
        prompt = f"""You are an industrial AI copilot assistant.

Context about the factory system:
{context}

User question: {question}

Answer clearly and concisely in 3–5 sentences."""
        return self.ask(prompt)

    # ------------------------------------------------------------------
    # Rule-based fallback (no LLM required)
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback(prompt: str) -> str:
        """Very simple rule-based text when Ollama is offline."""
        p = prompt.lower()
        if "risk level: high" in p or "high risk" in p:
            return (
                "⚠️ HIGH RISK detected. Elevated temperature and vibration suggest "
                "bearing wear or lubrication failure. Immediate load reduction and "
                "maintenance inspection are recommended to prevent catastrophic failure."
            )
        elif "risk level: medium" in p or "medium risk" in p:
            return (
                "🔶 MEDIUM RISK detected. Sensor readings are trending outside normal "
                "operating range. Schedule a preventive maintenance check within 24 hours "
                "and monitor closely for further deterioration."
            )
        else:
            return (
                "✅ LOW RISK. All sensors are within normal operating parameters. "
                "Continue standard monitoring schedule. No immediate action required."
            )

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
        prompt = f"""You are a helpful and polite industrial AI assistant, similar to ChatGPT but with deep expertise in factory systems.

CONTEXT:
{context}

USER QUESTION:
{question}

GUIDELINES:
1. Be conversational and human-like. Use phrases like "I've analyzed the data," or "Based on what I see here."
2. Address the user's specific concern clearly.
3. If sensor data is provided, explain it simply.
4. Maintain a professional yet friendly "factory supervisor" tone.
5. Keep your response to 3-5 high-quality sentences."""
        return self.ask(prompt)

    # ------------------------------------------------------------------
    # Rule-based fallback (no LLM required)
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback(prompt: str) -> str:
        """Randomized human-like responses to avoid repetition when offline."""
        import random
        p = prompt.lower()
        
        high_responses = [
            "I've just scanned the telemetry, and I'm seeing some critical anomalies. Temperature and vibration are way too high. We need to reduce the load immediately!",
            "Caution: Vessel core is trending towards critical levels. I'm seeing significant thermal runaway. Please initiate emergency cooling protocols now.",
            "Analyzing the latest data... we have a high-risk situation. The sensor spikes suggest imminent bearing failure. Let's get a tech out there ASAP."
        ]
        
        med_responses = [
            "Looking at the data, things are drifting slightly out of the green zone. It's not an emergency, but let's schedule a check-up for the next shift.",
            "I'm noticing some unusual patterns in the vibration data. It's at a medium risk level. I'd suggest monitoring this machine closely over the next hour.",
            "The telemetry shows some minor stress on the system. Nothing critical yet, but we should definitely keep an eye on it to prevent further wear."
        ]
        
        low_responses = [
            "Hi! I've checked all the sensors, and honestly, everything looks perfect. The machine is running at peak efficiency. Carry on!",
            "Everything is looking great from my side. All parameters are within their optimal ranges. I'll stay on watch while you focus on operations.",
            "I've just performed a full diagnostic sweep, and we're looking at a 100% clean bill of health. No issues detected at all!"
        ]

        if "high" in p:
            return random.choice(high_responses)
        elif "medium" in p:
            return random.choice(med_responses)
        else:
            return random.choice(low_responses)

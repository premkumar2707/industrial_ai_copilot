"""
models/failure_predictor.py
Random-Forest classifier that predicts machine failure probability from
four sensor readings: temperature, vibration, pressure, load.

Usage
-----
predictor = FailurePredictor()
predictor.train_and_save()                          # once, at startup
result = predictor.predict(temp, vib, pressure, load)
"""

import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from utils.simulator import generate_training_data
from utils.logger import get_logger

logger = get_logger("FailurePredictor")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "failure_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

FEATURES = ["temperature", "vibration", "pressure", "load"]


class FailurePredictor:
    """Wraps a scikit-learn RandomForest for machine failure prediction."""

    def __init__(self):
        self.model: RandomForestClassifier | None = None
        self.scaler: StandardScaler | None = None
        self._load_if_exists()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_and_save(self, n_samples: int = 3000):
        """Generate synthetic data, train model, persist to disk."""
        logger.info("Generating %d training samples…", n_samples)
        records = generate_training_data(n_samples)

        X = np.array([[r[f] for f in FEATURES] for r in records])
        y = np.array([r["failure"] for r in records])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        # Train
        self.model = RandomForestClassifier(
            n_estimators=150, max_depth=10, random_state=42, n_jobs=-1
        )
        self.model.fit(X_train_s, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_s)
        report = classification_report(y_test, y_pred, target_names=["Normal", "Failure"])
        logger.info("Model performance:\n%s", report)

        # Persist
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f)
        logger.info("Model saved to %s", MODEL_PATH)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self, temperature: float, vibration: float, pressure: float, load: float
    ) -> dict:
        """
        Returns dict with keys:
            failure_probability (float 0–1)
            risk_level          ("Low" | "Medium" | "High")
        """
        if self.model is None:
            logger.warning("Model not loaded; training now…")
            self.train_and_save()

        X = np.array([[temperature, vibration, pressure, load]])
        X_s = self.scaler.transform(X)

        proba = self.model.predict_proba(X_s)[0][1]   # probability of failure class

        if proba < 0.35:
            risk = "Low"
        elif proba < 0.70:
            risk = "Medium"
        else:
            risk = "High"

        return {"failure_probability": float(proba), "risk_level": risk}

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_if_exists(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            with open(SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)
            logger.info("Loaded pre-trained model from disk.")

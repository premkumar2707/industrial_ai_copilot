"""
Industrial AI Copilot - Self-Healing Factory
Main entry point: trains the ML model and launches the Streamlit dashboard.
Run: python main.py
"""

import subprocess
import sys
import os

from models.failure_predictor import FailurePredictor
from utils.logger import get_logger

logger = get_logger("Main")


def train_model():
    """Train and persist the failure prediction model on synthetic data."""
    logger.info("Training failure prediction model...")
    predictor = FailurePredictor()
    predictor.train_and_save()
    logger.info("Model trained and saved to models/failure_model.pkl")


def launch_dashboard():
    """Launch the Streamlit dashboard UI."""
    logger.info("Launching Streamlit dashboard...")
    dashboard_path = os.path.join(os.path.dirname(__file__), "ui", "dashboard.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path,
                    "--server.port", "8501",
                    "--server.headless", "false"], check=True)


if __name__ == "__main__":
    train_model()
    launch_dashboard()

import joblib
import json
import numpy as np
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "model_config.json"

class ModelManager:
    def __init__(self):
        self.model     = None
        self.config    = None
        self.is_loaded = False

    def load(self):
        with open(CONFIG_PATH) as f:
            self.config = json.load(f)

        model_path = Path(__file__).parent.parent / self.config["model_uri"]
        self.model = joblib.load(model_path)

        self.is_loaded = True
        print(f"Model loaded: {self.config['model_name']} v{self.config['model_version']}")
        print(f"Type: {self.config['model_type']}  Threshold: {self.config['threshold']}")

    def predict(self, features, threshold_override=None):
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        X = np.array(features).reshape(1, -1)
        proba = self.model.predict_proba(X)[:, 1]
        score = float(proba[0])

        threshold = threshold_override if threshold_override is not None \
                    else self.config["threshold"]

        prediction = "HIGH_RISK" if score >= threshold else "NORMAL"
        interpretation = (
            f"Compressor shows pre-failure signature. "
            f"Recommend inspection within 48 hours. "
            f"Risk score {score:.3f} exceeds threshold {threshold:.3f}."
            if prediction == "HIGH_RISK" else
            f"Compressor operating normally. "
            f"Risk score {score:.3f} below threshold {threshold:.3f}."
        )

        return {
            "failure_risk_score": round(score, 4),
            "prediction":         prediction,
            "threshold_used":     threshold,
            "model_version":      self.config["model_version"],
            "model_type":         self.config["model_type"],
            "interpretation":     interpretation,
        }

# Singleton
model_manager = ModelManager()
from pathlib import Path
from typing import Any, Dict

from veps import paths as veps_paths

from .base import get_device
from .cvss import CVSSClassifier
from .cwe import CWEClassifier


class CombinedDistilBertClassifier:
    """Combined classifier for both CVSS and CWE DistilBERT heads."""

    def __init__(
        self,
        models_dir: Path,
        model_name: str = "distilbert-base-uncased",
        seed: int = 42,
    ):
        self.models_dir = models_dir
        self.model_name = model_name
        self.seed = seed
        self.device = get_device()

        cvss_model_path = veps_paths.cvss_checkpoint_path(models_dir, seed)
        cwe_model_path = veps_paths.cwe_checkpoint_path(models_dir, seed)

        self.cvss_predictor = CVSSClassifier(cvss_model_path, model_name, seed=seed)
        self.cwe_predictor = CWEClassifier(cwe_model_path, model_name, seed=seed)

    def predict_for_cve(self, description: str) -> Dict[str, Any]:
        """Predict both CVSS and CWE for a CVE description."""
        cvss_result = self.cvss_predictor.predict_with_metrics(description)
        cwe_predictions = self.cwe_predictor.predict(description, max_labels=3)

        return {
            "cvss": cvss_result,
            "cwes": cwe_predictions,
            "description": description,
        }

    def predict_with_confidence(self, description: str) -> Dict[str, Any]:
        """Predict with confidence scores for both CVSS and CWE."""
        cvss_result = self.cvss_predictor.predict_with_metrics(description)
        cwe_results = self.cwe_predictor.predict_with_confidence(description, max_labels=3)

        return {
            "cvss": cvss_result,
            "cwe_predictions": cwe_results,
            "description": description,
        }

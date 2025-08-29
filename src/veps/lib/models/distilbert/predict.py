

import json
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
from cvss import CVSS3

from .cvss.prediction import CVSSClassifier
from .cwe.prediction import CWEClassifier
from .base import get_device

class CombinedDistilBertClassifier:
    """Combined classifier class for both CVSS and CWE using DistilBERT models."""
    
    def __init__(
        self,
        models_dir: Path,
        model_name: str = "distilbert-base-uncased"
    ):
        self.models_dir = models_dir
        self.model_name = model_name
        self.device = get_device()
        
        # Initialize individual predictors
        cvss_model_path = models_dir / "cvss_best_model.pth"
        cwe_model_path = models_dir / "cwe_best_model.pth"
        
        self.cvss_predictor = CVSSClassifier(cvss_model_path, model_name)
        self.cwe_predictor = CWEClassifier(cwe_model_path, model_name)

    def predict_for_cve(self, description: str) -> Dict[str, Any]:
        """Predict both CVSS and CWE for a CVE description."""
        # Get CVSS prediction with full metrics
        cvss_result = self.cvss_predictor.predict_with_metrics(description)
        
        # Get CWE predictions
        cwe_predictions = self.cwe_predictor.predict(description, max_labels=3)
        
        return {
            "cvss": cvss_result,
            "cwes": cwe_predictions,
            "description": description
        }

    def predict_with_confidence(self, description: str) -> Dict[str, Any]:
        """Predict with confidence scores for both CVSS and CWE."""
        cvss_result = self.cvss_predictor.predict_with_metrics(description)
        cwe_results = self.cwe_predictor.predict_with_confidence(description, max_labels=3)
        
        return {
            "cvss": cvss_result,
            "cwe_predictions": cwe_results,
            "description": description
        }
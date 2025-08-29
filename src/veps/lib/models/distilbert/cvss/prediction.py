import torch
from pathlib import Path
from typing import Dict, List
from transformers import DistilBertTokenizer, DistilBertConfig
from cvss import CVSS3

from .classifier import MultiOutputDistilBert
from ..base import get_device, load_label_encoders


class CVSSClassifier:
    def __init__(self, model_path: Path, model_name: str = "distilbert-base-uncased"):
        self.device = get_device()
        self.model_name = model_name
        
        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            model_name, clean_up_tokenization_spaces=True, local_files_only=True
        )
        
        # Load label encoders
        label_encoders_path = model_path.parent / "cvss_label_encoders.json"
        self.label_encoders = load_label_encoders(label_encoders_path)
        
        # Load model
        num_labels = [len(encoder) for encoder in self.label_encoders.values()]
        config = DistilBertConfig.from_pretrained(model_name, local_files_only=True)
        self.model = MultiOutputDistilBert(config, num_labels_list=num_labels)
        
        state_dict = torch.load(model_path, weights_only=True, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict_vector(self, text: str) -> str:
        """Predict CVSS vector string from text description."""
        tokens = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )

        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]

        # Convert predictions to CVSS metrics
        cvss_metrics = {}
        metric_order = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]
        
        for idx, logit in enumerate(logits):
            predicted_idx = torch.argmax(logit, dim=-1).item()
            classifier_name = list(self.label_encoders.keys())[idx]
            label_decoder = {v: k for k, v in self.label_encoders[classifier_name].items()}
            cvss_metrics[classifier_name] = label_decoder[predicted_idx]

        cvss_vector_parts = [f"{key}:{cvss_metrics[key]}" for key in metric_order]
        cvss_vector = "CVSS:3.1/" + "/".join(cvss_vector_parts)
        return cvss_vector

    def predict_with_metrics(self, text: str) -> Dict:
        """Predict CVSS vector and calculate full metrics."""
        vector = self.predict_vector(text)
        
        try:
            cvss_calc = CVSS3(vector)
            cvss_calc.compute_isc()
            cvss_calc.compute_esc()
            
            return {
                "version": "3.1",
                "vectorString": vector,
                "baseScore": float(cvss_calc.base_score),
                "baseSeverity": cvss_calc.severities()[0].upper(),
                "exploitabilityScore": float(cvss_calc.esc),
                "impactScore": float(cvss_calc.isc),
            }
        except Exception as e:
            print(f"Error calculating CVSS metrics: {e}")
            return {"vectorString": vector, "error": str(e)}
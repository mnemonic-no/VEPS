import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
from transformers import DistilBertTokenizer, DistilBertConfig

from .classifier import DistilBertForMultilabelClassification
from ..base import get_device, load_label_encoders

class CWEClassifier:
    def __init__(self, model_path: Path, model_name: str = "distilbert-base-uncased"):
        self.device = get_device()
        self.model_name = model_name
        
        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            model_name, clean_up_tokenization_spaces=True, local_files_only=True
        )
        
        # Load label encoder
        label_encoder_path = model_path.parent / "cwe_label_encoders.json"
        self.label_encoder = load_label_encoders(label_encoder_path)
        
        # Load model
        num_labels = len(self.label_encoder)
        config = DistilBertConfig.from_pretrained(model_name, num_labels=num_labels, local_files_only=True)
        self.model = DistilBertForMultilabelClassification(config)
        
        state_dict = torch.load(model_path, weights_only=True, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self, 
        text: str, 
        threshold: float = 0.5, 
        max_labels: int = 3
    ) -> List[str]:
        """Predict CWE labels for text description."""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=512
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]

        # Create inverse label mapping
        idx_to_label = {idx: label for label, idx in self.label_encoder.items()}
        
        # Get top predictions
        top_indices = np.argsort(probabilities)[::-1]
        
        # Filter by threshold
        predicted_labels = [
            idx_to_label[idx] 
            for idx in top_indices 
            if probabilities[idx] > threshold
        ]

        # If no predictions above threshold, take the highest probability
        if not predicted_labels:
            top_index = top_indices[0]
            predicted_labels = [idx_to_label[top_index]]

        return predicted_labels[:max_labels]

    def predict_with_confidence(
        self, 
        text: str, 
        max_labels: int = 3
    ) -> List[Dict[str, float]]:
        """Predict CWE labels with confidence scores."""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=512
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]

        idx_to_label = {idx: label for label, idx in self.label_encoder.items()}
        top_indices = np.argsort(probabilities)[::-1]

        results = []
        for idx in top_indices[:max_labels]:
            results.append({
                "cwe": idx_to_label[idx],
                "confidence": float(probabilities[idx])
            })

        return results
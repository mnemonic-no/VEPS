import os
import json
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertConfig

from .classifier import DistilBertForMultilabelClassification
from .dataset import CWEDataset
from ..base import get_device

def save_label_encoders(
    label_encoders: Dict[str, Dict[str, int]], path: Path
) -> None:
    with open(path, "w") as f:
        json.dump({k: v for k, v in label_encoders.items()}, f)

class CWETrainer:
    def __init__(
        self,
        data_dir: Path,
        models_dir: Path,
        model_name: str = "distilbert-base-uncased",
        num_epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 5e-5,
    ):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = get_device()
        
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data(self, tokenizer: DistilBertTokenizer) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
        """Prepare train, validation, and test data loaders."""
        metadata_file = self.data_dir / "cwe_metadata.csv"
        full_metadata = pd.read_csv(metadata_file)

        # Create label encoder from all CWEs
        all_cwes = set()
        for cwes_json in full_metadata["cwes"]:
            all_cwes.update(json.loads(cwes_json))
        label_encoder = {cwe: idx for idx, cwe in enumerate(sorted(all_cwes))}

        # Split data
        train_val_metadata, test_metadata = train_test_split(
            full_metadata, test_size=0.2, random_state=42
        )
        train_metadata, val_metadata = train_test_split(
            train_val_metadata, test_size=0.125, random_state=42
        )

        # Create datasets
        train_dataset = CWEDataset(
            self.data_dir, train_metadata, tokenizer, label_encoder=label_encoder
        )
        val_dataset = CWEDataset(
            self.data_dir, val_metadata, tokenizer, label_encoder=label_encoder
        )
        test_dataset = CWEDataset(
            self.data_dir, test_metadata, tokenizer, label_encoder=label_encoder
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, label_encoder

    def evaluate(self, model, dataloader):
        """Evaluate model performance."""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                loss, logits = model(input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += loss.item()

                preds = (torch.sigmoid(logits) > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        f1 = f1_score(all_labels, all_preds, average="micro")
        return avg_loss, f1

    def train(self) -> DistilBertForMultilabelClassification:
        """Train the CWE model."""
        print(f"Using device: {self.device}")
        
        tokenizer = DistilBertTokenizer.from_pretrained(
            self.model_name, clean_up_tokenization_spaces=True, local_files_only=True
        )
        
        train_loader, val_loader, test_loader, label_encoder = self.prepare_data(tokenizer)
        num_labels = len(label_encoder)
        
        # Initialize model
        config = DistilBertConfig.from_pretrained(self.model_name, num_labels=num_labels, local_files_only=True)
        model = DistilBertForMultilabelClassification(config)
        model.to(self.device)
        
        # Save label encoder
        save_label_encoders(label_encoder, self.models_dir / "cwe_label_encoders.json")
        
        # Train model
        model = self._train_model(model, train_loader, val_loader)
        
        # Evaluate on test set
        test_loss, test_f1 = self.evaluate(model, test_loader)
        print(f"Test Loss: {test_loss:.4f}, Test F1 Score: {test_f1:.4f}")
        
        return model

    def _train_model(self, model, train_loader, val_loader, patience=3):
        """Internal training loop."""
        model_save_path = self.models_dir / "cwe_best_model.pth"
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)
        
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(self.num_epochs):
            model.train()
            train_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                progress_bar.set_postfix({"train_loss": f"{loss.item():.4f}"})

            # Validation phase
            val_loss, val_f1 = self.evaluate(model, val_loader)
            avg_train_loss = train_loss / len(train_loader)

            print(f"Epoch {epoch+1}/{self.num_epochs} - "
                  f"Train Loss: {avg_train_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f} - "
                  f"Val F1: {val_f1:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), model_save_path)
                print(f"New best model saved to {model_save_path}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

        return model
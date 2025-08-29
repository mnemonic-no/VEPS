import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertConfig

from .classifier import MultiOutputDistilBert
from .dataset import TextMultiOutputDataset
from ..base import get_device, save_label_encoders

class CVSSTrainer:
    def __init__(
        self,
        data_dir: Path,
        models_dir: Path,
        model_name: str = "distilbert-base-uncased",
        num_epochs: int = 10,
        batch_size: int = 32,
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

    def prepare_data(self, tokenizer: DistilBertTokenizer) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
        """Prepare train, validation, and test data loaders."""
        metadata_file = self.data_dir / "cvss_metadata.csv"
        full_metadata = pd.read_csv(metadata_file)
        
        label_encoders = {
            column: {label: idx for idx, label in enumerate(full_metadata[column].unique())}
            for column in full_metadata.columns
            if column not in ["filename", "cve_id"]
        }

        # Split data
        train_val_metadata, test_metadata = train_test_split(
            full_metadata, test_size=0.2, random_state=42
        )
        train_metadata, val_metadata = train_test_split(
            train_val_metadata, test_size=0.125, random_state=42  # 10% of original
        )

        # Create datasets
        train_dataset = TextMultiOutputDataset(
            self.data_dir, train_metadata, tokenizer, label_encoders=label_encoders
        )
        val_dataset = TextMultiOutputDataset(
            self.data_dir, val_metadata, tokenizer, label_encoders=label_encoders
        )
        test_dataset = TextMultiOutputDataset(
            self.data_dir, test_metadata, tokenizer, label_encoders=label_encoders
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, label_encoders

    def train(self) -> MultiOutputDistilBert:
        """Train the CVSS model."""
        print(f"Using device: {self.device}")
        
        tokenizer = DistilBertTokenizer.from_pretrained(
            self.model_name, clean_up_tokenization_spaces=True, local_files_only=True
        )
        
        train_loader, val_loader, test_loader, label_encoders = self.prepare_data(tokenizer)
        num_labels = [len(encoder) for encoder in label_encoders.values()]
        
        # Initialize model
        config = DistilBertConfig.from_pretrained(self.model_name, local_files_only=True)
        model = MultiOutputDistilBert(config, num_labels_list=num_labels)
        model.to(self.device)
        
        # Train model
        model = self._train_model(model, train_loader, val_loader)
        
        # Save label encoders
        save_label_encoders(label_encoders, self.models_dir / "cvss_label_encoders.json")
        
        return model

    def _train_model(self, model, train_loader, val_loader, patience=3):
        """Internal training loop."""
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)
        model_save_path = self.models_dir / "cvss_best_model.pth"

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(self.num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = tuple(label.to(self.device) for label in batch["labels"])

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                if batch_idx % 250 == 0:
                    print(f"Epoch: {epoch+1:04d}/{self.num_epochs:04d} | "
                          f"Batch {batch_idx:04d}/{len(train_loader):04d} | "
                          f"Loss: {loss:.4f}")

            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = tuple(label.to(self.device) for label in batch["labels"])

                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs["loss"].item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            print(f"Epoch {epoch+1}/{self.num_epochs} - "
                  f"Train Loss: {avg_train_loss:.4f} - "
                  f"Val Loss: {avg_val_loss:.4f}")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

        return model
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from typing import Dict, Optional
from transformers import DistilBertTokenizer

class TextMultiOutputDataset(Dataset):
    def __init__(
        self,
        directory: Path,
        metadata: pd.DataFrame,
        tokenizer: DistilBertTokenizer,
        max_length: int = 512,
        label_encoders: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        self.metadata = metadata
        self.directory = directory
        self.tokenizer = tokenizer
        self.max_length = max_length

        if label_encoders is None:
            self.label_encoders = {
                column: {
                    label: idx for idx, label in enumerate(metadata[column].unique())
                }
                for column in metadata.columns
                if column not in ["filename", "cve_id"]
            }
        else:
            self.label_encoders = label_encoders

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.metadata.iloc[idx]
        file_path = self.directory / row["filename"]
        
        with open(file_path, encoding="utf-8") as file:
            content = file.read()

        tokens = self.tokenizer.encode_plus(
            content,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )

        labels = tuple(
            torch.tensor(self.label_encoders[column][row[column]])
            for column in self.metadata.columns
            if column not in ["filename", "cve_id"]
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": labels,
        }

    def get_label_encoders(self) -> Dict[str, Dict[str, int]]:
        return self.label_encoders
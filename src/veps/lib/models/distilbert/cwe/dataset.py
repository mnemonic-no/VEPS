import os
import json
import torch
import random
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict

class CWEDataset(Dataset):
    def __init__(
        self,
        directory: Path,
        metadata: pd.DataFrame,
        tokenizer,
        max_length: int = 512,
        label_encoder: Optional[Dict[str, int]] = None,
        max_cwes: int = 3,
    ):
        self.metadata = metadata
        self.directory = directory
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_cwes = max_cwes

        if label_encoder is None:
            all_cwes = set()
            for cwes_json in metadata["cwes"]:
                all_cwes.update(json.loads(cwes_json))
            self.label_encoder = {cwe: idx for idx, cwe in enumerate(sorted(all_cwes))}
        else:
            self.label_encoder = label_encoder

        self.num_classes = len(self.label_encoder)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
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

        cwes = json.loads(row["cwes"])
        if len(cwes) > self.max_cwes:
            cwes = random.sample(cwes, self.max_cwes)

        labels = torch.zeros(self.num_classes)
        for cwe in cwes:
            if cwe in self.label_encoder:
                labels[self.label_encoder[cwe]] = 1

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": labels,
        }

    def get_label_encoder(self) -> Dict[str, int]:
        return self.label_encoder
import json
import random
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split  # noqa: F401  (back-compat)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    DistilBertModel,
    DistilBertPreTrainedModel,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)

from veps import paths as veps_paths
from veps.config import DistilBertHyperparameters

from .base import (
    autocast_context,
    cwe_collate_fn,
    cwe_metrics,
    ensure_pretrained_model_cached,
    get_device,
    load_label_encoders,
    make_dataloader_generator,
    make_grad_scaler,
    make_optimizer_param_groups,
    print_cwe_split_stats,
    save_label_encoders,
    seed_worker,
    set_seed,
    temporal_split,
    tune_cwe_thresholds,
    write_token_length_diagnostic,
)


class DistilBertForMultilabelClassification(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Per-class BCE pos_weight. Defaults to ones (unweighted); the trainer
        # overwrites this from train-fold frequencies via set_pos_weight when
        # class_weight_enabled. Stored as a buffer so it travels with the
        # checkpoint and inference reloads it transparently.
        self.register_buffer("pos_weight", torch.ones(config.num_labels))

        self.post_init()

    def set_pos_weight(self, pos_weight: torch.Tensor) -> None:
        if pos_weight.shape != self.pos_weight.shape:
            raise ValueError(
                f"set_pos_weight: shape {tuple(pos_weight.shape)} != "
                f"({self.num_labels},)"
            )
        self.pos_weight.data.copy_(
            pos_weight.to(self.pos_weight.device, dtype=self.pos_weight.dtype)
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            loss = loss_fct(logits, labels)
            return (loss, logits)
        else:
            return logits


class CWEDataset(Dataset):
    """CWE multi-label dataset.

    ``__getitem__`` returns raw text + multi-hot label tensor. Tokenization
    + padding is deferred to the DataLoader's ``collate_fn`` so each batch
    pads to its own longest sequence instead of the model max.
    """

    def __init__(
        self,
        directory: Path,
        metadata: pd.DataFrame,
        label_encoder: Optional[Dict[str, int]] = None,
        max_cwes: int = 3,
        split: Literal["train", "val", "test"] = "train",
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.directory = directory
        self.max_cwes = max_cwes
        self.split = split

        if label_encoder is None:
            all_cwes: set = set()
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

        cwes = json.loads(row["cwes"])
        if self.split == "train" and len(cwes) > self.max_cwes:
            cwes = random.sample(cwes, self.max_cwes)

        labels = torch.zeros(self.num_classes)
        for cwe in cwes:
            if cwe in self.label_encoder:
                labels[self.label_encoder[cwe]] = 1

        return {"text": content, "labels": labels}

    def get_label_encoder(self) -> Dict[str, int]:
        return self.label_encoder

    def iter_texts(self) -> List[str]:
        """Read all texts for this split (diagnostic use)."""
        out: List[str] = []
        for _, row in self.metadata.iterrows():
            with open(self.directory / row["filename"], encoding="utf-8") as f:
                out.append(f.read())
        return out


class CWETrainer:
    def __init__(
        self,
        data_dir: Path,
        models_dir: Path,
        config: Optional[DistilBertHyperparameters] = None,
    ):
        if config is None:
            config = DistilBertHyperparameters()

        self.config = config
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.model_name = config.model_name
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size_cwe
        self.learning_rate = config.learning_rate
        self.test_months = config.test_months
        self.val_months = config.val_months
        self.max_length = config.max_length
        self.patience = config.patience
        self.seeds = list(config.seeds)
        self.num_workers = config.num_workers
        self.amp_enabled = config.amp_enabled
        self.weight_decay = config.weight_decay
        self.warmup_ratio = config.warmup_ratio
        self.grad_clip_max_norm = config.grad_clip_max_norm
        self.class_weight_enabled = config.class_weight_enabled
        self.cwe_pos_weight_clip = config.cwe_pos_weight_clip
        self.device = get_device()

        self.models_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data(
        self,
        tokenizer: DistilBertTokenizerFast,
        seed: int,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], CWEDataset]:
        """Prepare train, validation, and test data loaders via temporal split.

        Returns the train dataset too so the caller can run a token-length
        diagnostic on the train split before training starts.
        """
        metadata_file = self.data_dir / "cwe_metadata.csv"
        full_metadata = pd.read_csv(metadata_file)

        all_cwes: set = set()
        for cwes_json in full_metadata["cwes"]:
            all_cwes.update(json.loads(cwes_json))
        label_encoder = {cwe: idx for idx, cwe in enumerate(sorted(all_cwes))}

        train_metadata, val_metadata, test_metadata = temporal_split(
            full_metadata, test_months=self.test_months, val_months=self.val_months
        )
        print_cwe_split_stats(train_metadata, val_metadata, test_metadata)

        train_dataset = CWEDataset(
            self.data_dir, train_metadata, label_encoder=label_encoder, split="train"
        )
        val_dataset = CWEDataset(
            self.data_dir, val_metadata, label_encoder=label_encoder, split="val"
        )
        test_dataset = CWEDataset(
            self.data_dir, test_metadata, label_encoder=label_encoder, split="test"
        )

        collate = partial(cwe_collate_fn, tokenizer=tokenizer, max_length=self.max_length)
        generator = make_dataloader_generator(seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
        )

        return train_loader, val_loader, test_loader, label_encoder, train_dataset

    # ------------------------------------------------------------------
    # Multi-seed entry
    # ------------------------------------------------------------------
    def train(self, seeds: Optional[List[int]] = None) -> DistilBertForMultilabelClassification:
        seeds = list(seeds) if seeds else list(self.seeds)
        print(f"Using device: {self.device}")

        ensure_pretrained_model_cached(self.model_name)

        all_test_metrics: List[Dict[str, Any]] = []
        checkpoint_paths: List[str] = []
        last_model: Optional[DistilBertForMultilabelClassification] = None

        for seed in seeds:
            print(f"\n=== Seed {seed} ===")
            path, metrics, last_model = self._train_one_seed(seed)
            checkpoint_paths.append(str(path))
            all_test_metrics.append(metrics)

        self._print_aggregate(all_test_metrics)
        self._save_run_manifest(seeds, checkpoint_paths, all_test_metrics)
        return last_model

    def _train_one_seed(
        self, seed: int
    ) -> Tuple[Path, Dict[str, Any], DistilBertForMultilabelClassification]:
        set_seed(seed)

        tokenizer = DistilBertTokenizerFast.from_pretrained(
            self.model_name, clean_up_tokenization_spaces=True, local_files_only=True
        )

        (
            train_loader,
            val_loader,
            test_loader,
            label_encoder,
            train_dataset,
        ) = self.prepare_data(tokenizer, seed=seed)
        save_label_encoders(label_encoder, self.models_dir / "cwe_label_encoders.json")

        # Diagnostic: train-only token-length histogram. Informs a future
        # max_length decision; does not influence this run.
        write_token_length_diagnostic(
            tokenizer,
            train_dataset.iter_texts(),
            veps_paths.cwe_token_lengths_path(self.models_dir),
            model_label="cwe",
        )

        num_labels = len(label_encoder)
        model = DistilBertForMultilabelClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            local_files_only=True,
        )
        model.to(self.device)

        # NOTE: HF from_pretrained zeros our pos_weight buffer (it's "MISSING"
        # from the pretrained checkpoint), so we always seed it explicitly —
        # either with tuned weights or ones — before training. A pos_weight
        # of 0 would silently zero the positive term in BCE.
        if self.class_weight_enabled:
            pos_weight = self._compute_pos_weight(train_dataset, label_encoder)
        else:
            print("[cwe] class-weighting disabled; using unweighted BCE")
            pos_weight = torch.ones(num_labels)
        model.set_pos_weight(pos_weight)

        checkpoint_path = veps_paths.cwe_checkpoint_path(self.models_dir, seed)
        thresholds_path = veps_paths.cwe_thresholds_path(self.models_dir, seed)

        model, best_thresholds = self._train_model(
            model, train_loader, val_loader, label_encoder,
            checkpoint_path, thresholds_path,
        )

        # Reload best checkpoint, evaluate on test with saved thresholds.
        state = torch.load(checkpoint_path, weights_only=True, map_location=self.device)
        model.load_state_dict(state)
        probs, labels = self._predict_probs(model, test_loader)
        test_metrics = cwe_metrics(probs, labels, best_thresholds, label_encoder)
        print(f"[seed {seed}] test metrics: {self._format_metrics(test_metrics)}")

        return checkpoint_path, test_metrics, model

    def _compute_pos_weight(
        self,
        train_dataset: CWEDataset,
        label_encoder: Dict[str, int],
    ) -> torch.Tensor:
        """Per-class BCE pos_weight = (N - pos_i) / pos_i, clipped.

        Counts are read directly from the train fold's ``cwes`` JSON column
        (ignoring the per-epoch max_cwes subsample, which only affects rows
        with >3 CWEs — a vanishing fraction in practice). Zero-positive
        classes fall back to the clip cap.
        """
        num_classes = len(label_encoder)
        counts = np.zeros(num_classes, dtype=np.float64)
        for cwes_json in train_dataset.metadata["cwes"]:
            for cwe in json.loads(cwes_json):
                idx = label_encoder.get(cwe)
                if idx is not None:
                    counts[idx] += 1.0

        n = float(len(train_dataset))
        clip = float(self.cwe_pos_weight_clip)
        weights = np.full(num_classes, clip, dtype=np.float64)
        nonzero = counts > 0
        weights[nonzero] = (n - counts[nonzero]) / counts[nonzero]
        np.minimum(weights, clip, out=weights)

        idx_to_label = {idx: label for label, idx in label_encoder.items()}
        print(f"[cwe] class-weighting: pos_weight clipped at {clip}")
        print(f"[cwe] {'cwe':<14} {'count':>8} {'pos_weight':>12}")
        ranked = sorted(range(num_classes), key=lambda i: -counts[i])
        for i in ranked:
            print(
                f"[cwe] {idx_to_label[i]:<14} {int(counts[i]):>8} "
                f"{weights[i]:>12.3f}"
            )

        return torch.tensor(weights, dtype=torch.float32)

    def _train_model(
        self,
        model: DistilBertForMultilabelClassification,
        train_loader: DataLoader,
        val_loader: DataLoader,
        label_encoder: Dict[str, int],
        checkpoint_path: Path,
        thresholds_path: Path,
    ) -> Tuple[DistilBertForMultilabelClassification, np.ndarray]:
        """Train with early stopping on macro_f1 (using tuned thresholds)."""
        param_groups = make_optimizer_param_groups(model, self.weight_decay)
        optimizer = AdamW(param_groups, lr=self.learning_rate)
        total_steps = max(len(train_loader) * self.num_epochs, 1)
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        print(
            f"[cwe] optimizer: AdamW lr={self.learning_rate} "
            f"weight_decay={self.weight_decay} "
            f"grad_clip={self.grad_clip_max_norm}"
        )
        print(
            f"[cwe] schedule: total_steps={total_steps} "
            f"warmup_steps={warmup_steps} (warmup_ratio={self.warmup_ratio})"
        )
        scaler = make_grad_scaler(self.device)
        patience = self.patience

        best_metric = -float("inf")
        best_thresholds = np.full(len(label_encoder), 0.5, dtype=float)
        epochs_without_improvement = 0

        for epoch in range(self.num_epochs):
            model.train()
            train_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                with autocast_context(self.device, enabled=self.amp_enabled):
                    loss, _ = model(
                        input_ids, attention_mask=attention_mask, labels=labels
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=self.grad_clip_max_norm
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                train_loss += loss.item()
                progress_bar.set_postfix(
                    {
                        "train_loss": f"{loss.item():.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )

            val_loss, val_probs, val_labels = self._predict_probs_with_loss(model, val_loader)
            tuned = tune_cwe_thresholds(val_probs, val_labels)
            val_metrics = cwe_metrics(val_probs, val_labels, tuned, label_encoder)
            avg_train_loss = train_loss / max(len(train_loader), 1)

            print(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Val MacroF1: {val_metrics['macro_f1']:.4f} - "
                f"Val MicroF1: {val_metrics['micro_f1']:.4f}"
            )

            if val_metrics["macro_f1"] > best_metric:
                best_metric = val_metrics["macro_f1"]
                best_thresholds = tuned
                epochs_without_improvement = 0
                torch.save(model.state_dict(), checkpoint_path)
                self._save_thresholds(best_thresholds, label_encoder, thresholds_path)
                print(f"New best model saved to {checkpoint_path}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

        return model, best_thresholds

    # ------------------------------------------------------------------
    # Eval helpers
    # ------------------------------------------------------------------
    def _predict_probs(
        self, model: DistilBertForMultilabelClassification, loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        all_probs: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                with autocast_context(self.device, enabled=self.amp_enabled):
                    _, logits = model(
                        input_ids, attention_mask=attention_mask, labels=labels
                    )
                probs = torch.sigmoid(logits.float()).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.cpu().numpy())
        return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)

    def _predict_probs_with_loss(
        self, model: DistilBertForMultilabelClassification, loader: DataLoader
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        model.eval()
        all_probs: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                with autocast_context(self.device, enabled=self.amp_enabled):
                    loss, logits = model(
                        input_ids, attention_mask=attention_mask, labels=labels
                    )
                total_loss += loss.item()
                n += 1
                probs = torch.sigmoid(logits.float()).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.cpu().numpy())
        avg_loss = total_loss / n if n else 0.0
        return (
            avg_loss,
            np.concatenate(all_probs, axis=0),
            np.concatenate(all_labels, axis=0),
        )

    def _save_thresholds(
        self,
        thresholds: np.ndarray,
        label_encoder: Dict[str, int],
        path: Path,
    ) -> None:
        idx_to_label = {idx: label for label, idx in label_encoder.items()}
        payload = {idx_to_label[i]: float(thresholds[i]) for i in range(len(thresholds))}
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    @staticmethod
    def _format_metrics(metrics: Dict[str, Any]) -> str:
        return ", ".join(
            f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))
        )

    def _print_aggregate(self, all_metrics: List[Dict[str, Any]]) -> None:
        if not all_metrics:
            return
        scalar_keys = [
            k for k, v in all_metrics[0].items() if isinstance(v, (int, float))
        ]
        if len(all_metrics) == 1:
            print("[cwe-aggregate] single-seed run, no aggregation")
            return
        print("[cwe-aggregate] mean ± std across seeds:")
        for k in scalar_keys:
            values = np.array([m[k] for m in all_metrics], dtype=float)
            print(
                f"[cwe-aggregate]   {k}: {values.mean():.4f} ± {values.std():.4f}"
            )

    def _save_run_manifest(
        self,
        seeds: List[int],
        checkpoint_paths: List[str],
        all_metrics: List[Dict[str, Any]],
    ) -> None:
        manifest = {
            "model": "cwe",
            "seeds": list(seeds),
            "checkpoints": list(checkpoint_paths),
            "test_metrics": [
                {k: v for k, v in m.items() if isinstance(v, (int, float))}
                for m in all_metrics
            ],
        }
        path = veps_paths.cwe_run_manifest_path(self.models_dir)
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[cwe] wrote run manifest to {path}")


class CWEClassifier:
    def __init__(
        self,
        model_path: Path,
        model_name: str = "distilbert-base-uncased",
        seed: int = 42,
    ):
        self.device = get_device()
        self.model_name = model_name
        self.seed = seed

        ensure_pretrained_model_cached(model_name)

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            model_name, clean_up_tokenization_spaces=True, local_files_only=True
        )

        label_encoder_path = model_path.parent / "cwe_label_encoders.json"
        self.label_encoder: Dict[str, int] = load_label_encoders(label_encoder_path)

        num_labels = len(self.label_encoder)
        self.model = DistilBertForMultilabelClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            local_files_only=True,
        )

        state_dict = torch.load(model_path, weights_only=True, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Per-class thresholds; fall back to 0.5 if the sidecar is absent.
        self.thresholds = np.full(num_labels, 0.5, dtype=float)
        thresholds_path = veps_paths.cwe_thresholds_path(model_path.parent, seed)
        if thresholds_path.exists():
            with open(thresholds_path, "r") as f:
                payload = json.load(f)
            for cwe, idx in self.label_encoder.items():
                if cwe in payload:
                    self.thresholds[idx] = float(payload[cwe])
        else:
            print(
                f"[CWEClassifier] note: {thresholds_path.name} not found; "
                "using 0.5 for all classes."
            )

    def predict(
        self,
        text: str,
        max_labels: int = 3,
    ) -> List[str]:
        """Predict CWE labels for a text description using per-class thresholds."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]

        idx_to_label = {idx: label for label, idx in self.label_encoder.items()}

        # Apply per-class thresholds, then rank surviving classes by probability.
        passing = [i for i in range(len(probabilities)) if probabilities[i] >= self.thresholds[i]]
        passing.sort(key=lambda i: probabilities[i], reverse=True)
        predicted_labels = [idx_to_label[i] for i in passing]

        if not predicted_labels:
            top_index = int(np.argmax(probabilities))
            predicted_labels = [idx_to_label[top_index]]

        return predicted_labels[:max_labels]

    def predict_with_confidence(
        self,
        text: str,
        max_labels: int = 3,
    ) -> List[Dict[str, float]]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
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
                "cwe": idx_to_label[int(idx)],
                "confidence": float(probabilities[int(idx)]),
            })

        return results

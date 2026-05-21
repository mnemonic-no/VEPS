import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from cvss import CVSS3
from sklearn.model_selection import train_test_split  # noqa: F401  (kept for back-compat callers)
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
    CVSS_METRIC_ORDER,
    autocast_context,
    compute_inverse_frequency_class_weights,
    cvss_collate_fn,
    cvss_metrics,
    ensure_pretrained_model_cached,
    get_device,
    load_label_encoders,
    make_dataloader_generator,
    make_grad_scaler,
    make_optimizer_param_groups,
    print_cvss_split_stats,
    save_label_encoders,
    seed_worker,
    set_seed,
    temporal_split,
    weighted_head_loss,
    write_token_length_diagnostic,
)


class MultiOutputDistilBert(DistilBertPreTrainedModel):
    def __init__(self, config, num_labels_list=None):
        super().__init__(config)
        if num_labels_list is None:
            raise ValueError(
                "MultiOutputDistilBert requires `num_labels_list`; pass it via "
                "`from_pretrained(model_name, num_labels_list=[...])`."
            )

        self.distilbert = DistilBertModel(config)
        self.classifiers = nn.ModuleList(
            [nn.Linear(config.dim, num_labels) for num_labels in num_labels_list]
        )
        self.num_labels_list = list(num_labels_list)
        # Populated by the trainer once class weights are computed; defaults
        # to unweighted CE so the model is usable out-of-the-box (inference
        # path and tests that bypass the trainer).
        self.loss_fns: List[nn.Module] = [
            nn.CrossEntropyLoss() for _ in self.num_labels_list
        ]
        self.head_loss_weighting: str = "log_k"

        self.post_init()

    def set_loss_fns(
        self, loss_fns: List[nn.Module], head_loss_weighting: str
    ) -> None:
        """Inject per-head loss functions + cross-head weighting strategy."""
        if len(loss_fns) != len(self.num_labels_list):
            raise ValueError(
                f"set_loss_fns: expected {len(self.num_labels_list)} fns, "
                f"got {len(loss_fns)}"
            )
        self.loss_fns = list(loss_fns)
        self.head_loss_weighting = head_loss_weighting

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0]
        logits = [classifier(pooled_output) for classifier in self.classifiers]

        if labels is not None:
            per_head = [
                fn(logit, label)
                for fn, logit, label in zip(self.loss_fns, logits, labels)
            ]
            loss = weighted_head_loss(
                per_head, self.num_labels_list, self.head_loss_weighting
            )
            return {"loss": loss, "logits": logits} if return_dict else (loss, logits)

        return {"logits": logits} if return_dict else logits


class TextMultiOutputDataset(Dataset):
    """CVSS multi-output dataset.

    ``__getitem__`` returns raw text + label tensors. Tokenization +
    padding is deferred to the DataLoader's ``collate_fn`` so each batch
    pads to its own longest sequence instead of the model max.
    """

    def __init__(
        self,
        directory: Path,
        metadata: pd.DataFrame,
        label_encoders: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.directory = directory
        self.label_columns = [
            c for c in self.metadata.columns
            if c not in ("filename", "cve_id", "published_date")
        ]

        if label_encoders is None:
            self.label_encoders = {
                column: {
                    label: idx
                    for idx, label in enumerate(sorted(metadata[column].unique()))
                }
                for column in self.label_columns
            }
        else:
            self.label_encoders = label_encoders

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.metadata.iloc[idx]
        file_path = self.directory / row["filename"]

        with open(file_path, encoding="utf-8") as file:
            content = file.read()

        labels = tuple(
            torch.tensor(self.label_encoders[column][row[column]])
            for column in self.label_columns
        )

        return {"text": content, "labels": labels}

    def get_label_encoders(self) -> Dict[str, Dict[str, int]]:
        return self.label_encoders

    def iter_texts(self) -> List[str]:
        """Read all texts for this split (diagnostic use)."""
        out: List[str] = []
        for _, row in self.metadata.iterrows():
            with open(self.directory / row["filename"], encoding="utf-8") as f:
                out.append(f.read())
        return out


class CVSSTrainer:
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
        self.batch_size = config.batch_size_cvss
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
        self.head_loss_weighting = config.head_loss_weighting
        self.class_weight_enabled = config.class_weight_enabled
        self.class_weight_clip = config.class_weight_clip
        self.selection_metric = config.cvss_selection_metric
        self.device = get_device()

        self.models_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data(
        self,
        tokenizer: DistilBertTokenizerFast,
        seed: int,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Dict[str, int]], TextMultiOutputDataset]:
        """Prepare train, validation, and test data loaders via temporal split.

        Returns the train dataset too so the caller can run a token-length
        diagnostic on the train split before training starts.
        """
        metadata_file = self.data_dir / "cvss_metadata.csv"
        full_metadata = pd.read_csv(metadata_file)

        label_columns = [
            c for c in full_metadata.columns
            if c not in ("filename", "cve_id", "published_date")
        ]
        label_encoders = {
            column: {
                label: idx
                for idx, label in enumerate(sorted(full_metadata[column].unique()))
            }
            for column in label_columns
        }

        train_metadata, val_metadata, test_metadata = temporal_split(
            full_metadata, test_months=self.test_months, val_months=self.val_months
        )
        print_cvss_split_stats(train_metadata, val_metadata, test_metadata)

        train_dataset = TextMultiOutputDataset(
            self.data_dir, train_metadata, label_encoders=label_encoders
        )
        val_dataset = TextMultiOutputDataset(
            self.data_dir, val_metadata, label_encoders=label_encoders
        )
        test_dataset = TextMultiOutputDataset(
            self.data_dir, test_metadata, label_encoders=label_encoders
        )

        collate = partial(cvss_collate_fn, tokenizer=tokenizer, max_length=self.max_length)
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

        return train_loader, val_loader, test_loader, label_encoders, train_dataset

    # ------------------------------------------------------------------
    # Multi-seed entry point
    # ------------------------------------------------------------------
    def train(self, seeds: Optional[List[int]] = None) -> MultiOutputDistilBert:
        """Run training across one or more seeds."""
        seeds = list(seeds) if seeds else list(self.seeds)
        print(f"Using device: {self.device}")

        ensure_pretrained_model_cached(self.model_name)

        all_test_metrics: List[Dict[str, Any]] = []
        checkpoint_paths: List[str] = []
        last_model: Optional[MultiOutputDistilBert] = None

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
    ) -> Tuple[Path, Dict[str, Any], MultiOutputDistilBert]:
        set_seed(seed)

        tokenizer = DistilBertTokenizerFast.from_pretrained(
            self.model_name, clean_up_tokenization_spaces=True, local_files_only=True
        )

        (
            train_loader,
            val_loader,
            test_loader,
            label_encoders,
            train_dataset,
        ) = self.prepare_data(tokenizer, seed=seed)
        save_label_encoders(label_encoders, self.models_dir / "cvss_label_encoders.json")

        # Diagnostic: train-only token-length histogram. Informs a future
        # max_length decision; does not influence this run.
        write_token_length_diagnostic(
            tokenizer,
            train_dataset.iter_texts(),
            veps_paths.cvss_token_lengths_path(self.models_dir),
            model_label="cvss",
        )

        num_labels = [len(encoder) for encoder in label_encoders.values()]
        model = MultiOutputDistilBert.from_pretrained(
            self.model_name,
            num_labels_list=num_labels,
            local_files_only=True,
        )
        model.to(self.device)

        loss_fns = self._build_loss_fns(train_dataset, label_encoders)
        model.set_loss_fns(loss_fns, self.head_loss_weighting)

        checkpoint_path = veps_paths.cvss_checkpoint_path(self.models_dir, seed)
        model = self._train_model(
            model, train_loader, val_loader, label_encoders, checkpoint_path
        )

        # Test eval — reload best checkpoint and evaluate.
        state = torch.load(checkpoint_path, weights_only=True, map_location=self.device)
        model.load_state_dict(state)
        test_metrics = self._evaluate(model, test_loader, label_encoders)
        print(f"[seed {seed}] test metrics: {self._format_metrics(test_metrics)}")

        return checkpoint_path, test_metrics, model

    def _build_loss_fns(
        self,
        train_dataset: "TextMultiOutputDataset",
        label_encoders: Dict[str, Dict[str, int]],
    ) -> List[nn.Module]:
        """Build per-head loss functions, optionally with inverse-frequency weights.

        Class counts are taken from the train fold's metadata only. Weights
        live on the same device as the model.
        """
        head_names = list(label_encoders.keys())
        if not self.class_weight_enabled:
            print("[cvss] class-weighting disabled; using unweighted CE per head")
            return [nn.CrossEntropyLoss() for _ in head_names]

        loss_fns: List[nn.Module] = []
        for name in head_names:
            encoder = label_encoders[name]
            k = len(encoder)
            counts = [0] * k
            col = train_dataset.metadata[name]
            for label, idx in encoder.items():
                counts[idx] = int((col == label).sum())
            weight_tensor = compute_inverse_frequency_class_weights(
                counts, self.class_weight_clip
            ).to(self.device)
            loss_fns.append(nn.CrossEntropyLoss(weight=weight_tensor))
            print(
                f"[cvss] head={name} counts={counts} "
                f"weights={[round(w, 3) for w in weight_tensor.tolist()]}"
            )
        return loss_fns

    def _train_model(
        self,
        model: MultiOutputDistilBert,
        train_loader: DataLoader,
        val_loader: DataLoader,
        label_encoders: Dict[str, Dict[str, int]],
        checkpoint_path: Path,
    ) -> MultiOutputDistilBert:
        """Train with early stopping on the configured selection metric."""
        param_groups = make_optimizer_param_groups(model, self.weight_decay)
        optimizer = AdamW(param_groups, lr=self.learning_rate)
        total_steps = max(len(train_loader) * self.num_epochs, 1)
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        print(f"[cvss] selection_metric={self.selection_metric}")
        print(
            f"[cvss] optimizer: AdamW lr={self.learning_rate} "
            f"weight_decay={self.weight_decay} "
            f"head_loss_weighting={self.head_loss_weighting} "
            f"grad_clip={self.grad_clip_max_norm}"
        )
        print(
            f"[cvss] schedule: total_steps={total_steps} "
            f"warmup_steps={warmup_steps} (warmup_ratio={self.warmup_ratio})"
        )
        scaler = make_grad_scaler(self.device)
        patience = self.patience

        best_metric = -float("inf")
        epochs_without_improvement = 0

        for epoch in range(self.num_epochs):
            model.train()
            train_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = tuple(label.to(self.device) for label in batch["labels"])

                optimizer.zero_grad()
                with autocast_context(self.device, enabled=self.amp_enabled):
                    outputs = model(
                        input_ids, attention_mask=attention_mask, labels=labels
                    )
                    loss = outputs["loss"]

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

            val_loss, val_metrics = self._evaluate_with_loss(
                model, val_loader, label_encoders
            )
            avg_train_loss = train_loss / max(len(train_loader), 1)
            selection_value = val_metrics[self.selection_metric]

            print(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Val ExactMatch: {val_metrics['vector_exact_match']:.4f} - "
                f"Val MeanMacroF1: {val_metrics['mean_macro_f1']:.4f} - "
                f"Val BaseScoreMAE: {val_metrics['base_score_mae']:.4f}"
            )

            if selection_value > best_metric:
                best_metric = selection_value
                epochs_without_improvement = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

        return model

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def _predict_split(
        self,
        model: MultiOutputDistilBert,
        loader: DataLoader,
        label_encoders: Dict[str, Dict[str, int]],
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], float]:
        model.eval()
        head_names = list(label_encoders.keys())
        decoders = {
            name: {idx: label for label, idx in label_encoders[name].items()}
            for name in head_names
        }
        preds: Dict[str, List[str]] = {h: [] for h in head_names}
        truths: Dict[str, List[str]] = {h: [] for h in head_names}
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = tuple(label.to(self.device) for label in batch["labels"])
                with autocast_context(self.device, enabled=self.amp_enabled):
                    outputs = model(
                        input_ids, attention_mask=attention_mask, labels=labels
                    )
                total_loss += outputs["loss"].item()
                n_batches += 1
                logits = outputs["logits"]
                for i, name in enumerate(head_names):
                    pred_idx = torch.argmax(logits[i], dim=-1).cpu().tolist()
                    truth_idx = labels[i].cpu().tolist()
                    preds[name].extend(decoders[name][p] for p in pred_idx)
                    truths[name].extend(decoders[name][t] for t in truth_idx)

        # Re-key by the canonical CVSS metric order names used by cvss_metrics.
        preds_by_metric: Dict[str, List[str]] = {}
        truths_by_metric: Dict[str, List[str]] = {}
        for metric in CVSS_METRIC_ORDER:
            if metric in preds:
                preds_by_metric[metric] = preds[metric]
                truths_by_metric[metric] = truths[metric]
            else:
                raise KeyError(
                    f"CVSS head '{metric}' missing from label encoders "
                    f"(found: {head_names})."
                )

        avg_loss = total_loss / n_batches if n_batches else 0.0
        return preds_by_metric, truths_by_metric, avg_loss

    def _evaluate(
        self,
        model: MultiOutputDistilBert,
        loader: DataLoader,
        label_encoders: Dict[str, Dict[str, int]],
    ) -> Dict[str, Any]:
        preds, truths, _ = self._predict_split(model, loader, label_encoders)
        return cvss_metrics(preds, truths)

    def _evaluate_with_loss(
        self,
        model: MultiOutputDistilBert,
        loader: DataLoader,
        label_encoders: Dict[str, Dict[str, int]],
    ) -> Tuple[float, Dict[str, Any]]:
        preds, truths, avg_loss = self._predict_split(model, loader, label_encoders)
        return avg_loss, cvss_metrics(preds, truths)

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
            print("[cvss-aggregate] single-seed run, no aggregation")
            return
        print("[cvss-aggregate] mean ± std across seeds:")
        for k in scalar_keys:
            values = np.array([m[k] for m in all_metrics], dtype=float)
            print(
                f"[cvss-aggregate]   {k}: {values.mean():.4f} ± {values.std():.4f}"
            )

    def _save_run_manifest(
        self,
        seeds: List[int],
        checkpoint_paths: List[str],
        all_metrics: List[Dict[str, Any]],
    ) -> None:
        manifest = {
            "model": "cvss",
            "seeds": list(seeds),
            "checkpoints": list(checkpoint_paths),
            "test_metrics": [
                {k: v for k, v in m.items() if isinstance(v, (int, float))}
                for m in all_metrics
            ],
        }
        path = veps_paths.cvss_run_manifest_path(self.models_dir)
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[cvss] wrote run manifest to {path}")


class CVSSClassifier:
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

        label_encoders_path = model_path.parent / "cvss_label_encoders.json"
        self.label_encoders = load_label_encoders(label_encoders_path)

        num_labels = [len(encoder) for encoder in self.label_encoders.values()]
        self.model = MultiOutputDistilBert.from_pretrained(
            model_name,
            num_labels_list=num_labels,
            local_files_only=True,
        )

        state_dict = torch.load(model_path, weights_only=True, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict_vector(self, text: str) -> str:
        """Predict CVSS vector string from text description."""
        tokens = self.tokenizer(
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

        cvss_components: Dict[str, str] = {}
        for idx, logit in enumerate(logits):
            predicted_idx = torch.argmax(logit, dim=-1).item()
            classifier_name = list(self.label_encoders.keys())[idx]
            label_decoder = {v: k for k, v in self.label_encoders[classifier_name].items()}
            cvss_components[classifier_name] = label_decoder[predicted_idx]

        cvss_vector_parts = [f"{key}:{cvss_components[key]}" for key in CVSS_METRIC_ORDER]
        return "CVSS:3.1/" + "/".join(cvss_vector_parts)

    def predict_with_metrics(self, text: str) -> Dict:
        """Predict CVSS vector and calculate full metrics."""
        vector = self.predict_vector(text)

        try:
            cvss_calc = CVSS3(vector)
            cvss_calc.compute_isc()
            cvss_calc.compute_esc()

            vector_parts: Dict[str, str] = {}
            parts = vector.replace("CVSS:3.1/", "").split("/")
            for part in parts:
                key, value = part.split(":")
                vector_parts[key] = value

            component_mapping = {
                "AV": ("attackVector", {"N": "NETWORK", "A": "ADJACENT_NETWORK", "L": "LOCAL", "P": "PHYSICAL"}),
                "AC": ("attackComplexity", {"L": "LOW", "H": "HIGH"}),
                "PR": ("privilegesRequired", {"N": "NONE", "L": "LOW", "H": "HIGH"}),
                "UI": ("userInteraction", {"N": "NONE", "R": "REQUIRED"}),
                "S": ("scope", {"U": "UNCHANGED", "C": "CHANGED"}),
                "C": ("confidentialityImpact", {"N": "NONE", "L": "LOW", "H": "HIGH"}),
                "I": ("integrityImpact", {"N": "NONE", "L": "LOW", "H": "HIGH"}),
                "A": ("availabilityImpact", {"N": "NONE", "L": "LOW", "H": "HIGH"}),
            }

            cvss_data: Dict[str, Any] = {
                "version": "3.1",
                "vectorString": vector,
                "baseScore": float(cvss_calc.base_score),
                "baseSeverity": cvss_calc.severities()[0].upper(),
            }

            for key, (full_name, value_map) in component_mapping.items():
                if key in vector_parts:
                    cvss_data[full_name] = value_map.get(vector_parts[key], vector_parts[key])

            return {
                "cvssData": cvss_data,
                "exploitabilityScore": float(cvss_calc.esc),
                "impactScore": float(cvss_calc.isc),
            }

        except Exception as e:
            print(f"Error calculating CVSS metrics: {e}")
            return {"vectorString": vector, "error": str(e)}

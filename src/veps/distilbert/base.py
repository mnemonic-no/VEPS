import contextlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


CVSS_METRIC_ORDER = ("AV", "AC", "PR", "UI", "S", "C", "I", "A")

# Default DataLoader worker count. Conservative — large enough to overlap
# file I/O and tokenization with the GPU/MPS step, small enough not to
# flood macOS with processes on a laptop run.
DEFAULT_NUM_WORKERS = 2


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch (incl. CUDA + MPS when available)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


def ensure_pretrained_model_cached(model_name: str) -> None:
    """Make sure `model_name` (weights + tokenizer) is in the local HF cache.

    On a cold cache, performs a one-time online fetch and prints a notice.
    Subsequent calls are silent no-ops. Idempotent.
    """
    from huggingface_hub import try_to_load_from_cache
    from huggingface_hub.constants import HF_HUB_CACHE
    from huggingface_hub.file_download import _CACHED_NO_EXIST

    cfg = try_to_load_from_cache(repo_id=model_name, filename="config.json")
    tok = try_to_load_from_cache(repo_id=model_name, filename="tokenizer.json")
    cfg_ok = cfg is not None and cfg is not _CACHED_NO_EXIST
    tok_ok = tok is not None and tok is not _CACHED_NO_EXIST
    if cfg_ok and tok_ok:
        return

    from transformers import AutoTokenizer, DistilBertModel
    if not cfg_ok:
        print(
            f"[veps] DistilBERT weights not found in local cache "
            f"({HF_HUB_CACHE}). Fetching {model_name} once..."
        )
        DistilBertModel.from_pretrained(model_name)
    if not tok_ok:
        print(
            f"[veps] DistilBERT tokenizer not found in local cache "
            f"({HF_HUB_CACHE}). Fetching {model_name} tokenizer once..."
        )
        AutoTokenizer.from_pretrained(model_name)
    print(f"[veps] Cached {model_name} for future runs.")


def save_label_encoders(encoders: Dict[str, Any], path: Path) -> None:
    """Save label encoders to JSON file.

    Supports both flat encoders ({str: int}) and nested encoders
    ({str: {str: int}}). Nested values are normalized to plain dicts so
    that callers can pass mapping-like values produced by sklearn etc.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    def _normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return value
        if hasattr(value, "items"):
            return dict(value)
        return value

    normalized = {k: _normalize(v) for k, v in encoders.items()}
    with open(path, "w") as f:
        json.dump(normalized, f)


def load_label_encoders(path: Path) -> Dict[str, Any]:
    """Load label encoders from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Label encoders not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def load_json_data(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load JSON data from file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return None


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------

def temporal_split(
    metadata: pd.DataFrame,
    test_months: int = 6,
    val_months: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Relative-window temporal split based on the ``published_date`` column.

    test  = rows with published_date >= max(published_date) - test_months
    val   = rows with test_start - val_months <= published_date < test_start
    train = everything earlier

    Returns ``(train_df, val_df, test_df)``. Prints the resolved cutoff dates
    and per-split row counts.
    """
    if "published_date" not in metadata.columns:
        raise ValueError(
            "temporal_split requires a 'published_date' column on the dataframe."
        )

    df = metadata.copy()
    parsed = pd.to_datetime(
        df["published_date"], errors="coerce", utc=True, format="ISO8601"
    )
    n_null = int(parsed.isna().sum())
    if n_null:
        print(
            f"[temporal_split] dropping {n_null} rows with unparseable "
            f"published_date"
        )
        df = df.loc[~parsed.isna()].copy()
        parsed = parsed.loc[~parsed.isna()]

    df["_parsed_date"] = parsed.dt.tz_convert(None)
    df = df.sort_values("_parsed_date", kind="mergesort").reset_index(drop=True)

    max_date = df["_parsed_date"].max()
    test_start = max_date - pd.DateOffset(months=test_months)
    val_start = test_start - pd.DateOffset(months=val_months)

    train_mask = df["_parsed_date"] < val_start
    val_mask = (df["_parsed_date"] >= val_start) & (df["_parsed_date"] < test_start)
    test_mask = df["_parsed_date"] >= test_start

    train_df = df.loc[train_mask].drop(columns=["_parsed_date"]).reset_index(drop=True)
    val_df = df.loc[val_mask].drop(columns=["_parsed_date"]).reset_index(drop=True)
    test_df = df.loc[test_mask].drop(columns=["_parsed_date"]).reset_index(drop=True)

    print(
        f"[temporal_split] max_date={max_date.date()} "
        f"val_start={val_start.date()} test_start={test_start.date()}"
    )
    print(
        f"[temporal_split] train={len(train_df)} "
        f"val={len(val_df)} test={len(test_df)} "
        f"(total={len(df)})"
    )

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# CVSS helpers
# ---------------------------------------------------------------------------

def severity_from_score(base_score: float) -> str:
    """CVSS v3 severity bucket from base score."""
    if base_score <= 0.0:
        return "NONE"
    if base_score < 4.0:
        return "LOW"
    if base_score < 7.0:
        return "MEDIUM"
    if base_score < 9.0:
        return "HIGH"
    return "CRITICAL"


def _cvss_vector_from_row(row: pd.Series) -> str:
    parts = [f"{k}:{row[k]}" for k in CVSS_METRIC_ORDER]
    return "CVSS:3.1/" + "/".join(parts)


def _base_score_and_severity(vector: str) -> Tuple[float, str]:
    from cvss import CVSS3
    calc = CVSS3(vector)
    score = float(calc.base_score)
    return score, severity_from_score(score)


def print_cvss_split_stats(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    """Print per-severity-bucket counts for each split."""
    buckets = ("NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL")
    cache: Dict[str, str] = {}

    def _bucket(row: pd.Series) -> str:
        vec = _cvss_vector_from_row(row)
        sev = cache.get(vec)
        if sev is None:
            try:
                _, sev = _base_score_and_severity(vec)
            except Exception:
                sev = "UNKNOWN"
            cache[vec] = sev
        return sev

    def _counts(df: pd.DataFrame) -> Counter:
        if df.empty:
            return Counter()
        return Counter(df.apply(_bucket, axis=1))

    train_c, val_c, test_c = _counts(train_df), _counts(val_df), _counts(test_df)
    header = f"{'bucket':<10} {'train':>8} {'val':>8} {'test':>8}"
    print("[cvss-split-stats] " + header)
    for b in buckets:
        print(
            f"[cvss-split-stats] {b:<10} {train_c.get(b, 0):>8} "
            f"{val_c.get(b, 0):>8} {test_c.get(b, 0):>8}"
        )


def print_cwe_split_stats(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    """Print row counts and top-10 CWE counts for each split."""
    def _counter(df: pd.DataFrame) -> Counter:
        c: Counter = Counter()
        for cwes_json in df.get("cwes", []):
            try:
                for cwe in json.loads(cwes_json):
                    c[cwe] += 1
            except Exception:
                continue
        return c

    train_c, val_c, test_c = _counter(train_df), _counter(val_df), _counter(test_df)
    print(
        f"[cwe-split-stats] rows train={len(train_df)} "
        f"val={len(val_df)} test={len(test_df)}"
    )
    top10 = [cwe for cwe, _ in train_c.most_common(10)]
    print(f"[cwe-split-stats] {'cwe':<14} {'train':>8} {'val':>8} {'test':>8}")
    for cwe in top10:
        print(
            f"[cwe-split-stats] {cwe:<14} {train_c.get(cwe, 0):>8} "
            f"{val_c.get(cwe, 0):>8} {test_c.get(cwe, 0):>8}"
        )


# ---------------------------------------------------------------------------
# Metric suites
# ---------------------------------------------------------------------------

def cvss_metrics(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
) -> Dict[str, float]:
    """Compute the full CVSS metric suite.

    Returns a dict with per-head macro-F1 (keys ``f1_<head>``),
    ``vector_exact_match``, ``hamming_distance`` (mean unnormalized count
    of differing heads), ``base_score_mae`` and ``severity_bucket_accuracy``.
    """
    heads = list(CVSS_METRIC_ORDER)
    missing = [h for h in heads if h not in predictions or h not in ground_truth]
    if missing:
        raise ValueError(f"cvss_metrics missing heads: {missing}")

    n = len(predictions[heads[0]])
    if any(len(predictions[h]) != n or len(ground_truth[h]) != n for h in heads):
        raise ValueError("cvss_metrics: prediction/truth length mismatch")

    results: Dict[str, float] = {}

    # Per-head macro-F1
    for h in heads:
        results[f"f1_{h}"] = float(
            f1_score(ground_truth[h], predictions[h], average="macro", zero_division=0)
        )

    # Unweighted mean of per-head macro-F1 across the 8 CVSS heads.
    results["mean_macro_f1"] = float(
        sum(results[f"f1_{h}"] for h in heads) / len(heads)
    )

    # Vector exact match + hamming distance
    exact = 0
    hamming_total = 0
    for i in range(n):
        diffs = sum(
            1 for h in heads if predictions[h][i] != ground_truth[h][i]
        )
        if diffs == 0:
            exact += 1
        hamming_total += diffs
    results["vector_exact_match"] = exact / n if n else 0.0
    results["hamming_distance"] = hamming_total / n if n else 0.0

    # Base-score MAE + severity-bucket accuracy
    mae_sum = 0.0
    severity_correct = 0
    for i in range(n):
        pred_row = pd.Series({h: predictions[h][i] for h in heads})
        truth_row = pd.Series({h: ground_truth[h][i] for h in heads})
        pred_vec = _cvss_vector_from_row(pred_row)
        truth_vec = _cvss_vector_from_row(truth_row)
        try:
            pred_score, pred_sev = _base_score_and_severity(pred_vec)
        except Exception:
            pred_score, pred_sev = 0.0, "UNKNOWN"
        try:
            truth_score, truth_sev = _base_score_and_severity(truth_vec)
        except Exception:
            truth_score, truth_sev = 0.0, "UNKNOWN"
        mae_sum += abs(pred_score - truth_score)
        if pred_sev == truth_sev:
            severity_correct += 1
    results["base_score_mae"] = mae_sum / n if n else 0.0
    results["severity_bucket_accuracy"] = severity_correct / n if n else 0.0

    return results


def cwe_metrics(
    pred_probs: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
    encoder: Dict[str, int],
) -> Dict[str, Any]:
    """Compute the full CWE metric suite.

    Returns ``micro_f1``, ``macro_f1``, ``hamming_loss``,
    ``precision_at_1/2/3`` and a per-class F1 dict.
    """
    from sklearn.metrics import hamming_loss

    pred_probs = np.asarray(pred_probs)
    labels = np.asarray(labels)
    if pred_probs.shape != labels.shape:
        raise ValueError(
            f"cwe_metrics: pred_probs {pred_probs.shape} vs labels "
            f"{labels.shape} shape mismatch"
        )

    num_classes = pred_probs.shape[1]
    if np.isscalar(thresholds):
        thr = np.full(num_classes, float(thresholds))
    else:
        thr = np.asarray(thresholds, dtype=float)
        if thr.shape != (num_classes,):
            raise ValueError(
                f"cwe_metrics: thresholds shape {thr.shape} != ({num_classes},)"
            )

    preds = (pred_probs >= thr[None, :]).astype(int)

    results: Dict[str, Any] = {
        "micro_f1": float(
            f1_score(labels, preds, average="micro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(labels, preds, average="macro", zero_division=0)
        ),
        "hamming_loss": float(hamming_loss(labels, preds)),
    }

    # Precision @ k
    n = labels.shape[0]
    for k in (1, 2, 3):
        if num_classes < k or n == 0:
            results[f"precision_at_{k}"] = 0.0
            continue
        topk_idx = np.argpartition(-pred_probs, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(n)[:, None]
        hits = labels[rows, topk_idx].sum(axis=1)
        results[f"precision_at_{k}"] = float((hits / k).mean())

    # Per-class F1 (CWE-id keyed)
    idx_to_label = {idx: label for label, idx in encoder.items()}
    per_class = f1_score(labels, preds, average=None, zero_division=0)
    results["per_class_f1"] = {
        idx_to_label[i]: float(per_class[i]) for i in range(num_classes)
    }

    return results


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# DataLoader plumbing: seeded multi-worker init + reproducible generator
# ---------------------------------------------------------------------------

def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` that seeds numpy + python ``random``.

    Pattern from the PyTorch reproducibility docs: each worker derives its
    seed from ``torch.initial_seed()`` (which the main process sets per
    worker based on the DataLoader's generator).
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataloader_generator(seed: int) -> torch.Generator:
    """Generator passed to DataLoader for shuffle reproducibility."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


# ---------------------------------------------------------------------------
# Collate functions — pad once per batch instead of once per example
# ---------------------------------------------------------------------------

def cvss_collate_fn(
    batch: List[Dict[str, Any]],
    tokenizer: Any,
    max_length: int,
) -> Dict[str, Any]:
    """Batched-tokenization collate for CVSS multi-output dataset.

    Each ``batch[i]`` is ``{"text": str, "labels": Tuple[Tensor, ...]}``.
    Pads to the longest sequence in the batch (capped at ``max_length``).
    """
    texts = [item["text"] for item in batch]
    encodings = tokenizer(
        texts,
        add_special_tokens=True,
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    n_heads = len(batch[0]["labels"])
    labels = tuple(
        torch.stack([item["labels"][h] for item in batch]) for h in range(n_heads)
    )
    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
    }


def cwe_collate_fn(
    batch: List[Dict[str, Any]],
    tokenizer: Any,
    max_length: int,
) -> Dict[str, Any]:
    """Batched-tokenization collate for CWE multi-label dataset.

    Each ``batch[i]`` is ``{"text": str, "labels": Tensor[num_classes]}``.
    """
    texts = [item["text"] for item in batch]
    encodings = tokenizer(
        texts,
        add_special_tokens=True,
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = torch.stack([item["labels"] for item in batch])
    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Mixed-precision helpers
# ---------------------------------------------------------------------------

def autocast_context(
    device: torch.device, enabled: bool = True
) -> "contextlib.AbstractContextManager":
    """Return an autocast context appropriate for ``device``.

    - CUDA: ``bf16`` on capability >= 8.0 (Ampere+), else ``fp16``.
    - CPU: nullcontext — autocast overhead doesn't pay for itself at this
      model size.
    - MPS: nullcontext. ``torch.autocast(device_type="mps")`` works for
      forward on torch 2.12, but the official AMP docs (torch 2.11/2.12)
      only document CUDA + CPU; backward + GradScaler integration on MPS
      is not declared production-ready. Gated off pending evidence.
    """
    if not enabled:
        return contextlib.nullcontext()

    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        return torch.amp.autocast(device_type="cuda", dtype=dtype)

    return contextlib.nullcontext()


def make_grad_scaler(device: torch.device) -> "torch.amp.GradScaler":
    """Create a GradScaler. Enabled only for CUDA + fp16 (sub-Ampere).

    Returns a disabled scaler on CPU / MPS / CUDA-bf16 — calling
    ``scale``/``step``/``update`` becomes a no-op so the trainer can use
    a single code path.
    """
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        enabled = major < 8
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.amp.GradScaler(device.type, enabled=False)


# ---------------------------------------------------------------------------
# Token-length histogram diagnostic
# ---------------------------------------------------------------------------

def write_token_length_diagnostic(
    tokenizer: Any,
    texts: List[str],
    output_path: Path,
    model_label: str,
) -> Dict[str, Any]:
    """Tokenize ``texts`` (no padding, no truncation) and emit a length report.

    Prints percentiles + per-cutoff truncation fractions, and saves the
    raw lengths to ``output_path`` for post-hoc analysis. Diagnostic only:
    callers should not change ``max_length`` based on this in the same
    training run.
    """
    if not texts:
        print(f"[{model_label}-tok-lengths] no texts provided; skipping")
        return {}

    encodings = tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )
    lengths = [len(ids) for ids in encodings["input_ids"]]
    arr = np.asarray(lengths, dtype=int)

    percentiles = {
        f"p{p}": int(np.percentile(arr, p)) for p in (50, 75, 90, 95, 99, 100)
    }
    truncation_fracs = {
        str(c): float((arr > c).mean()) for c in (128, 192, 256, 384, 512)
    }

    pct_str = " ".join(f"{k}={v}" for k, v in percentiles.items())
    print(f"[{model_label}-tok-lengths] n={len(arr)} {pct_str}")
    for cutoff, frac in truncation_fracs.items():
        print(
            f"[{model_label}-tok-lengths] frac_truncated@{cutoff}={frac:.4f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "count": int(len(arr)),
        "percentiles": percentiles,
        "truncation_fractions": truncation_fracs,
        "lengths": [int(x) for x in lengths],
    }
    with open(output_path, "w") as f:
        json.dump(payload, f)
    return payload


# ---------------------------------------------------------------------------
# Optimizer + class-weight + cross-head-loss helpers
# ---------------------------------------------------------------------------

# Parameter-name substrings excluded from weight decay (BERT-paper convention).
_NO_DECAY_KEYS: Tuple[str, ...] = ("bias", "LayerNorm.weight")


def make_optimizer_param_groups(
    model: nn.Module, weight_decay: float
) -> List[Dict[str, Any]]:
    """Build AdamW parameter groups with biases + LayerNorm carved out of decay."""
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(key in name for key in _NO_DECAY_KEYS):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": float(weight_decay)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def compute_inverse_frequency_class_weights(
    counts: Sequence[int], clip: float
) -> torch.Tensor:
    """Inverse-frequency class weights, clipped at ``clip``.

    weight_i = N / (K * count_i), where N = sum(counts), K = len(counts).
    Zero-count classes fall back to the clip cap.
    """
    counts_arr = np.asarray(list(counts), dtype=np.float64)
    if counts_arr.ndim != 1 or counts_arr.size == 0:
        raise ValueError("counts must be a 1-D non-empty sequence")
    n = float(counts_arr.sum())
    k = float(counts_arr.size)
    weights = np.full_like(counts_arr, fill_value=float(clip), dtype=np.float64)
    nonzero = counts_arr > 0
    weights[nonzero] = n / (k * counts_arr[nonzero])
    np.minimum(weights, float(clip), out=weights)
    return torch.tensor(weights, dtype=torch.float32)


def weighted_head_loss(
    losses: Sequence[torch.Tensor],
    num_labels_list: Sequence[int],
    strategy: str,
) -> torch.Tensor:
    """Combine per-head losses across CVSS heads.

    ``strategy='equal'`` sums losses; ``strategy='log_k'`` weights each
    head's loss by ``1 / log(K_i)`` where ``K_i`` is the head's class count.
    """
    if len(losses) != len(num_labels_list):
        raise ValueError(
            f"weighted_head_loss: {len(losses)} losses vs "
            f"{len(num_labels_list)} num_labels entries"
        )
    if strategy == "equal":
        return sum(losses)
    if strategy == "log_k":
        weights = [1.0 / math.log(int(k)) for k in num_labels_list]
        return sum(w * l for w, l in zip(weights, losses))
    raise ValueError(f"Unknown head_loss_weighting strategy: {strategy!r}")


def tune_cwe_thresholds(
    pred_probs: np.ndarray,
    labels: np.ndarray,
    grid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pick per-class thresholds on val to maximize per-class F1.

    Returns an array of shape ``(num_classes,)``. Classes with no positives
    in val fall back to 0.5.
    """
    if grid is None:
        grid = np.arange(0.05, 0.51, 0.05)
    grid = np.asarray(grid, dtype=float)

    pred_probs = np.asarray(pred_probs)
    labels = np.asarray(labels)
    num_classes = pred_probs.shape[1]
    thresholds = np.full(num_classes, 0.5, dtype=float)

    for c in range(num_classes):
        y_true = labels[:, c].astype(int)
        if y_true.sum() == 0:
            continue
        probs = pred_probs[:, c]
        best_f1 = -1.0
        best_thr = 0.5
        for thr in grid:
            y_pred = (probs >= thr).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)
        thresholds[c] = best_thr

    return thresholds
